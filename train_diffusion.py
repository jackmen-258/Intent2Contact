import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import random

import wandb
import torch
from torch.optim import AdamW
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from copy import deepcopy

from lib.datasets.oishape_dataset import OIShape
from lib.diffusion.latent_diffusion_model import LatentHandDiffusion
from lib.contact.hand_object import HandObject

from lib.utils.config import get_config
from lib.utils.utils import cycle
from manotorch.manolayer import ManoLayer, MANOOutput
from lib.datasets.utils import CENTER_IDX

from torch.utils.data import WeightedRandomSampler


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.copy_(v * d + msd[k].detach() * (1. - d))

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.ema.load_state_dict(state_dict, strict=strict)


class PoseDisturber(torch.nn.Module):
    def __init__(self, tsl_sigma=0.02, pose_sigma=0.2, root_rot_sigma=0.004):
        super().__init__()
        self.tsl_sigma = float(tsl_sigma)
        self.pose_sigma = float(pose_sigma)
        self.root_rot_sigma = float(root_rot_sigma)

    def forward(self, hand_pose, hand_transl):
        batch_size = hand_pose.shape[0]
        device = hand_pose.device
        dtype = hand_pose.dtype

        hand_root_pose = hand_pose[:, :3]
        hand_rel_pose = hand_pose[:, 3:]

        hand_transl = hand_transl + torch.randn(batch_size, 3, device=device, dtype=dtype) * self.tsl_sigma
        hand_root_pose = hand_root_pose + torch.randn(batch_size, 3, device=device, dtype=dtype) * self.root_rot_sigma
        hand_rel_pose = hand_rel_pose + torch.randn(batch_size, hand_rel_pose.shape[1], device=device, dtype=dtype) * self.pose_sigma
        hand_pose = torch.cat([hand_root_pose, hand_rel_pose], dim=1)

        return hand_pose, hand_transl


class LatentHandDiffusionTrainer(object):
    def __init__(self, opt, diffusion_model, stage='vae', train_batch_size=64, 
                 train_num_steps=5000, save_and_val_every=500,
                 results_folder='runs-latent/weights', use_wandb=False, 
                 w_kl=0.1, w_chamfer=10.0):
        super().__init__()
        self.opt = opt
        
        assert stage in ('vae', 'diffusion', 'refine')
        self.stage = stage
        self.device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(config=vars(opt), project=opt.wandb_pj_name, 
                      entity=opt.entity, name=opt.exp_name, dir=opt.exp_dir)

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.step = 0
        self.results_folder = results_folder
        self.save_and_val_every = save_and_val_every

        self.w_kl = w_kl
        self.w_chamfer = w_chamfer

        self.cfg = get_config(config_file=opt.oishape_cfg)
        self.prep_dataloader(self.cfg)

        self.model = diffusion_model
        self.hand_object = HandObject(self.device)

        # MANO Layer for reconstruction
        self.mano_layer = ManoLayer(
            center_idx=CENTER_IDX,
            mano_assets_root="assets/mano_v1_2"
        ).to(self.device)

        self.use_ema = getattr(opt, 'use_ema', False)
        self.ema_start = getattr(opt, 'ema_start', 0)
        self.ema = None
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=getattr(opt, 'ema_decay', 0.999), device=self.device)
            print(f"[INFO] EMA enabled: decay={getattr(opt, 'ema_decay', 0.999)}, start_step={self.ema_start}")

        self.setup_stage_parameters()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.use_latent_norm = getattr(opt, 'use_latent_norm', True)
        self.pose_disturber = PoseDisturber(
            tsl_sigma=getattr(opt, "refine_tsl_sigma", 0.02),
            pose_sigma=getattr(opt, "refine_pose_sigma", 0.2),
            root_rot_sigma=getattr(opt, "refine_root_rot_sigma", 0.004),
        ).to(self.device)

    @staticmethod
    def _matches_prefix(key, prefixes):
        return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)

    def _load_compatible_state_dict(self, target_module, state_dict, only_prefixes=None, label="model"):
        current_state = target_module.state_dict()
        filtered_state = {}
        skipped_keys = []
        selected_keys = 0

        for key, value in state_dict.items():
            if only_prefixes is not None and not self._matches_prefix(key, only_prefixes):
                continue
            selected_keys += 1
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped_keys.append(key)

        missing_keys, unexpected_keys = target_module.load_state_dict(filtered_state, strict=False)
        if only_prefixes is not None:
            missing_keys = [key for key in missing_keys if self._matches_prefix(key, only_prefixes)]

        if skipped_keys:
            print(f"[INFO] Skipped {len(skipped_keys)} incompatible {label} keys during load.")
        if unexpected_keys:
            print(f"[INFO] Unexpected {label} keys ignored during load: {len(unexpected_keys)}")
        if missing_keys:
            print(f"[INFO] Missing {label} keys after compatible load: {len(missing_keys)}")
        if only_prefixes is not None:
            print(f"[INFO] Loaded {len(filtered_state)}/{selected_keys} selected {label} keys.")

    def _load_compatible_model_state(self, state_dict, only_prefixes=None, label="model"):
        self._load_compatible_state_dict(self.model, state_dict, only_prefixes=only_prefixes, label=label)

    def _load_compatible_ema_state(self, state_dict, only_prefixes=None, label="EMA"):
        if self.ema is None:
            return
        self._load_compatible_state_dict(self.ema.ema, state_dict, only_prefixes=only_prefixes, label=label)


    def setup_stage_parameters(self):
        def freeze_module(module, freeze=True):
            for param in module.parameters():
                param.requires_grad = not freeze

        if self.stage == 'vae':
            freeze_module(self.model.encoder, freeze=False)
            freeze_module(self.model.decoder, freeze=False) 

            freeze_module(self.model.denoise_fn, freeze=True)
            freeze_module(self.model.obj_pointnet, freeze=True)
            freeze_module(self.model.intent_embed, freeze=True)
            freeze_module(self.model.fusion, freeze=True)
            
            print("Stage 1: Training VAE (Encoder + Decoder)")
            
        elif self.stage == 'diffusion':
            freeze_module(self.model.encoder, freeze=True)
            freeze_module(self.model.decoder, freeze=True)
            freeze_module(self.model.refine_net, freeze=True)

            freeze_module(self.model.denoise_fn, freeze=False)
            freeze_module(self.model.obj_pointnet, freeze=False)
            freeze_module(self.model.intent_embed, freeze=False)
            freeze_module(self.model.fusion, freeze=False)

            print("Stage 2: Training Diffusion Model (VAE frozen)")
        elif self.stage == 'refine':
            freeze_module(self.model.encoder, freeze=True)
            freeze_module(self.model.decoder, freeze=True)
            freeze_module(self.model.denoise_fn, freeze=True)
            freeze_module(self.model.obj_pointnet, freeze=True)
            freeze_module(self.model.intent_embed, freeze=True)
            freeze_module(self.model.fusion, freeze=True)
            freeze_module(self.model.refine_net, freeze=False)
            print("Stage 3: Training GrabNet-style RefineNet (diffusion frozen)")

    def setup_optimizer(self):
        if self.stage == 'vae':
            param_groups = [
                {'params': self.model.encoder.parameters(), 'lr': 1e-4},
                {'params': self.model.decoder.parameters(), 'lr': 1e-4}
            ]
            
        elif self.stage == 'diffusion':
            param_groups = [
                {'params': self.model.denoise_fn.parameters(), 'lr': 1e-4},
                {'params': self.model.obj_pointnet.parameters(), 'lr': 5e-5},
                {'params': self.model.intent_embed.parameters(), 'lr': 1e-4},
                {'params': self.model.fusion.parameters(), 'lr': 5e-5}
            ]
        else:
            param_groups = [
                {'params': self.model.refine_net.parameters(), 'lr': 1e-4}
            ]

        return AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
    
    def setup_scheduler(self):
        if self.stage == 'vae':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_num_steps,
                eta_min=1e-5,
            )
        elif self.stage == 'diffusion':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.train_num_steps // 3,
                T_mult=1,
                eta_min=1e-6
            )
        else:
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_num_steps,
                eta_min=1e-6,
            )

    def prep_dataloader(self, cfg):
        train_cfg = deepcopy(cfg.DATASET.TRAIN)
        val_cfg = deepcopy(cfg.DATASET.VAL)

        if self.stage == "refine":
            train_cfg.N_SAMPLES = int(getattr(self.opt, "refine_n_obj_points", 4096))
            val_cfg.N_SAMPLES = int(getattr(self.opt, "refine_n_obj_points", 4096))
            train_cfg.AUG_RIGID_P = 0.0
            val_cfg.AUG_RIGID_P = 0.0

        train_dataset = OIShape(train_cfg)
        val_dataset = OIShape(val_cfg)

        self.ds = train_dataset
        self.val_ds = val_dataset

        # ==============================
        # ✅ Intent-balanced sampling
        # ==============================
        use_intent_balanced = getattr(self.opt, "intent_balanced", True)
        num_intents = getattr(self.opt, "num_intents", 4)

        train_loader_kwargs = dict(
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            drop_last=True,   # 建议开：便于 batch 统计稳定
        )

        if use_intent_balanced and self.stage == "diffusion":
            print("[INFO] Building intent-balanced sampler for training...")

            # 1) collect intent_id for each sample
            intents = []
            for i in range(len(self.ds)):
                item = self.ds[i]
                iid = item["intent_id"]
                # iid could be int / numpy / torch tensor
                if isinstance(iid, torch.Tensor):
                    iid = int(iid.item())
                elif hasattr(iid, "item"):
                    iid = int(iid.item())
                else:
                    iid = int(iid)
                intents.append(iid)

            intents = np.array(intents, dtype=np.int64)

            # 2) count
            counts = np.bincount(intents, minlength=num_intents).astype(np.float64)
            print(f"[INFO] Train intent counts: {counts.tolist()}")

            # avoid div0
            counts[counts == 0] = 1.0

            # 3) per-sample weight = 1 / count[intent]
            sample_weights = 1.0 / counts[intents]
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

            # 4) sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),  # one "epoch" length
                replacement=True
            )

            self.train_sampler = sampler
            self.dl = cycle(data.DataLoader(
                self.ds,
                sampler=self.train_sampler,
                shuffle=False,  # sampler and shuffle are mutually exclusive
                **train_loader_kwargs
            ))
            print("[INFO] Intent-balanced sampler enabled (replacement=True).")
        else:
            self.train_sampler = None
            self.dl = cycle(data.DataLoader(
                self.ds,
                shuffle=True,
                **train_loader_kwargs
            ))

        # val loader keep normal
        self.val_dl = cycle(data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=False
        ))

    def params_to_verts(self, mano_params):
        pose = mano_params[:, :48]
        trans = mano_params[:, 48:51]
        shape = mano_params[:, 51:]

        mano_output: MANOOutput = self.mano_layer(pose, shape)
        verts = mano_output.verts + trans.unsqueeze(1)
        return verts

    def params_to_contact(self, mano_params, obj_verts, obj_vn):
        hand_verts = self.params_to_verts(mano_params)
        
        data_out = self.hand_object.forward(hand_verts, obj_verts, obj_vn)
        contact_map = data_out["contacts_object"]
        return contact_map

    def compute_latent_statistics(self):
        if not self.use_latent_norm:
            print("[INFO] Latent normalization disabled")
            return

        print("[INFO] Computing latent statistics for normalization...")
        
        train_loader = data.DataLoader(
            self.ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True, 
            num_workers=4
        )
        
        self.model.compute_latent_stats(train_loader)

    def sample_refine_coarse_params(self, hand_pose, hand_trans, hand_shape):
        perturbed_pose, perturbed_trans = self.pose_disturber(hand_pose, hand_trans)
        return torch.cat([perturbed_pose, perturbed_trans, hand_shape], dim=-1)

    def train_step_vae(self, grasp):
        obj_verts = grasp["obj_verts"].to(self.device)
        hand_pose = grasp["hand_pose"].to(self.device)
        hand_trans = grasp["hand_tsl"].to(self.device)
        hand_shape = grasp["hand_shape"].to(self.device)
        hand_verts = grasp["hand_verts"].to(self.device)
        mano_params = torch.cat([hand_pose, hand_trans, hand_shape], dim=-1)

        loss_dict = self.model.compute_vae_loss(
            mano_params,
            obj_verts=obj_verts,
            hand_verts_gt=hand_verts,
        )

        weight_kl = self.kl_coeff(
            step=self.step,
            total_step=self.train_num_steps,
            constant_step=0,
            min_kl_coeff=1e-7,
            max_kl_coeff=self.w_kl
        )

        total_loss = (
            loss_dict['recon_loss'] + 
            weight_kl * loss_dict['kl_loss'] + 
            self.w_chamfer * loss_dict['cd_loss'] +
            loss_dict['vae_geom_loss']
        )
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def train_step_diffusion(self, grasp):
        if not self.model.latent_stats_computed and self.model.use_latent_norm:
            print("[WARNING] Latent stats not computed, computing now...")
            self.compute_latent_statistics() 

        obj_verts = grasp["obj_verts"].to(self.device)
        obj_vn = grasp["obj_vn"].to(self.device)
        hand_pose = grasp["hand_pose"].to(self.device)
        hand_trans = grasp["hand_tsl"].to(self.device)
        hand_shape = grasp["hand_shape"].to(self.device)
        intent_id = grasp["intent_id"].to(self.device)

        mano_params = torch.cat([hand_pose, hand_trans, hand_shape], dim=-1)
        
        diffusion_loss = self.model(
            mano_params, 
            obj_verts, 
            obj_vn, 
            intent_id,
        )

        total_loss = diffusion_loss
        loss_dict = {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
        }
        return total_loss, loss_dict

    def train_step_refine(self, grasp, split='train'):
        obj_verts = grasp["obj_verts"].to(self.device)
        hand_pose = grasp["hand_pose"].to(self.device)
        hand_trans = grasp["hand_tsl"].to(self.device)
        hand_shape = grasp["hand_shape"].to(self.device)
        hand_verts = grasp["hand_verts"].to(self.device)

        gt_mano_params = torch.cat([hand_pose, hand_trans, hand_shape], dim=-1)
        coarse_mano_params = self.sample_refine_coarse_params(hand_pose, hand_trans, hand_shape)

        total_loss, loss_dict = self.model.compute_refine_loss(
            coarse_mano_params=coarse_mano_params,
            gt_mano_params=gt_mano_params,
            obj_verts=obj_verts,
            hand_verts_gt=hand_verts,
        )
        return total_loss, loss_dict

    def train_step(self, grasp, split='train'):
        if self.stage == 'vae':
            return self.train_step_vae(grasp)
        elif self.stage == 'diffusion':
            return self.train_step_diffusion(grasp)
        elif self.stage == 'refine':
            return self.train_step_refine(grasp, split=split)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def evaluate(self, iters=10):
        if self.use_ema:
            orig_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict(), strict=False)

        self.model.eval()
        total_loss_sum = 0.
        loss_components = {}
        
        with torch.no_grad():
            for _ in range(iters):
                grasp = next(self.val_dl)
                total_loss, loss_dict = self.train_step(grasp, split='val')
                total_loss_sum += total_loss.item()
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.
                    loss_components[key] += value.item() if hasattr(value, 'item') else value

        avg_loss = total_loss_sum / iters
        avg_loss_components = {key: val / iters for key, val in loss_components.items()}
        
        if self.use_wandb:
            log_dict = {f"Val/{key}": val for key, val in avg_loss_components.items()}
            wandb.log(log_dict, step=self.step)

        if self.use_ema:
            self.model.load_state_dict(orig_state, strict=False)
            
        return avg_loss, avg_loss_components

    def save(self, step):
        data = {
            'step': self.step,
            'stage': self.stage,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        if self.use_ema and self.ema is not None:
            data['ema'] = self.ema.state_dict()

        torch.save(data, os.path.join(self.results_folder, f'model-{self.stage}-{step}.pt'))
        print(f"Saved checkpoint for stage {self.stage} at step {step}")

    def load(self, step, stage=None, only_vae=False):
        if stage is None:
            stage = self.stage
        
        checkpoint_path = os.path.join(self.results_folder, f'model-{stage}-{step}.pt')
        data = torch.load(checkpoint_path)

        self.step = data['step']

        if only_vae and stage == 'vae':
            self.model.encoder.load_state_dict(data['model'], strict=False)
            self.model.decoder.load_state_dict(data['model'], strict=False)
        else:
            self._load_compatible_model_state(data['model'])
            if stage == 'diffusion' and hasattr(self.model, 'latent_stats_computed'):
                self.model.latent_stats_computed = True

        if stage == self.stage:
            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])

        if self.use_ema and 'ema' in data and data['ema'] is not None:
            if self.ema is None:
                self.ema = ModelEMA(self.model, decay=getattr(self.opt, 'ema_decay', 0.999), device=self.device)
            self.ema.load_state_dict(data['ema'], strict=False)
            print(f"[INFO] Loaded EMA from checkpoint {checkpoint_path}")

    def train(self):
        if self.stage == 'diffusion' and not self.model.latent_stats_computed and self.model.use_latent_norm:
            self.compute_latent_statistics()

        init_step = self.step
        print(f"[INFO] Stage: {self.stage}")
        print(f"[INFO] Total number of samples: {len(self.ds)}")
        if self.stage == 'refine':
            print(
                "[INFO] Refine coarse source: GT + PoseDisturber "
                f"(tsl_sigma={self.pose_disturber.tsl_sigma}, "
                f"pose_sigma={self.pose_disturber.pose_sigma}, "
                f"root_rot_sigma={self.pose_disturber.root_rot_sigma})"
            )

        torch.autograd.set_detect_anomaly(True)
        
        for idx in range(init_step, self.train_num_steps + 1):
            self.model.train()
            self.optimizer.zero_grad()

            grasp = next(self.dl)
            total_loss, loss_dict = self.train_step(grasp)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group['params']], max_norm=5.0)
            
            self.optimizer.step()
            self.scheduler.step()

            if self.use_ema and self.step >= self.ema_start:
                self.ema.update(self.model)

            if self.use_wandb:
                log_dict = {f"Train/{key}": val.item() if hasattr(val, 'item') else val 
                           for key, val in loss_dict.items()}
                wandb.log(log_dict, step=self.step)

            if idx % 50 == 0:
                loss_str = ", ".join([f"{key}={val.item():.4f}" if hasattr(val, 'item') else f"{key}={val:.4f}" 
                                    for key, val in loss_dict.items()])
                print(f"Step: {idx}, Stage: {self.stage}, {loss_str}")

            if self.step != 0 and self.step % self.save_and_val_every == 0:
                _, avg_val_components = self.evaluate(iters=10)
                loss_str = ", ".join([f"{key}={val:.4f}" for key, val in avg_val_components.items()])
                print(f"[Val @ step={self.step}] Stage: {self.stage}, {loss_str}")
                self.save(self.step)

            self.step += 1

        print(f'Training complete for stage {self.stage}')
        if self.use_wandb:
            wandb.run.finish()

    @staticmethod
    def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
        return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    

def run_train(opt):
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(opt.exp_dir)
    wdir = exp_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    diffusion_model = LatentHandDiffusion(
        params_dim=61,
        latent_dim=opt.latent_dim, 
        obj_dim=opt.obj_dim,
        intent_dim=opt.intent_dim,
        fusion_dim=opt.fusion_dim,
        fusion_type=opt.fusion_type,
        disable_intent=opt.disable_intent,
        vae_geom_h2o_weight=opt.w_vae_h2o,
        vae_geom_o2h_weight=opt.w_vae_o2h,
    )
    diffusion_model.to(device)

    trainer = LatentHandDiffusionTrainer(
        opt,
        diffusion_model,
        stage=opt.stage,
        train_batch_size=opt.batch_size, 
        train_num_steps=opt.train_num_steps,
        save_and_val_every=opt.save_and_val_every,  # ✅ 使用命令行参数
        results_folder=str(wdir),
        use_wandb=True,
        w_kl=opt.w_kl,
        w_chamfer=opt.w_chamfer
    )
    
    if opt.resume_step > 0:
        print(f"=> Resume from step {opt.resume_step} (stage={opt.stage})")
        trainer.load(opt.resume_step, stage=opt.stage)
    elif opt.stage == 'diffusion' and opt.vae_checkpoint_step > 0:
        print(f"=> Load VAE checkpoint from step {opt.vae_checkpoint_step}")
        trainer.load(opt.vae_checkpoint_step, stage='vae', only_vae=False)
        trainer.step = 0
    elif opt.stage == 'refine' and opt.diffusion_checkpoint_step > 0:
        print(f"[INFO] Ignoring diffusion checkpoint step {opt.diffusion_checkpoint_step} for refine training.")

    trainer.train()
    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs-ablation', help='project root path')
    parser.add_argument('--exp_name', default='', help='save to project path')
    parser.add_argument('--device', default="1", help='cuda device')
    parser.add_argument('--wandb_pj_name', type=str, default='intent2contact', help='wandb project name')
    parser.add_argument('--entity', default='', help='W&B entity')
    parser.add_argument('--seed', type=int, default=3407, help='random seed for reproducibility')

    # Training options
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--train_num_steps', type=int, default=30000, help='global training steps')

    parser.add_argument('--oishape_cfg', default='config/all_cate.yaml', help='the path to oishape config')

    parser.add_argument('--save_and_val_every', type=int, default=5000, 
                       help='save checkpoint and validate every N steps')

    parser.add_argument('--latent_dim', type=int, default=64, help='latent dim for mano params')
    parser.add_argument('--obj_dim', type=int, default=64, help='point feats dim')
    parser.add_argument('--intent_dim', type=int, default=64, help='intent embedding dim')
    parser.add_argument('--fusion_dim', type=int, default=128, help='fusion layer dim')

    # Loss weight options
    parser.add_argument('--w_kl', type=float, default=0.01, help='Weight for KL loss')
    parser.add_argument('--w_chamfer', type=float, default=0.1, help='Weight for Chamfer distance loss')
    parser.add_argument('--w_vae_h2o', type=float, default=5.0, help='Weight for VAE hand-to-object distance matching')
    parser.add_argument('--w_vae_o2h', type=float, default=5.0, help='Weight for VAE object-to-hand signed distance matching')

    # EMA options
    parser.add_argument('--use_ema', action='store_true', help='enable EMA for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay (e.g., 0.999~0.9999)')
    parser.add_argument('--ema_start', type=int, default=500, help='start EMA update after this step')

    # Stage option and model selection
    parser.add_argument('--stage', type=str, default='vae', choices=['vae', 'diffusion', 'refine'], help='training stage')
    parser.add_argument('--resume_step', type=int, default=0, help='resume training from this step')
    parser.add_argument('--vae_checkpoint_step', type=int, default=20000, help='VAE checkpoint step for diffusion stage')
    parser.add_argument('--diffusion_checkpoint_step', type=int, default=0,
                        help='Deprecated for refine training; refine now uses GT+disturbance instead of diffusion coarse cache')

    # Latent normalization
    parser.add_argument('--use_latent_norm', action='store_true', help='enable latent variable normalization')

    # for ablation study
    parser.add_argument('--fusion_type', type=str, default='bi_attn', help='type of fusion module')
    parser.add_argument('--disable_intent', action='store_true', help='disable intent conditioning')

    # Intent-balanced sampling options
    parser.add_argument('--intent_balanced', action='store_true',
                    help='enable intent-balanced sampling (WeightedRandomSampler) in diffusion stage')
    parser.add_argument('--num_intents', type=int, default=4, help='number of intent classes')
    parser.add_argument('--refine_n_obj_points', type=int, default=4096, help='number of object surface points for refine stage')
    parser.add_argument('--refine_tsl_sigma', type=float, default=0.02, help='translation noise sigma for GT+disturbed refine training')
    parser.add_argument('--refine_pose_sigma', type=float, default=0.2, help='relative hand pose noise sigma for GT+disturbed refine training')
    parser.add_argument('--refine_root_rot_sigma', type=float, default=0.004, help='global hand rotation noise sigma for GT+disturbed refine training')
    parser.add_argument('--refine_cache_sampler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Deprecated; retained for backward compatibility')
    parser.add_argument('--refine_cache_ddim_steps', type=int, default=50,
                        help='Deprecated; retained for backward compatibility')


    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.exp_dir = os.path.join(opt.project, opt.exp_name)

    torch.cuda.set_device(int(opt.device))
    torch.cuda.empty_cache()

    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    run_train(opt)
