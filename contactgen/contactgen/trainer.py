import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from .utils.utils import makelogger, makepath
from .model import ContactGenModel
from .hand_object import HandObject
from .datasets.grab_dataset import Grab
from .datasets.oishape_dataset import OIShape


class Trainer:

    def __init__(self, cfg, logger=None):
        self.dtype = torch.float32
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)

        if logger:
            self.logger = logger
        else:
            self.logger = makelogger(makepath(os.path.join(cfg.work_dir, 'train.log'), isfile=True)).info

        # 创建 TensorBoard 的日志目录并初始化 SummaryWriter
        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        self.logger('Started training %s' % (starttime))
        self.logger('tensorboard --logdir=%s' % summary_logdir)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")

        # 记录 GPU 型号
        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        if use_cuda:
            self.logger('Using 1 CUDA cores [%s] for training!' % (gpu_brand))

        self.logger(cfg)

        self.cfg = cfg
        # 根据配置加载数据
        self.load_data(cfg)

        self.model = ContactGenModel(cfg).to(self.device)
        # HandObject 用于生成 GT contact/part/direction map
        self.hand_object = HandObject(self.device)
        self.model.cfg = cfg

        # 初始化 KL 散度系数，KL 散度损失权重，重建损失权重
        self.max_kl_coef = cfg.kl_coef
        self.weight_kl = None
        self.weight_rec = None
        
        # 通过 named_parameters() 方法获取模型的所有参数，并创建一个参数列表 vars_net
        # 计算模型中所有需要梯度的参数的总数
        vars_net = [var[1] for var in self.model.named_parameters()]
        net_n_params = sum(p.numel() for p in vars_net if p.requires_grad)
        self.logger('Total Trainable Parameters is %2.2f M.' % ((net_n_params) * 1e-6))

        # 创建 Adam 优化器和使用余弦退火策略的学习率调度器
        self.optimizer_net = optim.Adam(vars_net, lr=cfg.base_lr, weight_decay=cfg.reg_coef)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_net,
                                                                           T_0=cfg.n_epochs,
                                                                           eta_min=1e-4)
        # 初始化起始周期为 0
        self.start_epoch = 0

        # 如果 checkpoint 不为空，加载模型权重
        if cfg.checkpoint is not None:
            checkpoint = torch.load(cfg.checkpoint, map_location=self.device)
            weight = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self._get_net_model().load_state_dict(weight, strict=True)
            self.logger('Load model from %s' % cfg.checkpoint)
        
        # 设置已完成周期数为起始周期
        self.epoch_completed = self.start_epoch

    def load_data(self, cfg):
        # drop_last：如果数据在最后一个批次不完整，是否丢弃它
        train_kwargs = {'num_workers': cfg.n_workers,
                  'shuffle': True,
                  'drop_last': True
                  }
        
        val_kwargs = {
            'num_workers': cfg.n_workers,
            'shuffle': False,
            'drop_last': False,
        }
        
        dataset = getattr(cfg, 'dataset', 'grab')

        if dataset == 'grab':
            ds_train = Grab(ds_name="train", n_samples=cfg.n_samples)
            ds_val = Grab(ds_name="val", n_samples=cfg.n_samples)
        elif dataset == 'oishape':
            ds_train = OIShape(split='train')
            ds_val = OIShape(split='val')

        self.ds_train = DataLoader(ds_train, batch_size=cfg.batch_size, **train_kwargs)
        self.ds_val = DataLoader(ds_val, batch_size=min(len(ds_val), cfg.batch_size), **val_kwargs)


    def _get_net_model(self):
        # 当直接在单个 GPU 上训练模型，没有使用 DataParallel 时，self.model 就是模型
        # 当使用 DataParallel 来在多个 GPU 上训练模型时，self.model 实际上是一个 DataParallel 对象，它封装了原始模型
        # 在这种情况下，使用 self.model.module 可以访问到这个内部模型
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    
    # save_net 方法用于保存模型快照(E0_net.pt, E1000_net.pt, E2000_net.pt)
    def save_net(self):
        torch.save(self.model.module.state_dict()
                   if isinstance(self.model, torch.nn.DataParallel)
                   else self.model.state_dict(), self.cfg.checkpoint)

    # save_ckp 方法用于保存每次epoch后的模型权重(checkpoint.pt)
    def save_ckp(self, state, checkpoint_dir):
        f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        torch.save(state, f_path)

    def load_ckp(self, checkpoint, model):
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train(self, epoch_num):
        # 参数 epoch_num：表示当前的训练周期编号

        self.model.train()
        save_every_it = len(self.ds_train) // self.cfg.log_every_epoch

        # 初始化一个字典，用于存储网络训练过程中的损失
        train_loss_dict_net = {}
        # 启用自动求导的异常检测，以帮助捕获梯度计算中的错误
        torch.autograd.set_detect_anomaly(True)

        n_iters = len(self.ds_train)
        
        # it: __getitem__函数参数中的 item
        # input: self.ds_train[it]
        for it, input in enumerate(self.ds_train):
            
            # 将数据项移动到 GPU 上
            input = {k: input[k].to(self.device) for k in input.keys()}
            # 调用 HandObject 类的 forward 方法获取 GT 数据
            dorig = self.hand_object.forward(input['hand_verts'], input['hand_frames'],
                                             input['obj_verts'], input['obj_vn'])
            self.optimizer_net.zero_grad()

            if self.fit_net:
                # 预测
                drec_net = self.model(**dorig)
                
                global_step = epoch_num * n_iters + it
                num_total_iter = self.cfg.n_epochs * n_iters
                # 动态调整 KL 散度损失的权重
                self.weight_kl = self.kl_coeff(step=global_step,
                    total_step=num_total_iter,
                    constant_step=0,
                    min_kl_coeff=1e-7,
                    max_kl_coeff=self.max_kl_coef)
                self.weight_rec = 1.0
                
                # 调用 loss_net 方法计算损失
                # loss_total_net 是所有损失项的总和
                # cur_loss_dict_net 是不同损失项的字典
                loss_total_net, cur_loss_dict_net = self.loss_net(dorig, drec_net)

                loss_total_net.backward()
                self.optimizer_net.step()

                # 将当前迭代的损失累积到总训练损失
                # train_loss_dict_net.get(k, 0.0) 尝试从 train_loss_dict_net 中获取键 k 对应的值，如果 k 不存在，则返回默认值 0.0
                train_loss_dict_net = {k: train_loss_dict_net.get(k, 0.0) + v.item() for k, v in
                                       cur_loss_dict_net.items()}
                # 迭代输出日志消息
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_net = {k: v / (it + 1) for k, v in train_loss_dict_net.items()}
                    train_msg = self.create_loss_message(cur_train_loss_dict_net,
                                                         epoch_num=self.epoch_completed, it=it)

                    self.logger(train_msg)

                self.lr_scheduler.step(epoch_num + it / n_iters)

        # 在每个epoch结束后，计算所有损失项的平均值
        train_loss_dict_net = {k: v / len(self.ds_train) for k, v in train_loss_dict_net.items()}

        return train_loss_dict_net

    #  evaluate 方法用于在验证数据集上评估模型的性能
    def evaluate(self):
        self.model.eval()
        eval_loss_dict_net = {}
        
        with torch.no_grad():
            for input in self.ds_val:
                input = {k: input[k].to(self.device) for k in input.keys()}
                dorig = self.hand_object.forward(input['hand_verts'], input['hand_frames'],
                                                 input['obj_verts'], input['obj_vn'])
                
                drec_net = self.model(**dorig)
                _, cur_loss_dict_net = self.loss_net(dorig, drec_net)

                eval_loss_dict_net = {k: eval_loss_dict_net.get(k, 0.0) + v.item() for k, v in
                                        cur_loss_dict_net.items()}
                
            eval_loss_dict_net = {k: v / len(self.ds_val) for k, v in eval_loss_dict_net.items()}
          
        return eval_loss_dict_net

    def loss_net(self, dorig, drec):
        device = dorig['verts_object'].device
        dtype = dorig['verts_object'].dtype
        batch_size = dorig['verts_object'].shape[0]
        # 初始化一个字典，用于存储不同部分的损失
        loss_dict = {}

        # 定义正态分布 p_z 作为 KL 散度损失的先验分布，其均值为 0，标准差为 1
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype))

        # 分别计算 q_z_contact, q_z_part, q_z_uv 和 p_z 的 KL 散度损失
        q_z_contact = torch.distributions.normal.Normal(drec['mean_contact'], drec['std_contact'])
        loss_kl_contact = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_contact, p_z), dim=[1]))

        q_z_part = torch.distributions.normal.Normal(drec['mean_part'], drec['std_part'])
        loss_kl_part = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_part, p_z), dim=[1]))

        q_z_uv = torch.distributions.normal.Normal(drec['mean_uv'], drec['std_uv'])
        loss_kl_uv = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_uv, p_z), dim=[1]))

        # 如果启用了鲁棒 KL 散度，对 KL 散度损失应用鲁棒变换
        if self.cfg.robustkl:
            loss_kl_contact = torch.sqrt(1 + loss_kl_contact ** 2) - 1
            loss_kl_part = torch.sqrt(1 + loss_kl_part ** 2) - 1
            loss_kl_uv = torch.sqrt(1 + loss_kl_uv ** 2) - 1
            
        loss_dict = {'loss_kl_contact': loss_kl_contact,
                    'loss_kl_part': loss_kl_part,
                    'loss_kl_uv': loss_kl_uv}
        
        loss_kl_contact = loss_kl_contact * self.weight_kl
        loss_kl_part = loss_kl_part * self.weight_kl
        loss_kl_uv = loss_kl_uv * self.weight_kl
        
        # 计算重建损失
        target_contact = dorig['contacts_object'].to(device).squeeze(dim=-1)
        weight = 1. + 5. * target_contact
        contact_obj_sub = target_contact - drec['contacts_object'].squeeze(dim=-1)
        contact_obj_weighted = contact_obj_sub * weight
        
        # 使用L1损失计算 contact map 的重建损失
        loss_contact_rec = F.l1_loss(contact_obj_weighted, torch.zeros_like(contact_obj_weighted, device=target_contact.device, dtype=target_contact.dtype), reduction='none')
        loss_contact_rec = self.weight_rec * torch.mean(loss_contact_rec)
        
        # 使用NLL(负对数似然)损失计算 part map 的重建损失
        target_part = dorig['partition_object'].argmax(dim=-1).to(device)
        loss_part_rec = F.nll_loss(input=F.log_softmax(drec['partition_object'], dim=-1).float().permute(0, 2, 1),
                                   target=target_part.long(), reduction='none')
        loss_part_rec = self.weight_rec * 0.5 * torch.mean(weight * loss_part_rec)

        # 使用余弦相似度计算 direction map 的重建损失
        target_uv = dorig['uv_object'].to(device)
        ori_cos = torch.cosine_similarity(drec['uv_object'], target_uv, dim=-1)
        # 使用arccos函数将余弦相似度转换为角度差异，并使用clamp函数确保数值稳定性
        loss_uv_rec = torch.arccos(torch.clamp(ori_cos, min=-0.9999, max=0.9999))
        loss_uv_rec = self.weight_rec * torch.mean(weight * loss_uv_rec)

        loss_dict.update({'loss_contact_rec': loss_contact_rec,
                     'loss_part_rec': loss_part_rec,
                     'loss_uv_rec': loss_uv_rec})
        
        loss_total = loss_kl_contact + loss_kl_part + loss_kl_uv + loss_contact_rec + loss_part_rec + loss_uv_rec
        loss_dict['loss_total'] = loss_total

        # 返回总损失和损失字典
        return loss_total, loss_dict

    # fit 方法用于启动和执行模型的训练过程
    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        self.fit_net = True
        for epoch_num in range(self.start_epoch, n_epochs):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            # 调用 train 方法进行训练，并获取训练损失
            train_loss_dict_net = self.train(epoch_num)
            # 调用 evaluate 方法评估模型在验证集上的性能，并获取验证损失
            eval_loss_dict_net = self.evaluate()

            if self.fit_net:
                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_net, epoch_num=self.epoch_completed, it=len(self.ds_val))

                    self.cfg.checkpoint = makepath(os.path.join(self.cfg.work_dir, 'checkpoints', 'E%03d_net.pt' % (self.epoch_completed)), isfile=True)
                    if not epoch_num % self.cfg.snapshot_every_epoch:
                        self.save_net()
                    self.logger(eval_msg + ' ** ')

                    self.swriter.add_scalars('loss_net/kl_loss',
                                             {
                                                 'train_loss_kl_contact': train_loss_dict_net['loss_kl_contact'],
                                                 'evald_loss_kl_contact': eval_loss_dict_net['loss_kl_contact'],
                                                 'train_loss_kl_part': train_loss_dict_net['loss_kl_part'],
                                                 'evald_loss_kl_part': eval_loss_dict_net['loss_kl_part'],
                                                 'train_loss_kl_uv': train_loss_dict_net['loss_kl_uv'],
                                                 'evald_loss_kl_uv': eval_loss_dict_net['loss_kl_uv'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/total_rec_loss',
                                             {
                                                 'train_loss_total': train_loss_dict_net['loss_total'],
                                                 'evald_loss_total': eval_loss_dict_net['loss_total'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/contact_rec_loss',
                                             {
                                                 'train_loss_contact_rec': train_loss_dict_net[
                                                     'loss_contact_rec'],
                                                 'evald_loss_contact_rec': eval_loss_dict_net[
                                                     'loss_contact_rec'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/part_rec_loss',
                                             {
                                                 'train_loss_part_rec': train_loss_dict_net[
                                                     'loss_part_rec'],
                                                 'evald_loss_part_rec': eval_loss_dict_net[
                                                     'loss_part_rec'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/uv_rec_loss',
                                             {
                                                 'train_loss_uv_rec': train_loss_dict_net[
                                                     'loss_uv_rec'],
                                                 'evald_loss_uv_rec': eval_loss_dict_net[
                                                     'loss_uv_rec'],
                                             },
                                             self.epoch_completed)
            self.epoch_completed += 1

            checkpoint = {
                'epoch': epoch_num + 1,
                'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
            }
            self.save_ckp(checkpoint, self.cfg.work_dir)

            if not self.fit_net:
                self.logger('Stopping the training!')
                break

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s!\n' % (endtime - starttime))

    @staticmethod
    def create_loss_message(loss_dict, epoch_num=0, it=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return 'E%03d - It %05d : [T:%.2e] - [%s]' % (epoch_num, it, loss_dict['loss_total'], ext_msg)
    
    @staticmethod
    def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
        return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
