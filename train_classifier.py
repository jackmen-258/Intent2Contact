import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from lib.utils.utils import makelogger, makepath
from lib.networks.classifier import IntentClassifier_V2
from lib.contact.hand_object import HandObject
from lib.datasets.oishape_dataset import OIShape
from lib.utils.config import get_config
from lib.utils.cfg_parser import Config

class IntentClassifierTrainer:
    def __init__(self, cfg, logger=None):
        self.dtype = torch.float32
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)

        if logger:
            self.logger = logger
        else:
            self.logger = makelogger(makepath(os.path.join(cfg.work_dir, 'train.log'), isfile=True)).info
        
        self.logger('Started training %s' % (starttime))

        torch.cuda.set_device(int(cfg.cuda_id))
        torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")

        self.logger(cfg)

        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        oishape_cfg = get_config(cfg.oishape_cfg)
        self.load_data(oishape_cfg)

        # self.model = IntentClassifier(num_intents=4).to(self.device)
        self.model = IntentClassifier_V2(num_intents=4).to(self.device)
        self.hand_object = HandObject(self.device)
        
        vars_net = [var[1] for var in self.model.named_parameters()]
        net_n_params = sum(p.numel() for p in vars_net if p.requires_grad)
        self.logger('Total Trainable Parameters is %2.2f M.' % ((net_n_params) * 1e-6))

        self.optimizer_net = optim.Adam(vars_net, lr=cfg.base_lr, weight_decay=cfg.reg_coef)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_net, T_0=cfg.n_epochs, eta_min=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.patience = 0
        self.max_patience = 20

        if cfg.checkpoint is not None:
            checkpoint = torch.load(cfg.checkpoint, map_location=self.device)
            weight = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(weight, strict=True)
            self.logger('Load model from %s' % cfg.checkpoint)
        
        self.epoch_completed = self.start_epoch

    def load_data(self, cfg):
        ds_train = OIShape(cfg.DATASET.TRAIN)
        self.ds_train = DataLoader(
            ds_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            drop_last=True
        )

        ds_val = OIShape(cfg.DATASET.VAL)
        self.ds_val = DataLoader(
            ds_val, 
            batch_size=min(len(ds_val), self.batch_size), 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        )

    def save_net(self):
        torch.save(self.model.state_dict(), self.cfg.checkpoint)

    def save_ckp(self, state, checkpoint_dir):
        f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        torch.save(state, f_path)

    def load_ckp(self, checkpoint, model):
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train(self, epoch_num):
        self.model.train()
        save_every_it = len(self.ds_train) // self.cfg.log_every_epoch

        train_loss_dict_net = {}
        torch.autograd.set_detect_anomaly(True)

        n_iters = len(self.ds_train)
        for it, input_data in enumerate(self.ds_train):
            input_data = {k: input_data[k].to(self.device) 
                         for k in ['hand_verts', 'obj_verts', 'obj_vn', 'intent_id']}

            dorig = self.hand_object.forward(
                input_data['hand_verts'], 
                input_data['obj_verts'], 
                input_data['obj_vn']
            )
            
            self.optimizer_net.zero_grad()

            if self.fit_net:
                intent_logits = self.model(
                    dorig['verts_object'],     
                    dorig['feat_object'],      
                    dorig['contacts_object']   
                )
                
                loss_total_net, cur_loss_dict_net = self.loss_net(
                    intent_logits, input_data['intent_id']
                )

                loss_total_net.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_net.step()

                train_loss_dict_net = {
                    k: train_loss_dict_net.get(k, 0.0) + v.item() 
                    for k, v in cur_loss_dict_net.items()
                }

                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_net = {k: v / (it + 1) for k, v in train_loss_dict_net.items()}
                    train_msg = self.create_loss_message(
                        cur_train_loss_dict_net,
                        epoch_num=self.epoch_completed, 
                        it=it
                    )
                    self.logger(train_msg)

                self.lr_scheduler.step(epoch_num + it / n_iters)

        train_loss_dict_net = {k: v / len(self.ds_train) for k, v in train_loss_dict_net.items()}
        return train_loss_dict_net

    def evaluate(self):
        self.model.eval()
        eval_loss_dict_net = {}
        
        total_correct = 0
        total_samples = 0
        intent_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        intent_total = {0: 0, 1: 0, 2: 0, 3: 0}
        
        with torch.no_grad():
            for input_data in self.ds_val:
                input_data = {k: input_data[k].to(self.device) 
                             for k in ['hand_verts', 'obj_verts', 'obj_vn', 'intent_id']}
                
                dorig = self.hand_object.forward(
                    input_data['hand_verts'],
                    input_data['obj_verts'], 
                    input_data['obj_vn']
                )
                
                intent_logits = self.model(
                    dorig['verts_object'],
                    dorig['feat_object'],
                    dorig['contacts_object']
                )
                
                _, cur_loss_dict_net = self.loss_net(intent_logits, input_data['intent_id'])

                eval_loss_dict_net = {
                    k: eval_loss_dict_net.get(k, 0.0) + v.item() 
                    for k, v in cur_loss_dict_net.items()
                }
                
                pred_intents = torch.argmax(intent_logits, dim=1)
                correct = (pred_intents == input_data['intent_id'])
                
                total_correct += correct.sum().item()
                total_samples += input_data['intent_id'].size(0)
                
                for intent_id in range(4):
                    mask = (input_data['intent_id'] == intent_id)
                    intent_total[intent_id] += mask.sum().item()
                    intent_correct[intent_id] += (correct & mask).sum().item()
                
        eval_loss_dict_net = {k: v / len(self.ds_val) for k, v in eval_loss_dict_net.items()}
        eval_loss_dict_net['accuracy'] = total_correct / total_samples
        
        for intent_id in range(4):
            if intent_total[intent_id] > 0:
                eval_loss_dict_net[f'intent_{intent_id}_acc'] = (
                    intent_correct[intent_id] / intent_total[intent_id]
                )
            else:
                eval_loss_dict_net[f'intent_{intent_id}_acc'] = 0.0
          
        return eval_loss_dict_net

    def loss_net(self, intent_logits, intent_ids):
        loss_dict = {}
        
        loss_ce = self.criterion(intent_logits, intent_ids)
        loss_dict['loss_ce'] = loss_ce
        
        with torch.no_grad():
            pred_intents = torch.argmax(intent_logits, dim=1)
            accuracy = (pred_intents == intent_ids).float().mean()
            loss_dict['accuracy'] = accuracy

        loss_total = loss_ce
        loss_dict['loss_total'] = loss_total
        
        return loss_total, loss_dict

    def fit(self, n_epochs=None, message=None):
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        if message is not None:
            self.logger(message)

        self.fit_net = True
        
        for epoch_num in range(self.start_epoch, n_epochs):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict_net = self.train(epoch_num)
            eval_loss_dict_net = self.evaluate()

            if self.fit_net:
                with torch.no_grad():
                    eval_msg = self.create_loss_message(
                        eval_loss_dict_net, 
                        epoch_num=self.epoch_completed, 
                        it=0
                    )

                    checkpoint_dir = os.path.join(self.cfg.work_dir, 'checkpoints')
                    makepath(checkpoint_dir, isfile=False)

                    self.cfg.checkpoint = makepath(
                        os.path.join(checkpoint_dir, 'E%03d_classifier.pt' % (self.epoch_completed)), 
                        isfile=True
                    )
                    
                    current_acc = eval_loss_dict_net.get('accuracy', 0.0)
                    if current_acc > self.best_val_acc:
                        self.best_val_acc = current_acc
                        self.patience = 0
                        self.save_net()
                        self.logger(eval_msg + ' ** BEST **')
                    else:
                        self.patience += 1
                        self.logger(eval_msg)
                    
                    if not epoch_num % self.cfg.snapshot_every_epoch:
                        self.save_net()

            self.epoch_completed += 1

            checkpoint = {
                'epoch': epoch_num + 1,
                'state_dict': self.model.state_dict(),
                'best_val_acc': self.best_val_acc,
                'patience': self.patience
            }
            self.save_ckp(checkpoint, self.cfg.work_dir)

            if self.patience >= self.max_patience:
                self.logger(f'Early stopping at epoch {epoch_num} due to no improvement')
                self.fit_net = False

            if not self.fit_net:
                self.logger('Stopping the training!')
                break

        endtime = datetime.now().replace(microsecond=0)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(f'Best validation accuracy: {self.best_val_acc:.4f}')

    @staticmethod
    def create_loss_message(loss_dict, epoch_num=0, it=0):
        total_loss = loss_dict.get('loss_total', 0.0)
        other_metrics = {k: v for k, v in loss_dict.items() if k != 'loss_total'}
        
        ext_msg = ' | '.join(['%s = %.4f' % (k, v) for k, v in other_metrics.items()])
        return 'E%03d - It %05d : [T:%.4f] - [%s]' % (epoch_num, it, total_loss, ext_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intent Classifier Training')

    parser.add_argument('--work-dir', default='runs-classifier', type=str, help='exp dir')
    parser.add_argument('--batch-size', default=64, type=int, help='Training batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Training learning rate')
    parser.add_argument('--config_path', type=str, default='config/classifier.yaml')
    parser.add_argument('--oishape_cfg', type=str, default='config/all_cate.yaml')
    
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    cfg_path = args.config_path
    cfg = {
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'work_dir': args.work_dir,
        'oishape_cfg': args.oishape_cfg,
        'checkpoint': None,
    }

    cfg = Config(default_cfg_path=cfg_path, **cfg)

    trainer = IntentClassifierTrainer(cfg=cfg)
    trainer.fit()