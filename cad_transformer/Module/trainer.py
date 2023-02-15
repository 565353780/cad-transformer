#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Dataset.cad import CADDataLoader, CADDataset
from cad_transformer.Method.args import parse_args
from cad_transformer.Method.eval import do_eval, get_eval_criteria
from cad_transformer.Method.logger import create_logger
from cad_transformer.Method.path import createFileFolder
from cad_transformer.Method.time import getCurrentTime
from cad_transformer.Model.cad_transformer import CADTransformer

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(
            m, torch.nn.BatchNorm1d):
        m.momentum = momentum


class Trainer(object):

    def __init__(self):
        self.args = parse_args()
        self.cfg = update_config(config, self.args)

        self.model = CADTransformer(self.cfg).cuda()
        self.CE_loss = torch.nn.CrossEntropyLoss().cuda()
        val_dataset = CADDataset(split='val',
                                 do_norm=self.cfg.do_norm,
                                 cfg=self.cfg,
                                 max_prim=self.cfg.max_prim)
        self.val_dataloader = CADDataLoader(
            0,
            dataset=val_dataset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False)
        test_dataset = CADDataset(split='test',
                                  do_norm=self.cfg.do_norm,
                                  cfg=self.cfg,
                                  max_prim=self.cfg.max_prim)
        self.test_dataloader = CADDataLoader(
            0,
            dataset=test_dataset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False)
        train_dataset = CADDataset(split='train',
                                   do_norm=self.cfg.do_norm,
                                   cfg=self.cfg,
                                   max_prim=self.cfg.max_prim)
        self.train_dataloader = CADDataLoader(0,
                                              dataset=train_dataset,
                                              batch_size=self.cfg.batch_size,
                                              shuffle=True,
                                              num_workers=self.cfg.WORKERS,
                                              drop_last=True)

        self.start_epoch = 0
        self.eval_F1 = 0
        self.best_F1 = 0
        self.best_epoch = 0
        self.global_epoch = 0

        self.step = 0
        self.log_folder_name = getCurrentTime()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.learning_rate,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=self.cfg.weight_decay)

        self.logger = None
        self.summary_writer = None

        self.initLogger()
        return

    def initLogger(self):
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        if self.cfg.eval_only:
            self.logger = create_logger(self.cfg.log_dir, 'val')
        elif self.cfg.test_only:
            self.logger = create_logger(self.cfg.log_dir, 'test')
        else:
            self.logger = create_logger(self.cfg.log_dir, 'train')
        return True

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model_file not exist!")
            print("\t", model_file_path)
            return False

        model_dict = torch.load(model_file_path)
        #  map_location=torch.device("cpu"))

        self.start_epoch = model_dict['epoch']
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        epoch = model_dict['epoch']
        print(f'=> resume checkpoint: {model_file_path} (epoch: {epoch})')
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        return True

    def updateLearningRate(self, epoch):
        lr = max(
            self.cfg.learning_rate *
            (self.cfg.lr_decay**(epoch // self.cfg.step_size)),
            self.cfg.LEARNING_RATE_CLIP)
        if epoch <= self.cfg.epoch_warmup:
            lr = self.cfg.learning_rate_warmup

        print(f'Learning rate: {lr}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return True

    def updateMomentum(self, epoch):
        momentum = self.cfg.MOMENTUM_ORIGINAL * (self.cfg.MOMENTUM_DECCAY**
                                                 (epoch // self.cfg.step_size))
        if momentum < 0.01:
            momentum = 0.01
        print(f'BN momentum updated to: {momentum}')
        self.model = self.model.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        return True

    def trainStep(self, data):
        image, xy, target, rgb_info, nns, _, _, _, _ = data

        self.optimizer.zero_grad()

        seg_pred = self.model(image, xy, rgb_info, nns)
        seg_pred = seg_pred.contiguous().view(-1, cfg.num_class + 1)
        target = target.view(-1, 1)[:, 0]

        loss_seg = self.CE_loss(seg_pred, target)
        loss = loss_seg
        loss.backward()
        self.optimizer.step()
        return loss_seg, loss

    def saveModel(self, epoch, model_file_path):
        createFileFolder(model_file_path)

        state = {
            'epoch': epoch,
            'best_F1': self.best_F1,
            'best_epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, model_file_path)
        return True

    def eval(self):
        return True

    def train(self):
        self.model.train()

        #  torch.multiprocessing.set_start_method('spawn', force=True)

        if self.cfg.eval_only:
            self.eval_F1 = do_eval(self.model, self.val_dataloader,
                                   self.logger, self.cfg)
            exit(0)

        # Test Only
        if self.cfg.test_only:
            self.eval_F1 = do_eval(self.model, self.test_dataloader,
                                   self.logger, self.cfg)
            exit(0)

        print("> start epoch", self.start_epoch)
        for epoch in range(self.start_epoch, self.cfg.epoch):
            print(f"=> {self.cfg.log_dir}\n\n")
            print(
                f'Epoch {self.global_epoch + 1} ({epoch + 1}/{self.cfg.epoch})'
            )

            self.updateLearningRate(epoch)
            self.updateMomentum(epoch)

            self.model = self.model.train()

            # training loops
            with tqdm(self.train_dataloader,
                      total=len(self.train_dataloader),
                      smoothing=0.9) as _tqdm:
                for i, data in enumerate(_tqdm):
                    loss_seg, loss = self.trainStep(data)
                    _tqdm.set_postfix(loss=loss.item(), l_seg=loss_seg.item())

                    if i % self.args.log_step == 0:
                        print(
                            f'Train loss: {round(loss.item(), 5)}, loss seg: {round(loss_seg.item(), 5)})'
                        )

            print('Save last model...')
            savepath = os.path.join(self.cfg.log_dir, 'last_model.pth')
            self.saveModel(epoch, savepath)

            # assert validation?
            if get_eval_criteria(epoch):
                print('> do validation')
                self.eval_F1 = do_eval(self.model, self.val_dataloader,
                                       self.logger, self.cfg)

            # Save ckpt
            if self.eval_F1 > self.best_F1:
                self.best_F1 = self.eval_F1
                self.best_epoch = epoch

                print(
                    f'Save model... Best F1:{self.best_F1}, Best Epoch:{self.best_epoch}'
                )
                savepath = os.path.join(self.cfg.log_dir, 'best_model.pth')
                self.saveModel(epoch, savepath)

            self.global_epoch += 1
        return True
