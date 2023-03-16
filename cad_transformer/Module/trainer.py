#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.args import parse_args
from cad_transformer.Dataset.cad import CADDataLoader, CADDataset
from cad_transformer.Method.eval import do_eval
from cad_transformer.Method.path import createFileFolder, removeFile, renameFile
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
        self.eval_only = False
        self.test_only = False

        self.model = CADTransformer(self.cfg).cuda()

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

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.learning_rate,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=self.cfg.weight_decay)
        self.CE_loss = torch.nn.CrossEntropyLoss().cuda()

        self.step = 0
        self.epoch = 0
        self.best_F1 = 0
        self.log_folder_name = getCurrentTime()

        self.summary_writer = None
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path, load_model_only=False):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)
        #  map_location=torch.device("cpu"))

        self.model.load_state_dict(model_dict['model'])

        if not load_model_only:
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.step = model_dict['step']
            self.epoch = model_dict['epoch']
            self.best_F1 = model_dict['best_F1']
            self.log_folder_name = model_dict['log_folder_name']

        self.loadSummaryWriter()

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_F1': self.best_F1,
            'log_folder_name': self.log_folder_name,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def updateLearningRate(self):
        lr = max(
            self.cfg.learning_rate *
            (self.cfg.lr_decay**(self.epoch // self.cfg.step_size)),
            self.cfg.LEARNING_RATE_CLIP)
        if self.epoch <= self.cfg.epoch_warmup:
            lr = self.cfg.learning_rate_warmup

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.summary_writer.add_scalar("Train/lr", lr, self.step)
        return True

    def updateMomentum(self):
        momentum = self.cfg.MOMENTUM_ORIGINAL * (self.cfg.MOMENTUM_DECCAY**(
            self.epoch // self.cfg.step_size))
        if momentum < 0.01:
            momentum = 0.01
        self.model = self.model.apply(
            lambda x: bn_momentum_adjust(x, momentum))

        self.summary_writer.add_scalar("Train/momentum", momentum, self.step)
        return True

    def trainStep(self, data):
        image, xy, nns, target = data

        self.optimizer.zero_grad()

        seg_pred = self.model(image, xy, nns)
        seg_pred = seg_pred.contiguous().view(-1, self.cfg.num_class + 1)
        target = target.view(-1, 1)[:, 0]

        loss = self.CE_loss(seg_pred, target)
        loss.backward()
        self.optimizer.step()

        self.summary_writer.add_scalar("Loss/loss", round(loss.item(), 5),
                                       self.step)
        return True

    def eval(self, epoch):
        print("[INFO][Trainer::eval]")
        print("\t start eval at epoch " + str(epoch) + "...")
        eval_F1 = do_eval(self.model, self.val_dataloader, self.summary_writer,
                          self.cfg, self.step)

        if eval_F1 <= self.best_F1:
            return True

        self.best_F1 = eval_F1

        save_path = './output/' + self.log_folder_name + '/model_best.pth'
        self.saveModel(save_path)
        return True

    def train(self, print_progress=False):
        self.model.train()

        #  torch.multiprocessing.set_start_method('spawn', force=True)

        if self.eval_only:
            do_eval(self.model, self.val_dataloader, self.summary_writer,
                    self.cfg)
            return True

        if self.test_only:
            do_eval(self.model, self.test_dataloader, self.summary_writer,
                    self.cfg)
            return True

        while self.epoch < self.cfg.epoch:
            print(
                f'Epoch {self.epoch + 1} ({self.epoch + 1}/{self.cfg.epoch})')

            self.updateLearningRate()
            self.updateMomentum()

            self.model = self.model.train()

            for_data = self.train_dataloader
            if print_progress:
                print("[INFO][Trainer::train]")
                print("\t start train at epoch " + str(self.epoch + 1) + "...")
                for_data = tqdm(for_data, total=len(for_data), smoothing=0.9)
            for data in for_data:
                self.trainStep(data)
                self.step += 1

            self.epoch += 1

            save_path = './output/' + self.log_folder_name + '/model_last.pth'
            self.saveModel(save_path)

            self.eval(self.epoch)
        return True
