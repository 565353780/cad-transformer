#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.args import parse_args
from cad_transformer.Dataset.cad import CADDataLoader, CADDataset
from cad_transformer.Model.cad_transformer import CADTransformer

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


class Detector(object):

    def __init__(self):
        self.args = parse_args()
        self.cfg = update_config(config, self.args)

        self.model = CADTransformer(self.cfg).cuda()

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
        return

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path)
        #  map_location=torch.device("cpu"))

        self.model.load_state_dict(model_dict['model'])

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")

        self.model.eval()
        return True

    def detect(self, data):
        image, xy, _, rgb_info, nns, _, _, _, _ = data

        seg_pred = self.model(image, xy, rgb_info, nns)
        seg_pred = seg_pred.contiguous().view(-1, self.cfg.num_class + 1)
        return seg_pred

    def detectDataset(self, print_progress=False):
        for_data = self.test_dataloader
        if print_progress:
            print("[INFO][Detector::detectDataset]")
            print("\t start detect...")
            for_data = tqdm(for_data, total=len(for_data), smoothing=0.9)
        for data in for_data:
            seg_pred = self.detect(data)
            print(seg_pred)
        return True
