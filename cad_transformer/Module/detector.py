#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.args import parse_args
from cad_transformer.Config.image_net import IMAGENET_MEAN, IMAGENET_STD
from cad_transformer.Dataset.cad import CADDataset
from cad_transformer.Model.cad_transformer import CADTransformer

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


class Detector(object):

    def __init__(self):
        self.args = parse_args()
        self.cfg = update_config(config, self.args)

        self.model = CADTransformer(self.cfg).cuda()

        transform = [T.ToTensor()]
        transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform)
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

    def detect(self, image, xy, rgb_info, nns):
        seg_pred = self.model(image, xy, rgb_info, nns)
        seg_pred = seg_pred.contiguous().view(-1, self.cfg.num_class + 1)
        return seg_pred

    def detectDataset(self, print_progress=False):
        dataset = CADDataset(split='test',
                             do_norm=self.cfg.do_norm,
                             cfg=self.cfg,
                             max_prim=self.cfg.max_prim)

        for_data = range(len(dataset))
        if print_progress:
            print("[INFO][Detector::detectDataset]")
            print("\t start detect...")
            for_data = tqdm(for_data)
        for data_idx in for_data:
            img_path = dataset.image_path_list[data_idx]
            ann_path = dataset.anno_path_list[data_idx]
            assert os.path.basename(img_path).split(".")[0] == \
                os.path.basename(ann_path).split(".")[0]

            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.cfg.img_size, self.cfg.img_size))
            image = self.transform(image).cuda().unsqueeze(0)

            adj_node_classes = np.load(ann_path, \
                                allow_pickle=True).item()

            center = adj_node_classes["ct_norm"]
            xy = torch.from_numpy(np.array(
                center, dtype=np.float32)).cuda().unsqueeze(0)

            rgb_info = np.load(ann_path, allow_pickle=True).item()['ct_norm']
            rgb_info = torch.from_numpy(np.array(
                rgb_info, dtype=np.int64)).cuda().unsqueeze(0)

            nns = adj_node_classes["nns"]
            nns = torch.from_numpy(np.array(
                nns, dtype=np.int64)).cuda().unsqueeze(0)

            seg_pred = self.detect(image, xy, rgb_info, nns)
            print(seg_pred)
            del seg_pred
        return True
