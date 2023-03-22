#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../svg-render")

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from svg_render.Module.renderer import Renderer

from cad_transformer.Config.anno_config import AnnoList
from cad_transformer.Config.args import parse_args
from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.image_net import IMAGENET_MEAN, IMAGENET_STD
from cad_transformer.Method.time import getCurrentTime
from cad_transformer.Dataset.cad import CADDataset
from cad_transformer.Model.cad_transformer import CADTransformer

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def getRenderer():
    return Renderer


class Detector(object):

    def __init__(self, train_mode='all'):
        self.train_mode = train_mode

        self.class_num = len(AnnoList(self.train_mode).anno_list_all_reverse)

        self.args = parse_args()
        self.cfg = update_config(config, self.args)

        self.model = CADTransformer(self.cfg, self.class_num).cuda()

        transform = [T.ToTensor()]
        transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform)

        self.renderer = Renderer(4000, 4000, 50, 2560, 1440)
        self.render_mode = 'type+semantic+selected_semantic+custom_semantic'
        self.line_width = 3
        self.text_color = [0, 0, 255]
        self.text_size = 1
        self.text_line_width = 1
        self.print_progress = False
        self.selected_semantic_idx_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34
        ]
        self.wait_key = 0
        self.window_name = '[Renderer][' + self.render_mode + ']'
        return

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path)

        self.model.load_state_dict(model_dict['model'])

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")

        self.model.eval()
        return True

    def detect(self, image, xy, nns):
        seg_pred = self.model(image, xy, nns)
        seg_pred = seg_pred.contiguous().view(-1, self.class_num)
        pred_choice = seg_pred.data.max(1)[1]
        result = pred_choice.detach().cpu().numpy()
        del seg_pred
        del pred_choice
        return result

    def getResultImage(self, svg_file_path, result):
        self.renderer.renderFile(svg_file_path, self.render_mode,
                                 self.line_width, self.text_color,
                                 self.text_size, self.text_line_width,
                                 self.print_progress,
                                 self.selected_semantic_idx_list, result)
        self.renderer.show(self.wait_key, self.window_name)
        return self.renderer.getRenderImage()

    def detectDataset(self, print_progress=False):
        save_result_image_folder_path = './render/' + getCurrentTime() + '/'
        os.makedirs(save_result_image_folder_path)

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

            svg_file_path = img_path.replace('/png/',
                                             '/svg/').replace('.png', '.svg')
            assert os.path.exists(svg_file_path)

            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.cfg.img_size, self.cfg.img_size))
            image = self.transform(image).cuda().unsqueeze(0)

            adj_node_classes = np.load(ann_path, \
                                allow_pickle=True).item()

            center = adj_node_classes["ct_norm"]
            xy = torch.from_numpy(np.array(
                center, dtype=np.float32)).cuda().unsqueeze(0)

            nns = adj_node_classes["nns"]
            nns = torch.from_numpy(np.array(
                nns, dtype=np.int64)).cuda().unsqueeze(0)

            result = self.detect(image, xy, nns)

            result_image = self.getResultImage(svg_file_path, result)

            cv2.imwrite(save_result_image_folder_path + str(data_idx) + '.png',
                        result_image)
            exit()
        return True
