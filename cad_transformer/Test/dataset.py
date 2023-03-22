#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../svg-render")

import os
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from svg_render.Module.renderer import Renderer

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.args import parse_args
from cad_transformer.Config.image_net import IMAGENET_MEAN, IMAGENET_STD
from cad_transformer.Config.anno_config import TRAIN_CLASS_MAP_DICT
from cad_transformer.Method.label import mapSemanticLabel
from cad_transformer.Method.time import getCurrentTime
from cad_transformer.Dataset.cad import CADDataset


def test():
    train_mode = 'wall'
    load_num_max = 20

    renderer = Renderer(4000, 4000, 50, 2560, 1440)
    render_mode = 'type+semantic+selected_semantic'
    line_width = 3
    text_color = [0, 0, 255]
    text_size = 1
    text_line_width = 1
    print_progress = False
    selected_semantic_idx_list = TRAIN_CLASS_MAP_DICT[train_mode]['1']
    wait_key = 0
    window_name = '[Renderer][' + render_mode + ']'

    args = parse_args()
    cfg = update_config(config, args)

    dataset = CADDataset(split='train',
                         do_norm=cfg.do_norm,
                         cfg=cfg,
                         max_prim=cfg.max_prim,
                         load_num_max=load_num_max)

    transform = [T.ToTensor()]
    transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = T.Compose(transform)

    save_result_image_folder_path = './render_train/' + getCurrentTime() + '/'
    os.makedirs(save_result_image_folder_path)

    print("[INFO][dataset::test]")
    print("\t start save training dataset visual images...")
    for data_idx in tqdm(range(len(dataset))):
        img_path = dataset.image_path_list[data_idx]
        ann_path = dataset.anno_path_list[data_idx]
        assert os.path.basename(img_path).split(".")[0] == \
            os.path.basename(ann_path).split(".")[0]

        svg_file_path = img_path.replace('/png/',
                                         '/svg/').replace('.png', '.svg')
        assert os.path.exists(svg_file_path)

        adj_node_classes = np.load(ann_path, \
                            allow_pickle=True).item()

        target = adj_node_classes["cat"]
        target = np.array(target, dtype=np.int64).reshape(-1)

        target = mapSemanticLabel(target, train_mode)

        renderer.renderFile(svg_file_path, render_mode, line_width, text_color,
                            text_size, text_line_width, print_progress,
                            selected_semantic_idx_list, target)
        result_image = renderer.getRenderImage()

        #  renderer.show(wait_key, window_name)

        cv2.imwrite(save_result_image_folder_path + str(data_idx) + '.png',
                    result_image)
    return True
