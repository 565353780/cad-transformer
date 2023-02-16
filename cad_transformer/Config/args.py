#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        type=str,
                        default="cad_transformer/Config/hrnet48.yaml",
                        help='experiment configure file name')
    parser.add_argument('--data_root',
                        type=str,
                        default="/home/chli/chLi/FloorPlanCAD")
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default="/home/chli/chLi/HRNet/hrnetv2_w48_imagenet_pretrained.pth")
    parser.add_argument("--max_prim",
                        type=int,
                        default=4000,
                        help='maximum primitive number for each batch')
    parser.add_argument('--embed_backbone', type=str, default="hrnet48")
    args = parser.parse_args()
    return args
