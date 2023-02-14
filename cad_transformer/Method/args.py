#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        type=str,
                        default="cad_transformer/Config/hrnet48.yaml",
                        help='experiment configure file name')
    parser.add_argument('--val_only',
                        action="store_true",
                        help='flag to do evaluation on val set')
    parser.add_argument('--test_only',
                        action="store_true",
                        help='flag to do evaluation on test set')
    parser.add_argument('--data_root',
                        type=str,
                        default="/home/chli/chLi/FloorPlanCAD")
    parser.add_argument('--embed_backbone', type=str, default="hrnet48")
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default="/home/chli/chLi/HRNet/hrnetv2_w48_imagenet_pretrained.pth")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--log_step",
                        type=int,
                        default=100,
                        help='steps for logging')
    parser.add_argument("--img_size",
                        type=int,
                        default=700,
                        help='image size of rasterized image')
    parser.add_argument("--max_prim",
                        type=int,
                        default=3000,
                        help='maximum primitive number for each batch')
    parser.add_argument("--load_ckpt",
                        type=str,
                        default='',
                        help='load checkpoint')
    parser.add_argument("--resume_ckpt",
                        type=str,
                        default='',
                        help='continue train while loading checkpoint')
    parser.add_argument("--log_dir",
                        type=str,
                        default='train0',
                        help='logging directory')
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args
