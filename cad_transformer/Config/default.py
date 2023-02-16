#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from yacs.config import CfgNode as CN

ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
GPU_NUM = torch.cuda.device_count()

_C = CN()

# train parameters
_C.batch_size = 1
_C.test_batch_size = 1
_C.epoch = 300
_C.learning_rate = 2.5e-5
_C.learning_rate_warmup = 1e-5
_C.root = "{}/data/floorplan_v1".format(ROOT)  #dir to dataset

# Dataset
_C.do_clus = False
_C.old_model = True
_C.do_norm = True
_C.WORKERS = 0
_C.clus_num_per_batch = 16
_C.clus_nn = 64
_C.img_size = 700
_C.filter_num = 16
_C.aug_ratio = 0  #Random Layer data augmentation ratio

# Optimizer
_C.lr_decay = 0.5
_C.step_size = 20
_C.LEARNING_RATE_CLIP = 1e-5
_C.MOMENTUM_ORIGINAL = 0.1
_C.MOMENTUM_DECCAY = 0.5
_C.epoch_warmup = 0
_C.offset_weight = 0.3

# Model
_C.model = CN()
_C.model.model_nn = "4_8_16_32"
_C.model.model_k = "2_4_8_16"
_C.model.nblocks = 4  # invalid if use ViT
_C.model.transformer_dim = 128  # invalid if use ViT

_C.input_dim = 2
_C.input_embed_dim = 384
_C.inter_dim = 384
_C.rgb_dim = 32
_C.am_softmax = True
_C.deg_embed = False
_C.weight_decay = 1e-4
_C.num_class = 35

_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.BACKBONE = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.local_rank = 0
_C.seed = 304


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    if args.data_root is not None:
        cfg.defrost()
        cfg.root = args.data_root
    if args.pretrained_model is not None:
        cfg.defrost()
        cfg.MODEL.PRETRAINED = args.pretrained_model
    if args.embed_backbone is not None:
        cfg.defrost()
        cfg.MODEL.BACKBONE = args.embed_backbone
    if args.max_prim is not None:
        cfg.defrost()
        cfg.max_prim = args.max_prim
    return cfg


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
