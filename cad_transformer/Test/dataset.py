#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from cad_transformer.Config.default import _C as config
from cad_transformer.Config.default import update_config
from cad_transformer.Config.args import parse_args

from cad_transformer.Dataset.cad import CADDataset


def test():
    args = parse_args()
    cfg = update_config(config, args)

    dataset = CADDataset(split='train',
                         do_norm=cfg.do_norm,
                         cfg=cfg,
                         max_prim=cfg.max_prim)

    for i in tqdm(range(len(dataset))):
        image, xy, nns, target = dataset.__getitem__(i)
        print(image.size)
        print(xy.shape)
        print(nns.shape)
        print(target.shape)
    return True
