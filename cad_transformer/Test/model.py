#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from pdb import set_trace as st

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../", "../"))
sys.path.insert(0, os.path.join(BASE_DIR, "../"))

from cad_transformer.Config.default import _C as config, update_config
from cad_transformer.Model.cad_transformer import CADTransformer
from cad_transformer.Method.args import parse_args

args = parse_args()
cfg = update_config(config, args)


def test():

    model = CADTransformer(cfg)
    model.cuda()
    adj_node_classes = np.load("/ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/npy/train/0152-0012.npy", \
                        allow_pickle=True).item()
    target = adj_node_classes["cat"]
    target = torch.from_numpy(np.array(target,
                                       dtype=np.int64)).cuda().unsqueeze(0)

    center = adj_node_classes["ct_norm"]
    points = torch.from_numpy(np.array(center,
                                       dtype=np.float32)).cuda().unsqueeze(0)

    nns = adj_node_classes["nns"]
    nns = torch.from_numpy(np.array(nns, dtype=np.int64)).cuda().unsqueeze(0)

    image = torch.randn((1, 3, 700, 700)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(1):
        print("\n")
        seg_pred, adj_prob, nns = model(image, points, nns)
        adj_matrix = torch.zeros(adj_prob.shape[1],
                                 adj_prob.shape[1]).to(nns.device)
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, nns[0, i, :]] = adj_prob[0, i, :, 0]
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        st()
        # seg_pred = seg_pred.contiguous().view(-1, cfg.num_class+1)
        # target = target.view(-1, 1)[:, 0]
        # print(seg_pred.shape, target.shape)
        # loss = criterion(seg_pred, target)
        # loss.backward()


if __name__ == "__main__":
    test()
