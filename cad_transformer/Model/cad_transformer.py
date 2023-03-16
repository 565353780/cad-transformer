#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from cad_transformer.Model.layers import AMSoftmaxLayer
from cad_transformer.Model.input_embed import InputEmbed
from cad_transformer.Model.vit import get_vit


class CADTransformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.do_clus = cfg.do_clus
        self.clus_nn = cfg.clus_nn
        self.model_nn = cfg.model.model_nn
        self.n_c = cfg.num_class + 1
        self.inter_dim = cfg.inter_dim

        self.input_embed = InputEmbed(cfg)
        self.fc_bottleneck = nn.Linear(cfg.input_embed_dim, cfg.inter_dim)
        self.transformers = get_vit(pretrained=True, cfg=cfg)

        self.fc3 = nn.Sequential(
            nn.Linear(self.inter_dim, self.inter_dim * 2),
            nn.ReLU(),
            nn.Linear(self.inter_dim * 2, self.inter_dim * 2),
            nn.ReLU(),
        )

        self.last_linear = AMSoftmaxLayer(self.inter_dim * 2, self.n_c, s=30)
        #  self.last_linear = nn.Linear(self.inter_dim * 2, self.n_c)
        return

    def forward(self, image, xy, nns):
        xy_embed = self.input_embed(image, xy)
        xy_embed = self.fc_bottleneck(xy_embed)

        xy_embed_list = self.transformers([xy, xy_embed, nns])
        xy_embed, attns = xy_embed_list

        res = self.fc3(xy_embed)
        res = self.last_linear(res)
        return res
