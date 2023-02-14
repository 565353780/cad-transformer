#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from timm.models.layers import DropPath, Mlp

from cad_transformer.Model.vision_transformer.attention import Attention


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 model_nn=None,
                 model_k=None,
                 out=False):
        super().__init__()
        self.out = out
        self.model_nn = model_nn
        self.model_k = model_k
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              model_k=model_k,
                              model_nn=model_nn)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        # print("model_nn:{}, model_k:{}".format(model_nn, model_k))

    def forward(self, x):
        xyz, xy_embed, nns, xy_embed_list, _ = x
        x, attn = self.attn(xyz, self.norm1(xy_embed), nns)
        x = xy_embed + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.out:
            xy_embed_list.append(x)

        return xyz, x, nns, xy_embed_list, attn
