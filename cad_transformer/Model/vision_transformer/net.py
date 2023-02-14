#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.helpers import named_apply

from cad_transformer.Model.vision_transformer.block import Block


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    return True


class VisionTransformer_(VisionTransformer):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 representation_size=None,
                 distilled=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 weight_init='',
                 out_indices=[],
                 model_nn=None,
                 model_k=None):
        self.dist_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if distilled else None
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.num_heads = num_heads
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.out_indices = out_indices
        self.model_nn = model_nn
        self.model_k = model_k

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  model_k=self.model_k[i],
                  model_nn=self.model_nn[i],
                  out=i in out_indices) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(embed_dim, representation_size)),
                             ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(
            self.num_features,
            num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim,
                self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(
                partial(_init_vit_weights, head_bias=head_bias, jax_impl=True),
                self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_features(self, feat=[]):
        xy_embed_list = []
        xy, xy_embed, nns = feat

        _, xy_embed, _, xy_embed_list, attns = self.blocks(
            [xy, xy_embed, nns, xy_embed_list,
             None])  # [1, 145, 384] -> [1, 145, 384]
        xy_embed = self.norm(xy_embed)
        # res = sum(xy_embed_list) / len(xy_embed_list)
        res = xy_embed
        return [res, attns]

    def forward(self, feat=[]):
        x_list = self.forward_features(feat=feat)
        x, attns = x_list
        return [x, attns]
