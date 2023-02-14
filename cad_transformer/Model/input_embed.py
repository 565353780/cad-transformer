#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from cad_transformer.Model.high_reso.net import get_seg_model
from torch.nn import functional as F

from cad_transformer.Config.resnet import FEAT_DIMS


def vert_align_custom(feats,
                      verts,
                      interp_mode='bilinear',
                      padding_mode='zeros',
                      align_corners=True):
    if torch.is_tensor(verts):
        if verts.dim() != 3:
            raise ValueError("verts tensor should be 3 dimensional")
        grid = verts
    else:
        raise ValueError("verts must be a tensor or have a " +
                         "`points_padded' or`verts_padded` attribute.")
    grid = grid[:, None, :, :2]  # (N, 1, V, 2)
    if torch.is_tensor(feats):
        feats = [feats]
    for feat in feats:
        if feat.dim() != 4:
            raise ValueError("feats must have shape (N, C, H, W)")
        if grid.shape[0] != feat.shape[0]:
            raise ValueError("inconsistent batch dimension")
    feats_sampled = []
    for feat in feats:
        feat_sampled = F.grid_sample(
            feat,
            grid,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feats_sampled.append(feat_sampled)
    feats_sampled = torch.cat(feats_sampled, dim=2)  # (N, V, sum(C))
    return feats_sampled


class InputEmbed(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        print(f"> InputEmbed: {cfg.MODEL.BACKBONE}")
        if cfg.MODEL.BACKBONE == "hrnet48":
            self.EmbedBackbone = get_seg_model(cfg)
        else:
            raise NotImplementedError

        self.EmbedDim = FEAT_DIMS[cfg.MODEL.BACKBONE]
        self.bottleneck = nn.Linear(sum(self.EmbedDim), cfg.input_embed_dim)
        self.do_clus = cfg.do_clus

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

    def forward(self, image, x):
        if self.do_clus:
            batch, clus, pts_num, xy = x.shape
            x = x.view(batch, clus * pts_num, xy)
        device, dtype = x.device, x.dtype
        img_feats = self.EmbedBackbone(image)
        factor = torch.tensor([1, 1], device=device, dtype=dtype).view(1, 1, 2)
        xy_norm = x * factor

        vert_align_feats = vert_align_custom(img_feats, xy_norm)
        # vert_align_feats = F.relu(self.bottleneck(vert_align_feats))
        vert_align_feats = self.bottleneck(vert_align_feats)
        if self.do_clus:
            vert_align_feats = vert_align_feats.view(batch, clus, pts_num, -1)
        return vert_align_feats
