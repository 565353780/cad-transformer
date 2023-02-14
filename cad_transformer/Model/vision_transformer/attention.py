#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points.clone(), 1,
                       idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 model_k=None,
                 model_nn=None) -> None:
        super().__init__()
        assert model_k is not None
        assert model_nn is not None
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fc_delta = nn.Sequential(nn.Linear(2, dim), nn.ReLU(),
                                      nn.Linear(dim, dim))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.model_k = model_k
        self.model_nn = model_nn

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xy, xy_embed, nns):
        B, N, C = xy_embed.shape

        knn_idx = nns[:, :, :self.model_k]
        xy_knn = index_points(xy, knn_idx)
        qkv = self.qkv(xy_embed).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q_feat, k_feat, v_feat = qkv.unbind(0)
        q, k, v = q_feat, index_points(k_feat, knn_idx), index_points(
            v_feat, knn_idx)  # q: b x n x h*f, kv: b x n x k x h*f
        num_k = k.shape[-2]
        assert num_k == v.shape[-2]
        q = q.reshape([B, N, self.num_heads, -1]).permute(
            [0, 2, 1, 3])  # b x n x h*f -> b x n x h x f -> b x h x n x f
        k = k.reshape([B, N, num_k, self.num_heads, -1]).permute(
            [0, 3, 1, 2,
             4])  # b x n x k x h*f -> b x n x k x h x f -> b x h x n x k x f
        v = v.reshape([B, N, num_k, self.num_heads, -1]).permute(
            [0, 3, 1, 2,
             4])  # b x n x k x h*f -> b x n x k x h x f -> b x h x n x k x f

        pos_enc = self.fc_delta(xy[:, :, None] - xy_knn).permute(
            [0, 3, 1, 2])  # b x n x (xy) -> b x n x k x hf -> b x hf x n x k
        pos_enc = pos_enc.reshape([B, self.num_heads, -1, N, num_k]).permute(
            [0, 1, 3, 4,
             2])  # b x hf x n x k -> b x h x f x n x k -> b x h x n x k x f
        # main difference. Vanilla ViT: b x n x c @ b x c x n -> b x n x n
        attn = torch.sum(q[..., :, None, :] * k,
                         -1)  # b x h x n x k x f -> b x h x n x k
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)  # b x h x n x k
        attn = self.attn_drop(attn)
        v = v + pos_enc
        # b x h x n x k x f -> b x h x n x f ->(permute) -> b x n x h x f ->(reshape) b x n x (h x f)
        x = torch.sum(attn[..., None] * v, -2).permute([0, 2, 1,
                                                        3]).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, [attn, knn_idx]
