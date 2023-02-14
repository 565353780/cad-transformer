#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AMSoftmaxLayer(nn.Module):
    """AMSoftmaxLayer"""

    def __init__(self, in_feats, n_classes, s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes),
                                    requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        return

    def forward(self, x):
        batch, pts_num, embed_dim = x.shape
        x = x.view(batch * pts_num, embed_dim)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) * self.s
        costh = costh.view(batch, pts_num, -1)
        return costh
