#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def OffsetLoss(pred, gt, inst_id):
    """ offset loss for vertex movement """
    # pred = torch.clamp(pred, -1, 1)
    pt_diff = pred - gt
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    valid = (inst_id != -1).squeeze(-1).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
    return offset_norm_loss
