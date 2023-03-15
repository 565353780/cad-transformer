#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def square_distance(src, dst):
    '''Calculate Euclid distance between each two points using pytorch.
    '''
    return torch.sum((src[:, :, None] - dst[:, None])**2, dim=-1)


def get_nn(segments, max_degree=4):
    '''Calculate the neighbors of each point
    '''
    segments = torch.Tensor(segments)

    p_start = segments[:, :2].unsqueeze(0)
    p_end = segments[:, 2:].unsqueeze(0)

    nns_list = []
    for seg in segments:
        i_start = seg[:2].unsqueeze(0).unsqueeze(0)
        i_end = seg[2:].unsqueeze(0).unsqueeze(0)

        dist_istart_pstart = square_distance(i_start, p_start)[0, :]
        dist_istart_pend = square_distance(i_start, p_end)[0, :]
        dist_iend_pstart = square_distance(i_end, p_start)[0, :]
        dist_iend_pend = square_distance(i_end, p_end)[0, :]

        dist_cat = torch.cat([
            dist_istart_pstart, dist_istart_pend, dist_iend_pstart,
            dist_iend_pend
        ], 0)
        dist_min = torch.min(dist_cat, 0)[0]
        near_idx = dist_min.argsort()[:max_degree].numpy().tolist()

        nns_list.append(near_idx)
    return nns_list
