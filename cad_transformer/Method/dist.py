#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm


def square_distance(src, dst):
    '''Calculate Euclid distance between each two points using pytorch.
    '''
    return torch.sum((src[:, :, None] - dst[:, None])**2, dim=-1)


def get_nn(segments, max_degree=4, avoid_self_idx=False, print_progress=False):
    '''Calculate the neighbors of each point
    '''
    segments = torch.Tensor(segments)

    p_start = segments[:, :2].unsqueeze(0)
    p_end = segments[:, 2:].unsqueeze(0)

    nns_list = []

    for_data = enumerate(segments)
    if print_progress:
        print("[INFO][dist::get_nn]")
        print("\t start compute nearest neighbors...")
        for_data = tqdm(for_data)
    for i, seg in for_data:
        i_start = seg[:2].unsqueeze(0).unsqueeze(0)
        i_end = seg[2:].unsqueeze(0).unsqueeze(0)

        dist_istart_pstart = square_distance(i_start, p_start)[0, :]
        dist_istart_pend = square_distance(i_start, p_end)[0, :]
        dist_iend_pstart = square_distance(i_end, p_start)[0, :]
        dist_iend_pend = square_distance(i_end, p_end)[0, :]

        if avoid_self_idx:
            dist_istart_pstart[0][i] = float('inf')
            dist_istart_pend[0][i] = float('inf')
            dist_iend_pstart[0][i] = float('inf')
            dist_iend_pend[0][i] = float('inf')

        dist_cat = torch.cat([
            dist_istart_pstart, dist_istart_pend, dist_iend_pstart,
            dist_iend_pend
        ], 0)
        dist_min = torch.min(dist_cat, 0)[0]
        near_idx = dist_min.argsort()[:max_degree].numpy().tolist()

        nns_list.append(near_idx)
    return nns_list
