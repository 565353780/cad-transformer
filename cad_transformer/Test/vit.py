#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from cad_transformer.Model.vit import get_vit


def test():
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    # VIT = vit_small_patch32_384_(pretrained=True).cuda()
    VIT = get_vit(pretrained=True)
    n_point = 300
    xy = torch.randn((1, n_point, 2)).cuda()
    xy_embed = torch.randn((1, n_point, 768)).cuda()

    nns = torch.randn((1, n_point, 128)).long().cuda()
    nns = torch.clamp(nns, 0, 128)

    target = torch.randn((1, n_point)).long().cuda()
    target = torch.clamp(target, 0, 35)

    output = VIT([xy, xy_embed, nns])
    print(torch.mean(xy), torch.mean(xy_embed.float()),
          torch.mean(nns.float()))
    print(torch.mean(output))
    return True


if __name__ == "__main__":
    test()
