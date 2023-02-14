import torch
import math
import numpy as np
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, checkpoint_filter_fn, default_cfgs
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_
from timm.models.helpers import build_model_with_cfg, named_apply

from cad_transformer.Config.vit import vit_stage_layer_mapping

if __name__ == "__main__":
    import numpy as np
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
