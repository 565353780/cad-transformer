#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timm.models.vision_transformer import \
    _create_vision_transformer, checkpoint_filter_fn, default_cfgs
from timm.models.helpers import build_model_with_cfg

from cad_transformer.Config.vit import vit_stage_layer_mapping
from cad_transformer.Model.vision_transformer.net import VisionTransformer_


def vit_base_patch32_384_(pretrained=True, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        out_indices=[1, 3, 9, 11],
                        **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384',
                                       pretrained=pretrained,
                                       **model_kwargs)
    return model


def vit_small_patch32_384_(pretrained=True, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        out_indices=[1, 3, 9, 11],
                        **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384',
                                       pretrained=pretrained,
                                       **model_kwargs)
    return model


def _create_vision_transformer(variant,
                               pretrained=True,
                               default_cfg=None,
                               **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError(
            'features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        repr_size = None

    # use self-supervised trained model
    # default_cfg['url'] = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'

    model = build_model_with_cfg(VisionTransformer_,
                                 variant,
                                 pretrained,
                                 pretrained_cfg=default_cfg,
                                 representation_size=repr_size,
                                 pretrained_filter_fn=checkpoint_filter_fn,
                                 pretrained_custom_load='npz'
                                 in default_cfg['url'],
                                 **kwargs)
    return model


def get_vit(pretrained=True, cfg=None):
    model_nn = cfg.model.model_nn.split("_")
    model_nn = [int(x) for x in model_nn]
    model_k = cfg.model.model_k.split("_")
    model_k = [int(x) for x in model_k]

    model_nn_, model_k_ = list(), list()
    for key, val in vit_stage_layer_mapping.items():
        stage_num = int(key.split("stage")[1])
        for _ in val:
            nn_tmp = model_nn[stage_num - 1]
            k_tmp = model_k[stage_num - 1]
            model_nn_.append(nn_tmp)
            model_k_.append(k_tmp)

    ViT = vit_small_patch32_384_(pretrained=pretrained,
                                 model_nn=model_nn_,
                                 model_k=model_k_)
    # ViT = vit_base_patch32_384_(pretrained=pretrained, model_nn=model_nn_, model_k=model_k_)
    ViT.cuda()
    return ViT
