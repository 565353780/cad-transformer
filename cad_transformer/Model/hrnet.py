#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Model.high_resolution.net import HighResolutionNet


def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)
    return model
