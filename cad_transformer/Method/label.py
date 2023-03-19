#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from cad_transformer.Config.anno_config import TRAIN_CLASS_MAP_DICT


def mapSemanticLabel(label, train_mode='all'):
    new_label = np.array(deepcopy(label), dtype=int)

    if train_mode not in TRAIN_CLASS_MAP_DICT.keys():
        print("[ERROR][label::mapSemanticLabel]")
        print("\t train_mode [" + train_mode +
              "] not in TRAIN_CLASS_MAP_DICT.keys()!")
        return new_label

    train_class_map = TRAIN_CLASS_MAP_DICT[train_mode]
    mask_dict = {}
    for key, item in train_class_map.items():
        if key == 'others':
            continue

        mask = np.zeros_like(new_label, dtype=bool)
        for idx in item:
            mask = mask | (new_label == idx)

        mask_dict[key] = mask

    if 'others' in train_class_map.keys():
        others_value = train_class_map['others']
        new_label = np.ones_like(new_label, dtype=int) * others_value

    for key, item in mask_dict.items():
        new_label[item] = int(key)
    return new_label
