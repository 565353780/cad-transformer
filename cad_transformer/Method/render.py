#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np


def visualize_points(point_set,
                     seg_pred,
                     offset_pred,
                     save_dir,
                     basename,
                     instance_point_dict,
                     color_pallete,
                     re_norm=True):
    """ visualization """
    os.makedirs(save_dir, exist_ok=True)
    basename = str(basename[0].split(".")[0])
    img = np.zeros((700, 700, 3))

    point_set_noise = copy.deepcopy(point_set)
    for idx_center in range(point_set.shape[0]):
        point_class = int(seg_pred[idx_center].cpu().numpy())
        if point_class == 0 or 31 <= point_class <= 35:
            continue
        color = color_pallete[point_class]
        pts = point_set_noise[idx_center]
        offset = offset_pred[idx_center]
        pts -= offset
        pts = pts.cpu().numpy()
        if re_norm:
            pts = pts * 350 + 350
        pts = [int(p) for p in pts]
        cv2.circle(img, pts, 2, color)
    cv2.imwrite(
        os.path.join(save_dir, "{}_{}_pred.png".format(basename,
                                                       point_set.shape[0])),
        img)

    img = np.zeros((700, 700, 3))
    for key in instance_point_dict.keys():
        point_class = instance_point_dict[key]["point_class"]
        if point_class == 0 or 31 <= point_class <= 35:
            continue
        bottom_right = instance_point_dict[key]["max"]
        top_left = instance_point_dict[key]["min"]
        color = color_pallete[point_class]
        if re_norm:
            top_left = [_ * 350 + 350 for _ in top_left]
            top_left = [int(p) for p in top_left]
            bottom_right = [_ * 350 + 350 for _ in bottom_right]
            bottom_right = [int(p) for p in bottom_right]
        cv2.rectangle(img, top_left, bottom_right, color, 2)
    cv2.imwrite(
        os.path.join(save_dir, "{}_{}_gt.png".format(basename,
                                                     point_set.shape[0])), img)
    return True
