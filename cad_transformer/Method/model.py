#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift

from cad_transformer.Config import anno_config


def mean_shfit(X, bandwidth=None, save_path=None):
    """ clustering step for model predictions """
    # if bandwidth is None:
    #     bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    #     if bandwidth <= 0.1:
    #         bandwidth = 5
    # bandwidth = 10
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    #  cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(1)
        plt.clf()
        plt.xlim(0, 700)
        plt.ylim(0, 700)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            #  cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
            # plt.plot(
            #     cluster_center[0],
            #     cluster_center[1],
            #     "o",
            #     markerfacecolor=col,
            #     markeredgecolor="k",
            #     markersize=14,
            # )
        plt.title(f"C_num:{n_clusters_}, Bandwidth:{round(bandwidth, 4)}")
        plt.savefig(save_path)
        plt.close()
    return labels, n_clusters_

def get_pred_instance(points, seg_pred, offset_pred, \
            basename, pred_instance_dir, cluster_vis_dir=None):
    """ model predictions to instance prediction """
    # npy_path_list = glob(os.path.join(npy_dir, "*.npy"))
    os.makedirs(pred_instance_dir, exist_ok=True)
    if cluster_vis_dir is not None:
        os.makedirs(cluster_vis_dir, exist_ok=True)
    anno_list = anno_config.AnnoList().anno_list_all_reverse
    bandwidth_dict = anno_config.bandwidth_dict
    instances = np.zeros_like(seg_pred) - 1
    n_clusters_list = []
    n_clusters_list.append(0)
    for class_id in range(1, 31):
        bandwidth = bandwidth_dict[class_id]
        pts = points[np.where(seg_pred == class_id)]
        if pts.shape[0] <= 4:
            continue
        class_name = anno_list[class_id]
        class_id_idx = np.where(seg_pred == class_id)
        offset = offset_pred[class_id_idx]
        pts -= offset
        pts *= 350
        pts += 350
        if cluster_vis_dir is None:
            inst_labels, n_clusters = mean_shfit(pts, bandwidth, None)
        else:
            inst_labels, n_clusters = mean_shfit(pts, bandwidth, \
                os.path.join(cluster_vis_dir, basename + f"{class_name}.png"))
        n_clusters_list.append(n_clusters)
        instances[class_id_idx] = inst_labels + sum(n_clusters_list[:-1])
    assert instances.shape == seg_pred.shape
    data = {"instances": instances, "semantics": seg_pred}
    save_path = os.path.join(pred_instance_dir, basename + ".npy")
    np.save(save_path, data)
    return True
