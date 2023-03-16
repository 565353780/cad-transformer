#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm

from cad_transformer.Config.anno_config import AnnoList

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def get_eval_criteria(epoch):
    eval = False
    if epoch < 50:
        if epoch % 5 == 0:
            eval = True
    if 50 < epoch < 1e5:
        if epoch % 5 == 0:
            eval = True
    if epoch == 0 or epoch == 1:
        eval = True
    return eval


def do_eval(model, loaders, summary_writer, cfg, step):
    print("[INFO][eval::do_eval]")
    print("\t start eval...")
    with torch.no_grad():
        model = model.eval()
        anno_list = AnnoList().anno_list_all_reverse
        class_num = len(anno_list)
        cnt_prd, cnt_gt, cnt_tp = \
            [torch.Tensor([0 for _ in range(class_num)]).cuda() for _ in range(3)]
        with tqdm(loaders, total=len(loaders), smoothing=0.9) as _tqdm:
            for image, xy, nns, target in _tqdm:
                seg_pred = model(image, xy, nns)
                seg_pred = seg_pred.contiguous().view(-1, cfg.num_class + 1)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]
                # Squeeze
                xy = xy.squeeze(0)

                for prd, gt in zip(pred_choice, target):
                    cnt_prd[prd] += 1
                    cnt_gt[gt] += 1
                    if prd == gt:
                        cnt_tp[gt] += 1
            # Accumulating
            for cls_id in range(class_num):
                precision = cnt_tp[cls_id] / (cnt_prd[cls_id] + 1e-4)
                recall = cnt_tp[cls_id] / (cnt_gt[cls_id] + 1e-4)
                f1 = (2 * precision * recall) / (precision + recall + 1e-4)
                summary_writer.add_scalar("F1/" + anno_list[cls_id], f1, step)
                summary_writer.add_scalar("Precision/" + anno_list[cls_id],
                                          precision, step)
                summary_writer.add_scalar("Recall/" + anno_list[cls_id],
                                          recall, step)
            tp = sum(cnt_tp[1:])
            gt = sum(cnt_gt[1:])
            pred = sum(cnt_prd[1:])
            precision = tp / pred
            recall = tp / gt
            f1 = (2 * precision * recall) / (precision + recall + 1e-4)
            summary_writer.add_scalar("Total/F1", f1, step)
            summary_writer.add_scalar("Total/Precision", precision, step)
            summary_writer.add_scalar("Total/Recall", recall, step)
    return f1
