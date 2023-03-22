#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.detector import Detector


def demo():
    model_file_path = "./output/20230320_05:30:02_wall/model_best.pth"
    train_mode = 'wall'
    print_progress = True

    detector = Detector(train_mode)
    detector.loadModel(model_file_path)
    detector.detectDataset(print_progress)
    return True
