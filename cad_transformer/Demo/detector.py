#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.detector import Detector


def demo():
    model_file_path = "./output/20230219_14:10:00/model_best.pth"
    print_progress = True

    detector = Detector()
    detector.loadModel(model_file_path)
    detector.detectDataset(print_progress)
    return True
