#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.detector import Detector


def demo():
    model_file_path = "./output/20230322_16:04:51_wall/model_best.pth"
    train_mode = 'wall'
    load_num_max = 20
    print_progress = True

    detector = Detector(train_mode)
    detector.loadModel(model_file_path)
    detector.detectAllDataset(load_num_max, print_progress)
    return True


def demo_svg():
    model_file_path = "./output/20230322_16:04:51_wall/model_best.pth"
    train_mode = 'wall'
    load_num_max = 20
    print_progress = True

    detector = Detector(train_mode)
    detector.loadModel(model_file_path)
    detector.detectAllDatasetBySVGFile(load_num_max, print_progress)
    return True


def demo_dxf():
    model_file_path = "./output/20230322_16:04:51_wall/model_best.pth"
    train_mode = 'wall'
    dxf_folder_path = '/home/chli/chLi/CAD/给坤哥测试用例/'
    print_progress = True

    detector = Detector(train_mode)
    detector.loadModel(model_file_path)
    detector.detectDXFFolder(dxf_folder_path, print_progress)
    return True
