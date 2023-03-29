#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.detect_server import DetectServer


def demo():
    model_file_path = './output/20230322_16:04:51_wall/model_best.pth'
    train_mode = 'wall'
    port = 6006
    print_progress = True

    detect_server = DetectServer(model_file_path, train_mode, port,
                                 print_progress)
    detect_server.start()
    return True
