#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.trainer import Trainer


def demo():
    model_file_path = './output/20230322_16:04:51_wall/model_last.pth'
    #  model_file_path = ''
    train_mode = 'wall'
    load_model_only = False
    print_progress = True

    trainer = Trainer(train_mode)
    trainer.loadModel(model_file_path, load_model_only)
    trainer.train(print_progress)
    return True
