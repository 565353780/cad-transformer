#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.trainer import Trainer


def demo():
    model_file_path = "./output/20230219_14:10:00/model_best.pth"
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path)
    trainer.train(print_progress)
    return True
