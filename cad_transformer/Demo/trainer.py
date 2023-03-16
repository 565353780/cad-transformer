#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.trainer import Trainer


def demo():
    model_file_path = "./output/20230316_23:53:57/model_best.pth"
    load_model_only = False
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, load_model_only)
    trainer.train(print_progress)
    return True
