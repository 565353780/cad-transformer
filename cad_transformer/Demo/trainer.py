#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Module.trainer import Trainer


def demo():
    model_file_path = "./output/all_class_pretrained/model_best.pth"
    load_model_only = True
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, load_model_only)
    trainer.train(print_progress)
    return True
