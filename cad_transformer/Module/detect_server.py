#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr

from cad_transformer.Module.detector import Detector


class DetectServer(object):

    def __init__(self, model_file_path):
        self.detector = Detector(model_file_path)
        return
