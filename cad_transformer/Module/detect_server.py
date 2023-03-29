#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import gradio as gr

from cad_transformer.Module.detector import Detector


class DetectServer(object):

    def __init__(self,
                 model_file_path=None,
                 train_mode='all',
                 port=9363,
                 print_progress=False):
        self.model_file_path = None
        self.train_mode = train_mode
        self.port = port
        self.print_progress = print_progress

        if model_file_path is not None:
            self.loadDetector(model_file_path, train_mode)
        return

    def loadDetector(self, model_file_path, train_mode='all'):
        self.detector = Detector(train_mode)
        self.detector.loadModel(model_file_path)
        return True

    def getDXFResultImage(self, dxf_file):
        dxf_file_path = dxf_file.name
        print(dxf_file_path)
        assert os.path.exists(dxf_file_path)

        result = self.detector.detectDXFFile(dxf_file_path,
                                             self.print_progress)

        tmp_svg_file_path = './tmp/input.svg'
        result_image = self.detector.getResultImage(tmp_svg_file_path, result)
        return result_image

    def start(self):
        inputs = gr.File()
        outputs = gr.Image()

        interface = gr.Interface(self.getDXFResultImage, inputs, outputs)

        interface.launch(server_name='0.0.0.0', server_port=self.port)
        return True
