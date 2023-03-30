#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import gradio as gr

from cad_transformer.Module.detector import Detector


class DetectServer(object):

    def __init__(self,
                 model_file_path=None,
                 train_mode='all',
                 port=6006,
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
        assert os.path.exists(dxf_file_path)

        result = self.detector.detectDXFFile(dxf_file_path,
                                             self.print_progress)

        tmp_svg_file_path = './tmp/input.svg'
        result_image = self.detector.getResultImage(tmp_svg_file_path, result)
        result_image = result_image[..., ::-1]
        return result_image

    def getDXFResultImageInterface(self):
        inputs = gr.File()
        #FIXME: why invert_colors not work?
        outputs = gr.Image(label='Input+CADTransformer+DXFLayoutDetector',
                           invert_colors=True)

        interface = gr.Interface(self.getDXFResultImage, inputs, outputs)
        return interface

    def getDXFResultImageList(self, dxf_file):
        dxf_file_path = dxf_file.name
        assert os.path.exists(dxf_file_path)

        result = self.detector.detectDXFFile(dxf_file_path,
                                             self.print_progress)

        tmp_svg_file_path = './tmp/input.svg'
        result_image_list = self.detector.getResultImageList(
            tmp_svg_file_path, result)
        for i in range(len(result_image_list)):
            result_image_list[i] = result_image_list[i][..., ::-1]

        Input, CADTransformer, DXFLayoutDetector = result_image_list
        return Input, CADTransformer, DXFLayoutDetector

    def getDXFResultImageListInterface(self):
        inputs = gr.File()
        outputs = [
            gr.Image(label='Input', invert_colors=True),
            gr.Image(label='CADTransformer', invert_colors=True),
            gr.Image(label='DXFLayoutDetector', invert_colors=True)
        ]

        interface = gr.Interface(self.getDXFResultImageList, inputs, outputs)
        return interface

    def start(self):
        interface = self.getDXFResultImageListInterface()

        interface.launch(server_name='0.0.0.0', server_port=self.port)
        return True
