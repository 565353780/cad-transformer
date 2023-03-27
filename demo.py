#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Demo.trainer import demo as demo_train
from cad_transformer.Demo.detector import (demo as demo_detect_dataset,
                                           demo_svg as demo_detect_svg,
                                           demo_dxf as demo_detect_dxf,
                                           demo_dxf_folder as
                                           demo_detect_dxf_folder)

if __name__ == "__main__":
    #  demo_train()
    #  demo_detect_dataset()
    #  demo_detect_svg()
    demo_detect_dxf()
    #  demo_detect_dxf_folder()
