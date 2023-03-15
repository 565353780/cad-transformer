#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

from cad_transformer.Method.graph import svg2graph


def test():
    input_dir = "/home/chli/chLi/FloorPlanCAD/svg/train/"
    output_dir = "/home/chli/chLi/FloorPlanCAD/npy_test/train/"
    max_degree = 128
    visualize = False

    svg_filename_list = os.listdir(input_dir)
    print("[TEST][graph::test]")
    print("\t start test svg2graph...")
    for filename in tqdm(svg_filename_list):
        if filename[-4:] != ".svg":
            continue

        svg_file_path = input_dir + filename

        svg2graph(svg_file_path, output_dir, max_degree, visualize)
    return True
