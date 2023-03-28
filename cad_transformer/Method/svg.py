#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET


def getSVGSize(svg_file_path):
    assert os.path.exists(svg_file_path)

    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    ns = root.tag[:-3]

    svg_size = 0

    for g in root.iter(ns + 'g'):
        svg_size += len(list(g.iter(ns + 'path')))
        svg_size += len(list(g.iter(ns + 'circle')))
        svg_size += len(list(g.iter(ns + 'ellipse')))
    return svg_size
