#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import xml.etree.ElementTree as ET
from svgpathtools import parse_path, svg2paths, svg2paths2


def test():
    svg_folder_path = "/home/chli/chLi/FloorPlanCAD/svg/train/"

    svg_filename_list = os.listdir(svg_folder_path)
    for filename in svg_filename_list:
        if filename[-4:] != ".svg":
            continue

        svg_file_path = svg_folder_path + filename
        print(svg_file_path)

        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        ns = root.tag[:-3]
        minx, miny, width, height = [
            int(float(x)) for x in root.attrib['viewBox'].split(' ')
        ]
        half_width = width / 2
        half_height = height / 2

        for g in root.iter(ns + 'g'):
            for path in g.iter(ns + 'path'):
                path_repre = parse_path(path.attrib['d'])
                print(path_repre)
                print(path.attrib.keys())
                if 'semantic-id' in path.attrib:
                    print(path.attrib['semantic-id'])
                else:
                    print('semantic-id not exist! set to 0')
                if 'instance-id' in path.attrib:
                    print(path.attrib['instance-id'])
                else:
                    print('instance-id not exist! set to -1')
        return

        paths, attribs = svg2paths(svg_file_path)
        print(len(paths))
        for i in range(len(paths)):
            print(i)
            print("[INFO][svg::test]")
            print("\t reading paths[" + str(i + 1) + "]...")
            print("paths:")
            print(paths[i])
            print("attribs:")
            print(attribs[i])
            return

        if len(paths) == 0:
            continue

        paths, attribs, svg_attribs = svg2paths2(svg_file_path)
        for i in range(len(paths)):
            print(i)
            print("[INFO][svg::test]")
            print("\t reading paths[" + str(i + 1) + "]...")
            print("paths:")
            print(paths[i])
            print("attribs:")
            print(attribs[i])
            print("svg_attribs:")
            print(svg_attribs[i])
    return True
