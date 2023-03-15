#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import xml.etree.ElementTree as ET
from copy import deepcopy
from bs4 import BeautifulSoup
from svgpathtools import parse_path

from cad_transformer.Method.dist import get_nn


def visualize_graph(root, centers, nns, vis_path):
    '''Visualization of the constructed graph for verification
    '''
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    g = ET.SubElement(
        root, 'g', {
            'clip-path': 'url(#clipId0)',
            'fill': 'none',
            'stroke': 'rgb(255,0,0)',
            'stroke-width': '0.25',
            'tag': 'g'
        })

    # visualize center points
    for i in range(len(centers)):
        s0cx, s0cy = centers[i]
        ET.SubElement(
            g, 'circle', {
                'cx': f'{s0cx}',
                'cy': f'{s0cy}',
                "r": "0.1",
                "stroke": "rgb(255,0,0)",
                "fill": "rgb(255,0,0)",
                'tag': 'circle'
            })

    # visualize NNs
    for i in range(len(centers[:1])):
        s0cx, s0cy = centers[i]
        ET.SubElement(
            g, 'circle', {
                'cx': f'{s0cx}',
                'cy': f'{s0cy}',
                "r": "0.5",
                "stroke": "rgb(255,0,0)",
                "fill": "rgb(255,0,0)",
                'tag': 'circle'
            })
        for j in range(len(nns[i][:16])):
            jj = nns[i][j]
            s0cx, s0cy = centers[jj]
            ET.SubElement(
                g, 'circle', {
                    'cx': f'{s0cx}',
                    'cy': f'{s0cy}',
                    "r": "0.2",
                    "stroke": f"rgb(0,{15*j},0)",
                    "fill": f"rgb(0,{15*j},0)",
                    'tag': 'circle'
                })
    prettyxml = BeautifulSoup(ET.tostring(root, 'utf-8'), "xml").prettify()
    with open(vis_path, "w") as f:
        f.write(prettyxml)
    return True


def svg2graph(svg_path,
              output_dir,
              max_degree,
              visualize,
              avoid_self_idx=False):
    '''Construct the graph of each drawing
    '''
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = root.tag[:-3]
    minx, miny, width, height = [
        int(float(x)) for x in root.attrib['viewBox'].split(' ')
    ]
    half_width = width / 2
    half_height = height / 2

    # get all segments
    segments = []
    nodes = []
    centers = []
    classes = []
    instances = []
    #  starts_ends = []
    for g in root.iter(ns + 'g'):
        # path
        for path in g.iter(ns + 'path'):
            try:
                path_repre = parse_path(path.attrib['d'])
            except Exception as _:
                raise RuntimeError("Parse path failed!{}, {}".format(
                    svg_path, path.attrib['d']))
            start = path_repre.point(0)
            end = path_repre.point(1)
            segments.append([start.real, start.imag, end.real, end.imag])
            # starts_ends.append([start.real, start.imag, end.real, end.imag, end.real, end.imag, start.real, start.imag])
            mid = path_repre.point(0.5)
            # length = math.sqrt((start.real - end.real) ** 2 + (start.imag - end.imag) ** 2)
            length = path_repre.length()
            nodes.append([
                length / width, (mid.real - minx) / width,
                (mid.imag - miny) / height, 1, 0, 0
            ])
            centers.append([mid.real, mid.imag])
            if 'semantic-id' in path.attrib:
                classes.append([int(path.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in path.attrib:
                instances.append([int(path.attrib['instance-id'])])
            else:
                instances.append([-1])
        # circle
        for circle in g.iter(ns + 'circle'):
            cx = float(circle.attrib['cx'])
            cy = float(circle.attrib['cy'])
            r = float(circle.attrib['r'])
            segments.append([cx - r, cy, cx + r, cy])
            # starts_ends.append([cx - r, cy, cx + r, cy, cx + r, cy, cx - r, cy])
            nodes.append([
                r * 2.0 / width, (cx - minx) / width, (cy - miny) / height, 0,
                1, 0
            ])
            centers.append([cx, cy])
            if 'semantic-id' in circle.attrib:
                classes.append([int(circle.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in circle.attrib:
                instances.append([int(circle.attrib['instance-id'])])
            else:
                instances.append([-1])
        # ellipse
        for ellipse in g.iter(ns + 'ellipse'):
            cx = float(ellipse.attrib['cx'])
            cy = float(ellipse.attrib['cy'])
            rx = float(ellipse.attrib['rx'])
            ry = float(ellipse.attrib['ry'])
            segments.append([cx - rx, cy, cx + rx, cy])
            # starts_ends.append([cx - rx, cy, cx + rx, cy, cx + r, cy, cx - r, cy])
            nodes.append([(rx + ry) / width, (cx - minx) / width,
                          (cy - miny) / height, 0, 0, 1])
            centers.append([cx, cy])
            if 'semantic-id' in ellipse.attrib:
                classes.append([int(ellipse.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in ellipse.attrib:
                instances.append([int(ellipse.attrib['instance-id'])])
            else:
                instances.append([-1])

    segments = np.array(segments)
    nns = get_nn(deepcopy(segments), max_degree, avoid_self_idx)
    if segments.shape[0] < 2:
        print('Warning: too few segments')
        return

    basename = os.path.basename(svg_path)

    if visualize:
        vis_path = os.path.join(output_dir, './visualize/', basename)
        print(f"vis to {vis_path}")
        visualize_graph(root, centers, nns, vis_path)

    centers_norm = []
    for c in centers:
        centers_norm.append([(c[0] - half_width) / half_width,
                             (c[1] - half_height) / half_height])
    data_gcn = {
        "nd_ft": nodes,
        "ct": centers,
        "cat": classes,
        "ct_norm": centers_norm,
        "nns": nns,
        "inst": instances
    }
    npy_path = os.path.join(output_dir, basename.replace(".svg", ".npy"))
    np.save(npy_path, data_gcn)
    return True
