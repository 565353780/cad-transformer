#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from cad_transformer.Method.path import createFileFolder, renameFile, removeFile


def transDXFToSVG(dxf_file_path, save_svg_file_path):
    assert os.path.exists(dxf_file_path)

    tmp_svg_file_path = save_svg_file_path[:-4] + '_tmp.svg'
    createFileFolder(tmp_svg_file_path)

    removeFile(tmp_svg_file_path)
    removeFile(save_svg_file_path)

    try:
        doc, auditor = recover.readfile(dxf_file_path)
    except IOError:
        print(f'Not a DXF file or a generic I/O error.')
        sys.exit(1)
    except DXFStructureError:
        print(f'Invalid or corrupted DXF file.')
        sys.exit(2)

    assert not auditor.has_errors

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    #  fig.savefig(tmp_png_file_path, dpi=300)
    fig.savefig(tmp_svg_file_path)

    renameFile(tmp_svg_file_path, save_svg_file_path)
    return True
