#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cad_transformer.Test.model import test as test_model
from cad_transformer.Test.vit import test as test_vit
from cad_transformer.Test.graph import test as test_graph
from cad_transformer.Test.svg import test as test_svg

if __name__ == "__main__":
    #  test_model()
    #  test_vit()
    #  test_graph()
    test_svg()
