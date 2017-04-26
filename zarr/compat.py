# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys


import numpy as np


PY2 = sys.version_info[0] == 2


numpy_integer_types = np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, \
                      np.uint64


if PY2:  # pragma: no cover

    text_type = unicode
    binary_type = str
    integer_types = (int, long) + numpy_integer_types
    reduce = reduce

else:

    text_type = str
    binary_type = bytes
    integer_types = (int,) + numpy_integer_types
    from functools import reduce
