# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys


PY2 = sys.version_info[0] == 2


if PY2:  # pragma: py3 no cover

    text_type = unicode
    binary_type = str
    reduce = reduce
    from itertools import izip_longest as zip_longest

    class PermissionError(Exception):
        pass

    def OrderedDict_move_to_end(od, key):
        od[key] = od.pop(key)

    from collections import Mapping


else:  # pragma: py2 no cover

    text_type = str
    binary_type = bytes
    from functools import reduce
    from itertools import zip_longest
    PermissionError = PermissionError

    def OrderedDict_move_to_end(od, key):
        od.move_to_end(key)

    from collections.abc import Mapping
