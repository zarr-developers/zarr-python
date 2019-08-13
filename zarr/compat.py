# -*- coding: utf-8 -*-
# flake8: noqa

# TODO: we can probably eliminate all of this module
text_type = str
binary_type = bytes
from functools import reduce
from itertools import zip_longest

def OrderedDict_move_to_end(od, key):
    od.move_to_end(key)

from collections.abc import Mapping, MutableMapping
from os import scandir
