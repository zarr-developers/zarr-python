# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


def human_readable_size(size):
    if size < 2**10:
        return "%s" % size
    elif size < 2**20:
        return "%.1fK" % (size / float(2**10))
    elif size < 2**30:
        return "%.1fM" % (size / float(2**20))
    elif size < 2**40:
        return "%.1fG" % (size / float(2**30))
    else:
        return "%.1fT" % (size / float(2**40))
