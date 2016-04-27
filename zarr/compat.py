# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import sys


PY2 = sys.version_info[0] == 2


if PY2:

    def itervalues(d, **kw):
        return d.itervalues(**kw)

else:

    def itervalues(d, **kw):
        return iter(d.values(**kw))
