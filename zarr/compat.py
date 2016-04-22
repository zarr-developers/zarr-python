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


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper
