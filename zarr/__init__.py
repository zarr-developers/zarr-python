# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import atexit


from zarr.core import empty, zeros, ones, full, array, open
from zarr.ext import blosc_version, init as _init, destroy as _destroy, \
    set_blosc_options
from zarr import defaults
from zarr import constants
from zarr.version import version as __version__


import multiprocessing
_cpu_count = multiprocessing.cpu_count()
_init()
set_blosc_options(use_context=False, nthreads=_cpu_count)
atexit.register(_destroy)
