# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import atexit


from zarr.chunk import zchunk, _blosc_init, _blosc_destroy, \
    _blosc_set_nthreads as blosc_set_nthreads, blosc_version
from zarr import defaults
from zarr.version import version as __version__


_blosc_init()
blosc_set_nthreads(1)
atexit.register(_blosc_destroy)
