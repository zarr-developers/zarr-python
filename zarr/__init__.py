# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import multiprocessing
import atexit


from zarr.core import Array
from zarr.creation import create, array, empty, zeros, ones, full, open, \
    empty_like, zeros_like, ones_like, full_like, open_like, open_array
from zarr.storage import DictStore, DirectoryStore, ZipStore, init_array, \
    init_group, init_store
from zarr.hierarchy import group, open_group, Group
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.version import version as __version__
from zarr.filters import DeltaFilter, FixedScaleOffsetFilter, \
    QuantizeFilter, PackBitsFilter, CategoryFilter


try:
    from zarr import blosc
except ImportError:  # pragma: no cover
    pass
else:
    ncores = multiprocessing.cpu_count()
    blosc.init()
    blosc.set_nthreads(min(8, ncores))
    atexit.register(blosc.destroy)
