# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import multiprocessing
import atexit


from zarr.creation import create, array, empty, zeros, ones, full, open, \
    empty_like, zeros_like, ones_like, full_like, open_like
from zarr.storage import init_store, DirectoryStore
from zarr.core import Array
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer, \
    SynchronizedArray
from zarr.version import version as __version__


try:
    from zarr import blosc
except ImportError:  # pragma: no cover
    pass
else:
    ncores = multiprocessing.cpu_count()
    blosc.init()
    # diminishing returns beyond 4 threads?
    blosc.set_nthreads(min(4, ncores))
    atexit.register(blosc.destroy)
