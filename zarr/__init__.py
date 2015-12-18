# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.ext import Chunk, Array, SynchronizedChunk, blosc_version
from zarr.core import empty, zeros, ones, full, array
from zarr import defaults
from zarr.version import version as __version__
