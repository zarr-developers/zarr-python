# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.create import empty, zeros, ones, full, array, open, empty_like, \
    zeros_like, ones_like, full_like, open_like
from zarr.ext import blosc_version
from zarr import defaults
from zarr import constants
from zarr.version import version as __version__
