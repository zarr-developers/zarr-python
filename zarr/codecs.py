# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from numcodecs import *
from numcodecs.registry import codec_registry
# alias gzip for compatibility with h5py (TODO migrate this to numcodecs)
codec_registry['gzip'] = Zlib
