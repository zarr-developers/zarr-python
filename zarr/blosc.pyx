# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as np


def decompress(bytes cdata, np.ndarray array):
    """Decompress data into a numpy array.

    Parameters
    ----------
    cdata : bytes
        Compressed data, including blosc header.
    array : ndarray
        Numpy array to decompress into.

    Notes
    -----
    Assumes that the size of the destination array is correct for the size of
    the uncompressed data.

    """
    # TODO
    pass


def compress(np.ndarray array, bytes cname, int clevel, int shuffle):
    """Compress data in a numpy array.

    Parameters
    ----------
    array : ndarray
        Numpy array containing data to be compressed.
    cname : bytes
        Name of compression library to use.
    clevel : int
        Compression level.
    shuffle : int
        Shuffle filter.

    Returns
    -------
    cdata : bytes
        Compressed data.

    """
    # TODO
    pass
