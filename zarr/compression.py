# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import zlib


import numpy as np


from zarr import blosc
from zarr.compat import PY2, text_type, binary_type


registry = dict()


class BloscCompressor(object):

    default_cname = 'blosclz'
    default_clevel = 5
    default_shuffle = 1

    def __init__(self, compression_opts):
        # at this point we expect compression_opts to be fully specified and
        # normalized
        if PY2:
            self.cname = compression_opts['cname']
        else:
            self.cname = compression_opts['cname'].encode('ascii')
        self.clevel = compression_opts['clevel']
        self.shuffle = compression_opts['shuffle']

    @classmethod
    def normalize_compression_opts(cls, compression_opts):
        """Convenience function to normalize compression options."""

        if compression_opts is None:
            compression_opts = dict()
        cname = compression_opts.get('cname', None)
        clevel = compression_opts.get('clevel', None)
        shuffle = compression_opts.get('shuffle', None)

        # determine internal compression library
        cname = cname if cname is not None else cls.default_cname

        # check internal compressor is available
        if blosc.compname_to_compcode(cname) < 0:
            raise ValueError('blosc internal compressor not available: %s' %
                             cname)

        # determine compression level
        clevel = clevel if clevel is not None else cls.default_clevel
        clevel = int(clevel)
        if clevel < 0 or clevel > 9:
            raise ValueError('invalid compression level: %s' % clevel)

        # determine shuffle filter
        shuffle = shuffle if shuffle is not None else cls.default_shuffle
        shuffle = int(shuffle)
        if shuffle not in [0, 1, 2]:
            raise ValueError('invalid shuffle: %s' % shuffle)

        # construct normalised options
        compression_opts = dict(
            cname=cname, clevel=clevel, shuffle=shuffle
        )
        return compression_opts

    # noinspection PyMethodMayBeStatic
    def decompress(self, cdata, array):
        blosc.decompress(cdata, array)

    def compress(self, array):
        return blosc.compress(array, self.cname, self.clevel, self.shuffle)


registry['blosc'] = BloscCompressor


class ZlibCompressor(object):

    default_level = 1

    def __init__(self, compression_opts):
        self.level = compression_opts

    @classmethod
    def normalize_compression_opts(cls, compression_opts):
        """Convenience function to normalize compression options."""
        if compression_opts is None:
            level = cls.default_level
        else:
            level = int(compression_opts)
        if level < 0 or level > 9:
            raise ValueError('invalid compression level: %s' % level)
        return level

    # noinspection PyMethodMayBeStatic
    def decompress(self, cdata, array):
        data = zlib.decompress(cdata)
        src = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
        np.copyto(array, src)

    def compress(self, array):
        data = array.tobytes()
        return zlib.compress(data, self.level)


registry['zlib'] = ZlibCompressor
registry['gzip'] = ZlibCompressor  # alias


def get_compressor_cls(compression):
    try:
        return registry[compression]
    except KeyError:
        raise ValueError('compressor not available: %s' % compression)
