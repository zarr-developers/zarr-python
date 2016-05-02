# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import zlib
import bz2


import numpy as np


from zarr.compat import text_type


registry = dict()


try:
    from zarr import blosc
except ImportError:
    pass
else:

    class BloscCompressor(object):

        default_cname = 'blosclz'
        default_clevel = 5
        default_shuffle = 1

        def __init__(self, compression_opts):
            # at this point we expect compression_opts to be fully specified
            # and normalized
            cname = compression_opts['cname']
            if isinstance(cname, text_type):
                cname = cname.encode('ascii')
            self.cname = cname
            self.clevel = compression_opts['clevel']
            self.shuffle = compression_opts['shuffle']

        @classmethod
        def normalize_opts(cls, compression_opts):
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
                raise ValueError('blosc internal compressor not available: %s'
                                 % cname)

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
    def normalize_opts(cls, compression_opts):
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


class BZ2Compressor(object):

    default_level = 1

    def __init__(self, compression_opts):
        self.level = compression_opts

    @classmethod
    def normalize_opts(cls, compression_opts):
        """Convenience function to normalize compression options."""
        if compression_opts is None:
            level = cls.default_level
        else:
            level = int(compression_opts)
        if level < 1 or level > 9:
            raise ValueError('invalid compression level: %s' % level)
        return level

    # noinspection PyMethodMayBeStatic
    def decompress(self, cdata, array):
        data = bz2.decompress(cdata)
        src = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
        np.copyto(array, src)

    def compress(self, array):
        data = array.tobytes()
        return bz2.compress(data, self.level)


registry['bz2'] = BZ2Compressor


try:
    import lzma
except ImportError:  # pragma: no cover
    pass
else:

    class LZMACompressor(object):

        default_format = lzma.FORMAT_XZ
        default_check = lzma.CHECK_NONE
        default_preset = 1
        default_filters = None

        def __init__(self, compression_opts):
            # at this point we expect compression_opts to be fully specified
            # and normalized
            self.format = compression_opts['format']
            self.check = compression_opts['check']
            self.preset = compression_opts['preset']
            self.filters = compression_opts['filters']

        @classmethod
        def normalize_opts(cls, compression_opts):
            """Convenience function to normalize compression options."""

            if compression_opts is None:
                compression_opts = dict()

            format = compression_opts.get('format', None)
            check = compression_opts.get('check', None)
            preset = compression_opts.get('preset', None)
            filters = compression_opts.get('filters', None)

            # normalize format
            if format is None:
                format = cls.default_format
            if format not in [lzma.FORMAT_XZ, lzma.FORMAT_ALONE,
                              lzma.FORMAT_RAW]:
                raise ValueError('invalid format: %s' % format)

            # normalize check
            if check is None:
                check = cls.default_check
            if check not in [lzma.CHECK_NONE, lzma.CHECK_CRC32,
                             lzma.CHECK_CRC64, lzma.CHECK_SHA256]:
                raise ValueError('invalid check: %s' % check)

            # normalize preset
            if preset is None:
                preset = cls.default_preset
            if preset < 0 or preset > 9:
                raise ValueError('invalid preset: %s' % preset)

            # handle filters
            if filters:
                # cannot specify both preset and filters
                preset = None

            # construct normalized options
            compression_opts = dict(
                format=format, check=check, preset=preset, filters=filters
            )
            return compression_opts

        # noinspection PyMethodMayBeStatic
        def decompress(self, cdata, array):
            if self.format == lzma.FORMAT_RAW:
                # filters needed
                filters = self.filters
            else:
                # filters should not be specified
                filters = None
            data = lzma.decompress(cdata, format=self.format, filters=filters)
            src = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
            np.copyto(array, src)

        def compress(self, array):
            data = array.tobytes()
            return lzma.compress(data, format=self.format, check=self.check,
                                 preset=self.preset, filters=self.filters)

    registry['lzma'] = LZMACompressor


class NoCompressor(object):

    def __init__(self, compression_opts):
        pass

    @classmethod
    def normalize_opts(cls, compression_opts):
        if compression_opts is not None:
            raise ValueError('no compression options supported')
        return None

    # noinspection PyMethodMayBeStatic
    def decompress(self, cdata, array):
        src = np.frombuffer(cdata, dtype=array.dtype).reshape(array.shape)
        np.copyto(array, src)

    # noinspection PyMethodMayBeStatic
    def compress(self, array):
        data = array.tobytes()
        return data


registry[None] = NoCompressor


def get_compressor_cls(compression):
    try:
        return registry[compression]
    except KeyError:
        raise ValueError('compressor not available: %s' % compression)
