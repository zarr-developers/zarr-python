# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import zlib
import bz2
import array
import math
import multiprocessing
import atexit


import numpy as np


from zarr.compat import text_type, binary_type


codec_registry = dict()


def get_codec(config):
    """Obtain a codec for the given configuration.

    Parameters
    ----------
    config : dict-like
        Configuration object.

    Returns
    -------
    codec : Codec

    """
    codec_id = config.pop('id', None)
    cls = codec_registry.get(codec_id, None)
    if cls is None:
        raise ValueError('codec not available: %r' % codec_id)
    return cls.from_config(config)


class Codec(object):  # pragma: no cover
    """Codec abstract base class."""

    # override in sub-class
    id = None

    def encode(self, buf):
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol or `array.array`.

        Returns
        -------
        enc : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol or `array.array`.

        """
        # override in sub-class
        raise NotImplementedError

    def decode(self, buf, out=None):
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol or `array.array`.
        out : buffer-like, optional
            Buffer to store decoded data.

        Returns
        -------
        out : buffer-like
            Decoded data. May be any object supporting the new-style buffer
            protocol or `array.array`.

        """
        # override in sub-class
        raise NotImplementedError

    def get_config(self):
        """Return a dictionary holding configuration parameters for this
        codec. All values must be compatible with JSON encoding."""
        # override in sub-class
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        """Instantiate from a configuration object."""
        # override if need special decoding of config values
        return cls(**config)


def _buffer_copy(buf, out=None):

    if out is None:
        # no-op
        return buf

    # handle ndarray destination
    if isinstance(out, np.ndarray):

        # view source as destination dtype
        if isinstance(buf, np.ndarray):
            buf = buf.view(dtype=out.dtype).reshape(-1, order='A')
        else:
            buf = np.frombuffer(buf, dtype=out.dtype)

        # ensure shapes are compatible
        if buf.shape != out.shape:
            if out.flags.f_contiguous:
                order = 'F'
            else:
                order = 'C'
            buf = buf.reshape(out.shape, order=order)

        # copy via numpy
        np.copyto(out, buf)

    # handle generic buffer destination
    else:

        # obtain memoryview of destination
        dest = memoryview(out)

        # ensure source is 1D
        if isinstance(buf, np.ndarray):
            buf = buf.reshape(-1, order='A')
            # try to match itemsize
            dtype = 'u%s' % dest.itemsize
            buf = buf.view(dtype=dtype)

        # try to copy via memoryview
        dest[:] = buf

    return out


class Zlib(Codec):
    """Provides compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'zlib'

    def __init__(self, level=1):
        self.level = level

    def encode(self, buf):

        # if numpy array, can only handle C contiguous directly
        if isinstance(buf, np.ndarray) and not buf.flags.c_contiguous:
            buf = buf.tobytes(order='A')

        # do compression
        return zlib.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        # do decompression
        dec = zlib.decompress(buf)

        # handle destination
        return _buffer_copy(dec, out)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['level'] = self.level
        return config

    def __repr__(self):
        r = '%s(level=%s)' % (type(self).__name__, self.level)
        return r


codec_registry[Zlib.codec_id] = Zlib
codec_registry['gzip'] = Zlib  # alias


class BZ2(Codec):
    """Provides compression using bzip2 via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'bz2'

    def __init__(self, level=1):
        self.level = level

    def encode(self, buf):

        # if numpy array, can only handle C contiguous directly
        if isinstance(buf, np.ndarray) and not buf.flags.c_contiguous:
            buf = buf.tobytes(order='A')

        # do compression
        return bz2.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        # BZ2 cannot handle ndarray directly at all, coerce everything to
        # memoryview
        if not isinstance(buf, array.array):
            buf = memoryview(buf)

        # do decompression
        dec = bz2.decompress(buf)

        # handle destination
        return _buffer_copy(dec, out)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['level'] = self.level
        return config

    def __repr__(self):
        r = '%s(level=%s)' % (type(self).__name__, self.level)
        return r


codec_registry[BZ2.codec_id] = BZ2


try:
    import lzma
except ImportError:  # pragma: no cover
    pass
else:

    # noinspection PyShadowingBuiltins
    class LZMA(Codec):
        """Provides compression using lzma via the Python standard library
        (only available under Python 3).

        Parameters
        ----------
        format : integer, optional
            One of the lzma format codes, e.g., ``lzma.FORMAT_XZ``.
        check : integer, optional
            One of the lzma check codes, e.g., ``lzma.CHECK_NONE``.
        preset : integer, optional
            An integer between 0 and 9 inclusive, specifying the compression
            level.
        filters : list, optional
            A list of dictionaries specifying compression filters. If
            filters are provided, 'preset' must be None.

        """

        codec_id = 'lzma'

        def __init__(self, format=1, check=-1, preset=None, filters=None):
            self.format = format
            self.check = check
            self.preset = preset
            self.filters = filters

        def encode(self, buf):

            # if numpy array, can only handle C contiguous directly
            if isinstance(buf, np.ndarray) and not buf.flags.c_contiguous:
                buf = buf.tobytes(order='A')

            # do compression
            return lzma.compress(buf, format=self.format, check=self.check,
                                 preset=self.preset, filters=self.filters)

        def decode(self, buf, out=None):

            # setup filters
            if self.format == lzma.FORMAT_RAW:
                # filters needed
                filters = self.filters
            else:
                # filters should not be specified
                filters = None

            # do decompression
            dec = lzma.decompress(buf, format=self.format, filters=filters)

            # handle destination
            return _buffer_copy(dec, out)

        def get_config(self):
            config = dict()
            config['id'] = self.codec_id
            config['format'] = self.format
            config['check'] = self.check
            config['preset'] = self.preset
            config['filters'] = self.filters
            return config

        def __repr__(self):
            r = '%s(format=%r, check=%r, preset=%r, filters=%r)' % \
                (type(self).__name__, self.format, self.check, self.preset,
                 self.filters)
            return r

    codec_registry[LZMA.codec_id] = LZMA

try:
    from zarr import blosc
except ImportError:  # pragma: no cover
    pass
else:

    class Blosc(Codec):
        """Provides compression using the blosc meta-compressor.

        Parameters
        ----------
        cname : string, optional
            A string naming one of the compression algorithms available
            within blosc, e.g., 'blosclz', 'lz4', 'zlib' or 'snappy'.
        clevel : integer, optional
            An integer between 0 and 9 specifying the compression level.
        shuffle : integer, optional
            Either 0 (no shuffle), 1 (byte shuffle) or 2 (bit shuffle).

        """

        codec_id = 'blosc'

        def __init__(self, cname='lz4', clevel=5, shuffle=1):
            if isinstance(cname, text_type):
                cname = cname.encode('ascii')
            self.cname = cname
            self.clevel = clevel
            self.shuffle = shuffle

        def encode(self, buf):
            return blosc.compress(buf, self.cname, self.clevel, self.shuffle)

        def decode(self, buf, out=None):
            return blosc.decompress(buf, out)

        def get_config(self):
            config = dict()
            config['id'] = self.codec_id
            config['cname'] = text_type(self.cname, 'ascii')
            config['clevel'] = self.clevel
            config['shuffle'] = self.shuffle
            return config

        def __repr__(self):
            r = '%s(cname=%r, clevel=%r, shuffle=%r)' % \
                (type(self).__name__, text_type(self.cname, 'ascii'),
                 self.clevel, self.shuffle)
            return r

    codec_registry[Blosc.codec_id] = Blosc

    # initialize blosc
    ncores = multiprocessing.cpu_count()
    blosc.init()
    blosc.set_nthreads(min(8, ncores))
    atexit.register(blosc.destroy)


def _ndarray_from_buffer(buf, dtype):
    if isinstance(buf, np.ndarray):
        arr = buf.reshape(-1, order='A').view(dtype)
    else:
        arr = np.frombuffer(buf, dtype=dtype)
    return arr


class Delta(Codec):
    """Filter to encode data as the difference between adjacent values.

    Parameters
    ----------
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Notes
    -----
    If `astype` is an integer data type, please ensure that it is
    sufficiently large to store encoded values. No checks are made and data
    may become corrupted due to integer overflow if `astype` is too small.
    Note also that the encoded data for each chunk includes the absolute
    value of the first element in the chunk, and so the encoded data type in
    general needs to be large enough to store absolute values from the array.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> x = np.arange(100, 120, 2, dtype='i8')
    >>> f = zarr.Delta(dtype='i8', astype='i1')
    >>> y = f.encode(x)
    >>> y
    array([100,   2,   2,   2,   2,   2,   2,   2,   2,   2], dtype=int8)
    >>> z = f.decode(y)
    >>> z
    array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

    """  # flake8: noqa

    codec_id = 'delta'

    def __init__(self, dtype, astype=None):
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)

    def encode(self, buf):

        # view input data as 1D array
        arr = _ndarray_from_buffer(buf, self.dtype)

        # setup encoded output
        enc = np.empty_like(arr, dtype=self.astype)

        # set first element
        enc[0] = arr[0]

        # compute differences
        enc[1:] = np.diff(arr)

        return enc

    def decode(self, buf, out=None):

        # view encoded data as 1D array
        enc = _ndarray_from_buffer(buf, self.astype)

        # setup decoded output
        if isinstance(out, np.ndarray):
            # optimization, can decode directly to out
            dec = out.reshape(-1, order='A')
            copy_needed = False
        else:
            dec = np.empty_like(enc, dtype=self.dtype)
            copy_needed = True

        # decode differences
        np.cumsum(enc, out=dec)

        # handle output
        if copy_needed:
            out = _buffer_copy(dec, out)

        return out

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['dtype'] = self.dtype.str
        config['astype'] = self.astype.str
        return config

    def __repr__(self):
        r = '%s(dtype=%s' % (type(self).__name__, self.dtype)
        if self.astype != self.dtype:
            r += ', astype=%s' % self.astype
        r += ')'
        return r


codec_registry[Delta.codec_id] = Delta


class FixedScaleOffset(Codec):
    """Simplified version of the scale-offset filter available in HDF5.
    Applies the transformation `(x - offset) * scale` to all chunks. Results
    are rounded to the nearest integer but are not packed according to the
    minimum number of bits.

    Parameters
    ----------
    offset : float
        Value to subtract from data.
    scale : int
        Value to multiply by data.
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Notes
    -----
    If `astype` is an integer data type, please ensure that it is
    sufficiently large to store encoded values. No checks are made and data
    may become corrupted due to integer overflow if `astype` is too small.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> x = np.linspace(1000, 1001, 10, dtype='f8')
    >>> x
    array([ 1000.        ,  1000.11111111,  1000.22222222,  1000.33333333,
            1000.44444444,  1000.55555556,  1000.66666667,  1000.77777778,
            1000.88888889,  1001.        ])
    >>> f1 = zarr.FixedScaleOffset(offset=1000, scale=10, dtype='f8', astype='u1')
    >>> y1 = f1.encode(x)
    >>> y1
    array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10], dtype=uint8)
    >>> z1 = f1.decode(y1)
    >>> z1
    array([ 1000. ,  1000.1,  1000.2,  1000.3,  1000.4,  1000.6,  1000.7,
            1000.8,  1000.9,  1001. ])
    >>> f2 = zarr.FixedScaleOffset(offset=1000, scale=10**2, dtype='f8', astype='u1')
    >>> y2 = f2.encode(x)
    >>> y2
    array([  0,  11,  22,  33,  44,  56,  67,  78,  89, 100], dtype=uint8)
    >>> z2 = f2.decode(y2)
    >>> z2
    array([ 1000.  ,  1000.11,  1000.22,  1000.33,  1000.44,  1000.56,
            1000.67,  1000.78,  1000.89,  1001.  ])
    >>> f3 = zarr.FixedScaleOffset(offset=1000, scale=10**3, dtype='f8', astype='u2')
    >>> y3 = f3.encode(x)
    >>> y3
    array([   0,  111,  222,  333,  444,  556,  667,  778,  889, 1000], dtype=uint16)
    >>> z3 = f3.decode(y3)
    >>> z3
    array([ 1000.   ,  1000.111,  1000.222,  1000.333,  1000.444,  1000.556,
            1000.667,  1000.778,  1000.889,  1001.   ])

    """  # flake8: noqa

    codec_id = 'fixedscaleoffset'

    def __init__(self, offset, scale, dtype, astype=None):
        self.offset = offset
        self.scale = scale
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)

    def encode(self, buf):

        # interpret buffer as 1D array
        arr = _ndarray_from_buffer(buf, self.dtype)

        # compute scale offset
        enc = (arr - self.offset) * self.scale

        # round to nearest integer
        enc = np.around(enc)

        # convert dtype
        enc = enc.astype(self.astype, copy=False)

        return enc

    def decode(self, buf, out=None):

        # interpret buffer as 1D array
        enc = _ndarray_from_buffer(buf, self.astype)

        # decode scale offset
        dec = (enc / self.scale) + self.offset

        # convert dtype
        dec = dec.astype(self.dtype, copy=False)

        # handle output
        return _buffer_copy(dec, out)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['astype'] = self.astype.str
        config['dtype'] = self.dtype.str
        config['scale'] = self.scale
        config['offset'] = self.offset
        return config

    def __repr__(self):
        r = '%s(scale=%s, offset=%s, dtype=%s' % \
            (type(self).__name__, self.scale, self.offset, self.dtype)
        if self.astype != self.dtype:
            r += ', astype=%s' % self.astype
        r += ')'
        return r

codec_registry[FixedScaleOffset.codec_id] = FixedScaleOffset


class Quantize(Codec):
    """Lossy filter to reduce the precision of floating point data.

    Parameters
    ----------
    digits : int
        Desired precision (number of decimal digits).
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10, dtype='f8')
    >>> x
    array([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
            0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])
    >>> f1 = zarr.Quantize(digits=1, dtype='f8')
    >>> y1 = f1.encode(x)
    >>> y1
    array([ 0.    ,  0.125 ,  0.25  ,  0.3125,  0.4375,  0.5625,  0.6875,
            0.75  ,  0.875 ,  1.    ])
    >>> f2 = zarr.Quantize(digits=2, dtype='f8')
    >>> y2 = f2.encode(x)
    >>> y2
    array([ 0.       ,  0.109375 ,  0.21875  ,  0.3359375,  0.4453125,
            0.5546875,  0.6640625,  0.78125  ,  0.890625 ,  1.       ])
    >>> f3 = zarr.Quantize(digits=3, dtype='f8')
    >>> y3 = f3.encode(x)
    >>> y3
    array([ 0.        ,  0.11132812,  0.22265625,  0.33300781,  0.44433594,
            0.55566406,  0.66699219,  0.77734375,  0.88867188,  1.        ])

    """

    codec_id = 'quantize'

    def __init__(self, digits, dtype, astype=None):
        self.digits = digits
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)
        if self.dtype.kind != 'f' or self.astype.kind != 'f':
            raise ValueError('only floating point data types are supported')

    def encode(self, buf):

        # interpret buffer as 1D array
        arr = _ndarray_from_buffer(buf, self.dtype)

        # apply scaling
        precision = 10. ** -self.digits
        exp = math.log(precision, 10)
        if exp < 0:
            exp = int(math.floor(exp))
        else:
            exp = int(math.ceil(exp))
        bits = math.ceil(math.log(10. ** -exp, 2))
        scale = 2. ** bits
        enc = np.around(scale * arr) / scale

        # cast dtype
        enc = enc.astype(self.astype, copy=False)

        return enc

    def decode(self, buf, out=None):
        # filter is lossy, decoding is no-op
        dec = _ndarray_from_buffer(buf, self.astype)
        dec = dec.astype(self.dtype, copy=False)
        return _buffer_copy(dec, out)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['digits'] = self.digits
        config['dtype'] = self.dtype.str
        config['astype'] = self.astype.str
        return config

    def __repr__(self):
        r = '%s(digits=%s, dtype=%s' % \
            (type(self).__name__, self.digits, self.dtype)
        if self.astype != self.dtype:
            r += ', astype=%s' % self.astype
        r += ')'
        return r


codec_registry[Quantize.codec_id] = Quantize


class PackBits(Codec):
    """Filter to pack elements of a boolean array into bits in a uint8 array.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> f = zarr.PackBits()
    >>> x = np.array([True, False, False, True], dtype=bool)
    >>> y = f.encode(x)
    >>> y
    array([  4, 144], dtype=uint8)
    >>> z = f.decode(y)
    >>> z
    array([ True, False, False,  True], dtype=bool)

    Notes
    -----
    The first element of the encoded array stores the number of bits that
    were padded to complete the final byte.

    """

    codec_id = 'packbits'

    def __init__(self):
        pass

    def encode(self, buf):

        # view input as ndarray
        arr = _ndarray_from_buffer(buf, bool)

        # determine size of packed data
        n = arr.size
        n_bytes_packed = (n // 8)
        n_bits_leftover = n % 8
        if n_bits_leftover > 0:
            n_bytes_packed += 1

        # setup output
        enc = np.empty(n_bytes_packed + 1, dtype='u1')

        # store how many bits were padded
        if n_bits_leftover:
            n_bits_padded = 8 - n_bits_leftover
        else:
            n_bits_padded = 0
        enc[0] = n_bits_padded

        # apply encoding
        enc[1:] = np.packbits(arr)

        return enc

    def decode(self, buf, out=None):

        # view encoded data as ndarray
        enc = _ndarray_from_buffer(buf, 'u1')

        # find out how many bits were padded
        n_bits_padded = int(enc[0])

        # apply decoding
        dec = np.unpackbits(enc[1:])

        # remove padded bits
        if n_bits_padded:
            dec = dec[:-n_bits_padded]

        # view as boolean array
        dec = dec.view(bool)

        # handle destination
        return _buffer_copy(dec, out)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        return config

    def __repr__(self):
        r = '%s()' % type(self).__name__
        return r


codec_registry[PackBits.codec_id] = PackBits


def _ensure_bytes(l):
    if isinstance(l, binary_type):
        return l
    elif isinstance(l, text_type):
        return l.encode('ascii')
    else:
        raise ValueError('expected bytes, found %r' % l)


def _ensure_text(l):
    if isinstance(l, text_type):
        return l
    elif isinstance(l, binary_type):
        return text_type(l, 'ascii')
    else:
        raise ValueError('expected text, found %r' % l)


class Categorize(Codec):
    """Filter encoding categorical string data as integers.

    Parameters
    ----------
    labels : sequence of strings
        Category labels.
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> x = np.array([b'male', b'female', b'female', b'male', b'unexpected'])
    >>> x
    array([b'male', b'female', b'female', b'male', b'unexpected'],
          dtype='|S10')
    >>> f = zarr.Categorize(labels=[b'female', b'male'], dtype=x.dtype)
    >>> y = f.encode(x)
    >>> y
    array([2, 1, 1, 2, 0], dtype=uint8)
    >>> z = f.decode(y)
    >>> z
    array([b'male', b'female', b'female', b'male', b''],
          dtype='|S10')

    """

    codec_id = 'categorize'

    def __init__(self, labels, dtype, astype='u1'):
        self.dtype = np.dtype(dtype)
        if self.dtype.kind == 'S':
            self.labels = [_ensure_bytes(l) for l in labels]
        elif self.dtype.kind == 'U':
            self.labels = [_ensure_text(l) for l in labels]
        else:
            raise ValueError('data type not supported')
        self.astype = np.dtype(astype)

    def encode(self, buf):

        # view input as ndarray
        arr = _ndarray_from_buffer(buf, self.dtype)

        # setup output array
        enc = np.zeros_like(arr, dtype=self.astype)

        # apply encoding, reserving 0 for values not specified in labels
        for i, l in enumerate(self.labels):
            enc[arr == l] = i + 1

        return enc

    def decode(self, buf, out=None):

        # view encoded data as ndarray
        enc = _ndarray_from_buffer(buf, self.astype)

        # setup output
        if isinstance(out, np.ndarray):
            # optimization, decode directly to output
            dec = out.reshape(-1, order='A')
            copy_needed = False
        else:
            dec = np.zeros_like(enc, dtype=self.dtype)
            copy_needed = True

        # apply decoding
        for i, l in enumerate(self.labels):
            dec[enc == (i + 1)] = l

        # handle output
        if copy_needed:
            dec = _buffer_copy(dec, out)

        return dec

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['labels'] = [_ensure_text(l) for l in self.labels]
        config['dtype'] = self.dtype.str
        config['astype'] = self.astype.str
        return config

    def __repr__(self):
        # make sure labels part is not too long
        labels = repr(self.labels[:3])
        if len(self.labels) > 3:
            labels = labels[:-1] + ', ...]'
        r = '%s(dtype=%s, astype=%s, labels=%s)' % \
            (type(self).__name__, self.dtype, self.astype, labels)
        return r


codec_registry[Categorize.codec_id] = Categorize


__all__ = ['get_codec', 'codec_registry']
for _cls in codec_registry.values():
    __all__.append(_cls.__name__)
