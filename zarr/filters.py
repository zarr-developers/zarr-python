# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import math


import numpy as np


from zarr.meta import encode_dtype, decode_dtype
from zarr.compressors import registry as compressor_registry
from zarr.compat import text_type, binary_type


filter_registry = dict()


def _ndarray_from_buffer(buf, dtype):
    if isinstance(buf, np.ndarray):
        arr = buf.reshape(-1, order='A').view(dtype)
    else:
        arr = np.frombuffer(buf, dtype=dtype)
    return arr


class DeltaFilter(object):
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
    >>> f = zarr.DeltaFilter(dtype='i8', astype='i1')
    >>> y = f.encode(x)
    >>> y
    array([100,   2,   2,   2,   2,   2,   2,   2,   2,   2], dtype=int8)
    >>> z = f.decode(y)
    >>> z
    array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

    """  # flake8: noqa

    filter_name = 'delta'

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

    def decode(self, buf):
        # view encoded data as 1D array
        enc = _ndarray_from_buffer(buf, self.astype)
        # setup decoded output
        dec = np.empty_like(enc, dtype=self.dtype)
        # decode differences
        np.cumsum(enc, out=dec)
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['dtype'] = encode_dtype(self.dtype)
        config['astype'] = encode_dtype(self.astype)
        return config

    @classmethod
    def from_filter_config(cls, config):
        dtype = decode_dtype(config['dtype'])
        astype = decode_dtype(config['astype'])
        return cls(dtype=dtype, astype=astype)


filter_registry[DeltaFilter.filter_name] = DeltaFilter


class FixedScaleOffsetFilter(object):
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
    >>> f1 = zarr.FixedScaleOffsetFilter(offset=1000, scale=10, dtype='f8',
    ...                                  astype='u1')
    >>> y1 = f1.encode(x)
    >>> y1
    array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10], dtype=uint8)
    >>> z1 = f1.decode(y1)
    >>> z1
    array([ 1000. ,  1000.1,  1000.2,  1000.3,  1000.4,  1000.6,  1000.7,
            1000.8,  1000.9,  1001. ])
    >>> f2 = zarr.FixedScaleOffsetFilter(offset=1000, scale=10**2, dtype='f8',
    ...                                  astype='u1')
    >>> y2 = f2.encode(x)
    >>> y2
    array([  0,  11,  22,  33,  44,  56,  67,  78,  89, 100], dtype=uint8)
    >>> z2 = f2.decode(y2)
    >>> z2
    array([ 1000.  ,  1000.11,  1000.22,  1000.33,  1000.44,  1000.56,
            1000.67,  1000.78,  1000.89,  1001.  ])
    >>> f3 = zarr.FixedScaleOffsetFilter(offset=1000, scale=10**3, dtype='f8',
    ...                                  astype='u2')
    >>> y3 = f3.encode(x)
    >>> y3
    array([   0,  111,  222,  333,  444,  556,  667,  778,  889, 1000], dtype=uint16)
    >>> z3 = f3.decode(y3)
    >>> z3
    array([ 1000.   ,  1000.111,  1000.222,  1000.333,  1000.444,  1000.556,
            1000.667,  1000.778,  1000.889,  1001.   ])

    """  # flake8: noqa

    filter_name = 'fixedscaleoffset'

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

    def decode(self, buf):
        # interpret buffer as 1D array
        enc = _ndarray_from_buffer(buf, self.astype)
        # decode scale offset
        dec = (enc / self.scale) + self.offset
        # convert dtype
        dec = dec.astype(self.dtype, copy=False)
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['astype'] = encode_dtype(self.astype)
        config['dtype'] = encode_dtype(self.dtype)
        config['scale'] = self.scale
        config['offset'] = self.offset
        return config

    @classmethod
    def from_filter_config(cls, config):
        astype = decode_dtype(config['astype'])
        dtype = decode_dtype(config['dtype'])
        scale = config['scale']
        offset = config['offset']
        return cls(astype=astype, dtype=dtype, scale=scale,
                   offset=offset)


filter_registry[FixedScaleOffsetFilter.filter_name] = FixedScaleOffsetFilter


class QuantizeFilter(object):
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
    >>> f1 = zarr.QuantizeFilter(digits=1, dtype='f8')
    >>> y1 = f1.encode(x)
    >>> y1
    array([ 0.    ,  0.125 ,  0.25  ,  0.3125,  0.4375,  0.5625,  0.6875,
            0.75  ,  0.875 ,  1.    ])
    >>> f2 = zarr.QuantizeFilter(digits=2, dtype='f8')
    >>> y2 = f2.encode(x)
    >>> y2
    array([ 0.       ,  0.109375 ,  0.21875  ,  0.3359375,  0.4453125,
            0.5546875,  0.6640625,  0.78125  ,  0.890625 ,  1.       ])
    >>> f3 = zarr.QuantizeFilter(digits=3, dtype='f8')
    >>> y3 = f3.encode(x)
    >>> y3
    array([ 0.        ,  0.11132812,  0.22265625,  0.33300781,  0.44433594,
            0.55566406,  0.66699219,  0.77734375,  0.88867188,  1.        ])

    """

    filter_name = 'quantize'

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

    def decode(self, buf):
        # filter is lossy, decoding is no-op
        enc = _ndarray_from_buffer(buf, self.astype)
        return enc.astype(self.dtype, copy=False)

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['digits'] = self.digits
        config['dtype'] = encode_dtype(self.dtype)
        config['astype'] = encode_dtype(self.astype)
        return config

    @classmethod
    def from_filter_config(cls, config):
        dtype = decode_dtype(config['dtype'])
        astype = decode_dtype(config['astype'])
        digits = config['digits']
        return cls(digits=digits, dtype=dtype, astype=astype)


filter_registry[QuantizeFilter.filter_name] = QuantizeFilter


# noinspection PyMethodMayBeStatic
class PackBitsFilter(object):
    """Filter to pack elements of a boolean array into bits in a uint8 array.

    Examples
    --------
    >>> import zarr
    >>> import numpy as np
    >>> f = zarr.PackBitsFilter()
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

    filter_name = 'packbits'

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
        # remember how many bits were padded
        if n_bits_leftover:
            n_bits_padded = 8 - n_bits_leftover
        else:
            n_bits_padded = 0
        enc[0] = n_bits_padded
        # apply encoding
        enc[1:] = np.packbits(arr)
        return enc

    def decode(self, buf):
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
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        return config

    @classmethod
    def from_filter_config(cls, config):
        return cls()


filter_registry[PackBitsFilter.filter_name] = PackBitsFilter


def _ensure_bytes(l):
    if isinstance(l, binary_type):
        return l
    elif isinstance(l, text_type):
        return l.encode('ascii')
    else:
        raise ValueError('expected bytes, found %r' % l)


class CategoryFilter(object):
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
    >>> f = zarr.CategoryFilter(labels=[b'female', b'male'], dtype=x.dtype)
    >>> y = f.encode(x)
    >>> y
    array([2, 1, 1, 2, 0], dtype=uint8)
    >>> z = f.decode(y)
    >>> z
    array([b'male', b'female', b'female', b'male', b''],
          dtype='|S10')

    """

    filter_name = 'category'

    def __init__(self, labels, dtype, astype='u1'):
        self.labels = [_ensure_bytes(l) for l in labels]
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != 'S':
            raise ValueError('only string data types are supported')
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

    def decode(self, buf):
        # view encoded data as ndarray
        enc = _ndarray_from_buffer(buf, self.astype)
        # setup output
        dec = np.zeros_like(enc, dtype=self.dtype)
        # apply decoding
        for i, l in enumerate(self.labels):
            dec[enc == (i + 1)] = l
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['labels'] = [text_type(l, 'ascii') for l in self.labels]
        config['dtype'] = encode_dtype(self.dtype)
        config['astype'] = encode_dtype(self.astype)
        return config

    @classmethod
    def from_filter_config(cls, config):
        dtype = decode_dtype(config['dtype'])
        astype = decode_dtype(config['astype'])
        labels = config['labels']
        return cls(labels=labels, dtype=dtype, astype=astype)


filter_registry[CategoryFilter.filter_name] = CategoryFilter


# add in compressors as filters
for cls in compressor_registry.values():
    if hasattr(cls, 'filter_name'):
        filter_registry[cls.filter_name] = cls


def get_filters(configs):
    if not configs:
        return None
    else:
        filters = list()
        for config in configs:
            name = config['name']
            cls = filter_registry[name]
            f = cls.from_filter_config(config)
            filters.append(f)
            # TODO error handling
        return filters
