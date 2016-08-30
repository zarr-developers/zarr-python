# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import math


import numpy as np


from zarr.meta import encode_dtype, decode_dtype
from zarr.compressors import registry as compressor_registry


filter_registry = dict()


class DeltaFilter(object):

    filter_name = 'delta'

    def __init__(self, enc_dtype, dec_dtype):
        self.enc_dtype = np.dtype(enc_dtype)
        self.dec_dtype = np.dtype(dec_dtype)

    def encode(self, buf):
        # interpret buffer as array
        arr = np.frombuffer(buf, dtype=self.dec_dtype)
        # setup encoded output
        enc = np.empty_like(arr, dtype=self.enc_dtype)
        # set first element
        enc[0] = arr[0]
        # compute differences
        enc[1:] = np.diff(arr)
        return enc

    def decode(self, buf):
        # interpret buffer as array
        enc = np.frombuffer(buf, dtype=self.enc_dtype)
        # setup decoded output
        dec = np.empty_like(enc, dtype=self.dec_dtype)
        # decode differences
        np.cumsum(enc, out=dec)
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['enc_dtype'] = encode_dtype(self.enc_dtype)
        config['dec_dtype'] = encode_dtype(self.dec_dtype)
        return config

    @classmethod
    def from_filter_config(cls, config):
        enc_dtype = decode_dtype(config['enc_dtype'])
        dec_dtype = decode_dtype(config['dec_dtype'])
        return cls(enc_dtype=enc_dtype, dec_dtype=dec_dtype)


filter_registry[DeltaFilter.filter_name] = DeltaFilter


class ScaleOffsetFilter(object):

    filter_name = 'scaleoffset'

    def __init__(self, offset, scale, enc_dtype, dec_dtype):
        self.offset = offset
        self.scale = scale
        self.enc_dtype = np.dtype(enc_dtype)
        self.dec_dtype = np.dtype(dec_dtype)

    def encode(self, buf):
        # interpret buffer as array
        arr = np.frombuffer(buf, dtype=self.dec_dtype)
        # compute scale offset
        enc = ((arr - self.offset) / self.scale).astype(self.enc_dtype)
        return enc

    def decode(self, buf):
        # interpret buffer as array
        enc = np.frombuffer(buf, dtype=self.enc_dtype)
        # decode scale offset
        dec = ((enc * self.scale) + self.offset).astype(self.dec_dtype)
        return dec

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['enc_dtype'] = encode_dtype(self.enc_dtype)
        config['dec_dtype'] = encode_dtype(self.dec_dtype)
        config['scale'] = self.scale
        config['offset'] = self.offset
        return config

    @classmethod
    def from_filter_config(cls, config):
        enc_dtype = decode_dtype(config['enc_dtype'])
        dec_dtype = decode_dtype(config['dec_dtype'])
        scale = config['scale']
        offset = config['offset']
        return cls(enc_dtype=enc_dtype, dec_dtype=dec_dtype, scale=scale,
                   offset=offset)


filter_registry[ScaleOffsetFilter.filter_name] = ScaleOffsetFilter


class QuantizeFilter(object):

    filter_name = 'quantize'

    def __init__(self, digits, dtype):
        self.digits = digits
        self.dtype = np.dtype(dtype)

    def encode(self, buf):
        # interpret buffer as array
        arr = np.frombuffer(buf, dtype=self.dtype)
        # apply encoding
        precision = 10. ** -self.digits
        exp = math.log(precision, 10)
        if exp < 0:
            exp = int(math.floor(exp))
        else:
            exp = int(math.ceil(exp))
        bits = math.ceil(math.log(10. ** -exp, 2))
        scale = 2. ** bits
        enc = np.around(scale * arr) / scale
        return enc

    def decode(self, buf):
        # filter is lossy, decoding is no-op
        return buf

    def get_filter_config(self):
        config = dict()
        config['name'] = self.filter_name
        config['digits'] = self.digits
        config['dtype'] = encode_dtype(self.dtype)
        return config

    @classmethod
    def from_filter_config(cls, config):
        dtype = decode_dtype(config['dtype'])
        digits = config['digits']
        return cls(digits=digits, dtype=dtype)


filter_registry[QuantizeFilter.filter_name] = QuantizeFilter


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
