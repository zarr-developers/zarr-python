# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json


import numpy as np


from zarr.compat import PY2, text_type, binary_type
from zarr.errors import MetadataError


ZARR_FORMAT = 2


def decode_array_metadata(s):
    if isinstance(s, binary_type):
        s = text_type(s, 'ascii')
    meta = json.loads(s)
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)
    try:
        dtype = decode_dtype(meta['dtype'])
        fill_value = decode_fill_value(meta['fill_value'], dtype)
        meta = dict(
            zarr_format=meta['zarr_format'],
            shape=tuple(meta['shape']),
            chunks=tuple(meta['chunks']),
            dtype=dtype,
            compressor=meta['compressor'],
            fill_value=fill_value,
            order=meta['order'],
            filters=meta['filters'],
        )
    except Exception as e:
        raise MetadataError('error decoding metadata: %s' % e)
    else:
        return meta


def encode_array_metadata(meta):
    meta = dict(
        zarr_format=ZARR_FORMAT,
        shape=meta['shape'],
        chunks=meta['chunks'],
        dtype=encode_dtype(meta['dtype']),
        compressor=meta['compressor'],
        fill_value=encode_fill_value(meta['fill_value']),
        order=meta['order'],
        filters=meta['filters'],
    )
    s = json.dumps(meta, indent=4, sort_keys=True, ensure_ascii=True)
    b = s.encode('ascii')
    return b


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return d.descr


def _decode_dtype_descr(d):
    # need to convert list of lists to list of tuples
    if isinstance(d, list):
        # recurse to handle nested structures
        if PY2:  # pragma: no cover
            # under PY2 numpy rejects unicode field names
            d = [(f.encode('ascii'), _decode_dtype_descr(v))
                 for f, v in d]
        else:
            d = [(f, _decode_dtype_descr(v)) for f, v in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)


def decode_group_metadata(s):
    if isinstance(s, binary_type):
        s = text_type(s, 'ascii')
    meta = json.loads(s)
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)
    meta = dict(
        zarr_format=ZARR_FORMAT,
    )
    return meta


def encode_group_metadata(meta=None):
    meta = dict(
        zarr_format=ZARR_FORMAT,
    )
    s = json.dumps(meta, indent=4, sort_keys=True, ensure_ascii=True)
    b = s.encode('ascii')
    return b


FLOAT_FILLS = {
    'NaN': np.nan,
    'Infinity': np.PINF,
    '-Infinity': np.NINF
}


def decode_fill_value(v, dtype):
    if dtype.kind == 'f':
        if v == 'NaN':
            return np.nan
        elif v == 'Infinity':
            return np.PINF
        elif v == '-Infinity':
            return np.NINF
        else:
            return v
    else:
        return v


def encode_fill_value(v):
    try:
        if np.isnan(v):
            return 'NaN'
        elif np.isposinf(v):
            return 'Infinity'
        elif np.isneginf(v):
            return '-Infinity'
        else:
            return v
    except TypeError:
        return v
