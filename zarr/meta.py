# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import base64


import numpy as np
from numcodecs.compat import ensure_bytes


from zarr.compat import PY2, Mapping
from zarr.errors import MetadataError


ZARR_FORMAT = 2


def ensure_str(s):
    if not isinstance(s, str):
        s = ensure_bytes(s)
        if not PY2:  # pragma: py2 no cover
            s = s.decode('ascii')
    return s


def json_dumps(o):
    """Write JSON in a consistent, human-readable way."""
    return json.dumps(o, indent=4, sort_keys=True, ensure_ascii=True,
                      separators=(',', ': '))


def parse_metadata(s):

    # Here we allow that a store may return an already-parsed metadata object,
    # or a string of JSON that we will parse here. We allow for an already-parsed
    # object to accommodate a consolidated metadata store, where all the metadata for
    # all groups and arrays will already have been parsed from JSON.

    if isinstance(s, Mapping):
        # assume metadata has already been parsed into a mapping object
        meta = s

    else:
        # assume metadata needs to be parsed as JSON
        s = ensure_str(s)
        meta = json.loads(s)

    return meta


def decode_array_metadata(s):
    meta = parse_metadata(s)

    # check metadata format
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)

    # extract array metadata fields
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
    dtype = meta['dtype']
    sdshape = ()
    if dtype.subdtype is not None:
        dtype, sdshape = dtype.subdtype
    meta = dict(
        zarr_format=ZARR_FORMAT,
        shape=meta['shape'] + sdshape,
        chunks=meta['chunks'],
        dtype=encode_dtype(dtype),
        compressor=meta['compressor'],
        fill_value=encode_fill_value(meta['fill_value'], dtype),
        order=meta['order'],
        filters=meta['filters'],
    )
    s = json_dumps(meta)
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
        if PY2:  # pragma: py3 no cover
            # under PY2 numpy rejects unicode field names
            d = [(k[0].encode("ascii"), _decode_dtype_descr(k[1])) + tuple(k[2:]) for k in d]
        else:  # pragma: py2 no cover
            d = [(k[0], _decode_dtype_descr(k[1])) + tuple(k[2:]) for k in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)


def decode_group_metadata(s):
    meta = parse_metadata(s)

    # check metadata format version
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)

    meta = dict(zarr_format=zarr_format)
    return meta


# N.B., keep `meta` parameter as a placeholder for future
# noinspection PyUnusedLocal
def encode_group_metadata(meta=None):
    meta = dict(
        zarr_format=ZARR_FORMAT,
    )
    s = json_dumps(meta)
    b = s.encode('ascii')
    return b


FLOAT_FILLS = {
    'NaN': np.nan,
    'Infinity': np.PINF,
    '-Infinity': np.NINF
}


def decode_fill_value(v, dtype):
    # early out
    if v is None:
        return v
    if dtype.kind == 'f':
        if v == 'NaN':
            return np.nan
        elif v == 'Infinity':
            return np.PINF
        elif v == '-Infinity':
            return np.NINF
        else:
            return np.array(v, dtype=dtype)[()]
    elif dtype.kind in 'c':
        v = (decode_fill_value(v[0], dtype.type().real.dtype),
             decode_fill_value(v[1], dtype.type().imag.dtype))
        v = v[0] + 1j * v[1]
        return np.array(v, dtype=dtype)[()]
    elif dtype.kind == 'S':
        # noinspection PyBroadException
        try:
            v = base64.standard_b64decode(v)
        except Exception:
            # be lenient, allow for other values that may have been used before base64
            # encoding and may work as fill values, e.g., the number 0
            pass
        v = np.array(v, dtype=dtype)[()]
        return v
    elif dtype.kind == 'V':
        v = base64.standard_b64decode(v)
        v = np.array(v, dtype=dtype.str).view(dtype)[()]
        return v
    elif dtype.kind == 'U':
        # leave as-is
        return v
    else:
        return np.array(v, dtype=dtype)[()]


def encode_fill_value(v, dtype):
    # early out
    if v is None:
        return v
    if dtype.kind == 'f':
        if np.isnan(v):
            return 'NaN'
        elif np.isposinf(v):
            return 'Infinity'
        elif np.isneginf(v):
            return '-Infinity'
        else:
            return float(v)
    elif dtype.kind in 'ui':
        return int(v)
    elif dtype.kind == 'b':
        return bool(v)
    elif dtype.kind in 'c':
        v = (encode_fill_value(v.real, dtype.type().real.dtype),
             encode_fill_value(v.imag, dtype.type().imag.dtype))
        return v
    elif dtype.kind in 'SV':
        v = base64.standard_b64encode(v)
        if not PY2:  # pragma: py2 no cover
            v = str(v, 'ascii')
        return v
    elif dtype.kind == 'U':
        return v
    elif dtype.kind in 'mM':
        return int(v.view('i8'))
    else:
        return v
