import base64
import itertools
import os
from collections.abc import Mapping

import numpy as np

from zarr.errors import MetadataError
from zarr.util import json_dumps, json_loads

from typing import cast, Union, Any, List, Mapping as MappingType

ZARR_FORMAT = 2
ZARR_FORMAT_v3 = 3

FLOAT_FILLS = {"NaN": np.nan, "Infinity": np.PINF, "-Infinity": np.NINF}


_v3_core_type = set(
    "".join(d)
    for d in itertools.product("<>", ("u", "i", "f"), ("2", "4", "8"))
)
_v3_core_type = {"bool", "i1", "u1"} | _v3_core_type

ZARR_V3_CORE_DTYPES_ONLY = int(os.environ.get("ZARR_V3_CORE_DTYPES_ONLY", False))
ZARR_V3_ALLOW_COMPLEX = int(os.environ.get("ZARR_V3_ALLOW_COMPLEX",
                                           not ZARR_V3_CORE_DTYPES_ONLY))
ZARR_V3_ALLOW_DATETIME = int(os.environ.get("ZARR_V3_ALLOW_DATETIME",
                                            not ZARR_V3_CORE_DTYPES_ONLY))
ZARR_V3_ALLOW_STRUCTURED = int(os.environ.get("ZARR_V3_ALLOW_STRUCTURED",
                                              not ZARR_V3_CORE_DTYPES_ONLY))
ZARR_V3_ALLOW_OBJECTARRAY = int(os.environ.get("ZARR_V3_ALLOW_OBJECTARRAY",
                                               not ZARR_V3_CORE_DTYPES_ONLY))
ZARR_V3_ALLOW_BYTES_ARRAY = int(os.environ.get("ZARR_V3_ALLOW_BYTES_ARRAY",
                                               not ZARR_V3_CORE_DTYPES_ONLY))
ZARR_V3_ALLOW_UNICODE_ARRAY = int(os.environ.get("ZARR_V3_ALLOW_UNICODE_ARRAY",
                                                 not ZARR_V3_CORE_DTYPES_ONLY))

_default_entry_point_metadata_v3 = {
    'zarr_format': "https://purl.org/zarr/spec/protocol/core/3.0",
    'metadata_encoding': "https://purl.org/zarr/spec/protocol/core/3.0",
    'metadata_key_suffix': '.json',
    "extensions": [],
}


class Metadata2:
    ZARR_FORMAT = ZARR_FORMAT

    @classmethod
    def parse_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:

        # Here we allow that a store may return an already-parsed metadata object,
        # or a string of JSON that we will parse here. We allow for an already-parsed
        # object to accommodate a consolidated metadata store, where all the metadata for
        # all groups and arrays will already have been parsed from JSON.

        if isinstance(s, Mapping):
            # assume metadata has already been parsed into a mapping object
            meta = s

        else:
            # assume metadata needs to be parsed as JSON
            meta = json_loads(s)

        return meta

    @classmethod
    def decode_array_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)

        # check metadata format
        zarr_format = meta.get("zarr_format", None)
        if zarr_format != cls.ZARR_FORMAT:
            raise MetadataError("unsupported zarr format: %s" % zarr_format)

        # extract array metadata fields
        try:
            dtype = cls.decode_dtype(meta["dtype"])
            if dtype.hasobject:
                import numcodecs
                object_codec = numcodecs.get_codec(meta['filters'][0])
            else:
                object_codec = None

            dimension_separator = meta.get("dimension_separator", None)
            fill_value = cls.decode_fill_value(meta['fill_value'], dtype, object_codec)
            meta = dict(
                zarr_format=meta["zarr_format"],
                shape=tuple(meta["shape"]),
                chunks=tuple(meta["chunks"]),
                dtype=dtype,
                compressor=meta["compressor"],
                fill_value=fill_value,
                order=meta["order"],
                filters=meta["filters"],
            )
            if dimension_separator:
                meta['dimension_separator'] = dimension_separator
        except Exception as e:
            raise MetadataError("error decoding metadata") from e
        else:
            return meta

    @classmethod
    def encode_array_metadata(cls, meta: MappingType[str, Any]) -> bytes:
        dtype = meta["dtype"]
        sdshape = ()
        if dtype.subdtype is not None:
            dtype, sdshape = dtype.subdtype

        dimension_separator = meta.get("dimension_separator")
        if dtype.hasobject:
            import numcodecs
            object_codec = numcodecs.get_codec(meta['filters'][0])
        else:
            object_codec = None

        meta = dict(
            zarr_format=cls.ZARR_FORMAT,
            shape=meta["shape"] + sdshape,
            chunks=meta["chunks"],
            dtype=cls.encode_dtype(dtype),
            compressor=meta["compressor"],
            fill_value=cls.encode_fill_value(meta["fill_value"], dtype, object_codec),
            order=meta["order"],
            filters=meta["filters"],
        )
        if dimension_separator:
            meta['dimension_separator'] = dimension_separator

        if dimension_separator:
            meta["dimension_separator"] = dimension_separator

        return json_dumps(meta)

    @classmethod
    def encode_dtype(cls, d: np.dtype):
        if d.fields is None:
            return d.str
        else:
            return d.descr

    @classmethod
    def _decode_dtype_descr(cls, d) -> List[Any]:
        # need to convert list of lists to list of tuples
        if isinstance(d, list):
            # recurse to handle nested structures
            d = [(k[0], cls._decode_dtype_descr(k[1])) + tuple(k[2:]) for k in d]
        return d

    @classmethod
    def decode_dtype(cls, d) -> np.dtype:
        d = cls._decode_dtype_descr(d)
        return np.dtype(d)

    @classmethod
    def decode_group_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)

        # check metadata format version
        zarr_format = meta.get("zarr_format", None)
        if zarr_format != cls.ZARR_FORMAT:
            raise MetadataError("unsupported zarr format: %s" % zarr_format)

        meta = dict(zarr_format=zarr_format)
        return meta

    # N.B., keep `meta` parameter as a placeholder for future
    # noinspection PyUnusedLocal
    @classmethod
    def encode_group_metadata(cls, meta=None) -> bytes:
        meta = dict(zarr_format=cls.ZARR_FORMAT)
        return json_dumps(meta)

    @classmethod
    def decode_fill_value(cls, v: Any, dtype: np.dtype, object_codec: Any = None) -> Any:
        # early out
        if v is None:
            return v
        if dtype.kind == 'V' and dtype.hasobject:
            if object_codec is None:
                raise ValueError('missing object_codec for object array')
            v = base64.standard_b64decode(v)
            v = object_codec.decode(v)
            v = np.array(v, dtype=dtype)[()]
            return v
        if dtype.kind == "f":
            if v == "NaN":
                return np.nan
            elif v == "Infinity":
                return np.PINF
            elif v == "-Infinity":
                return np.NINF
            else:
                return np.array(v, dtype=dtype)[()]
        elif dtype.kind in "c":
            v = (
                cls.decode_fill_value(v[0], dtype.type().real.dtype),
                cls.decode_fill_value(v[1], dtype.type().imag.dtype),
            )
            v = v[0] + 1j * v[1]
            return np.array(v, dtype=dtype)[()]
        elif dtype.kind == "S":
            # noinspection PyBroadException
            try:
                v = base64.standard_b64decode(v)
            except Exception:
                # be lenient, allow for other values that may have been used before base64
                # encoding and may work as fill values, e.g., the number 0
                pass
            v = np.array(v, dtype=dtype)[()]
            return v
        elif dtype.kind == "V":
            v = base64.standard_b64decode(v)
            v = np.array(v, dtype=dtype.str).view(dtype)[()]
            return v
        elif dtype.kind == "U":
            # leave as-is
            return v
        else:
            return np.array(v, dtype=dtype)[()]

    @classmethod
    def encode_fill_value(cls, v: Any, dtype: np.dtype, object_codec: Any = None) -> Any:
        # early out
        if v is None:
            return v
        if dtype.kind == 'V' and dtype.hasobject:
            if object_codec is None:
                raise ValueError('missing object_codec for object array')
            v = object_codec.encode(v)
            v = str(base64.standard_b64encode(v), 'ascii')
            return v
        if dtype.kind == "f":
            if np.isnan(v):
                return "NaN"
            elif np.isposinf(v):
                return "Infinity"
            elif np.isneginf(v):
                return "-Infinity"
            else:
                return float(v)
        elif dtype.kind in "ui":
            return int(v)
        elif dtype.kind == "b":
            return bool(v)
        elif dtype.kind in "c":
            c = cast(np.complex128, np.dtype(complex).type())
            v = (cls.encode_fill_value(v.real, c.real.dtype, object_codec),
                 cls.encode_fill_value(v.imag, c.imag.dtype, object_codec))
            return v
        elif dtype.kind in "SV":
            v = str(base64.standard_b64encode(v), "ascii")
            return v
        elif dtype.kind == "U":
            return v
        elif dtype.kind in "mM":
            return int(v.view("i8"))
        else:
            return v


class Metadata3(Metadata2):
    ZARR_FORMAT = ZARR_FORMAT_v3

    @classmethod
    def decode_dtype(cls, d):
        d = cls._decode_dtype_descr(d)
        dtype = np.dtype(d)
        if dtype.kind == 'c':
            if not ZARR_V3_ALLOW_COMPLEX:
                raise ValueError("complex-valued arrays not supported")
        elif dtype.kind in 'mM':
            if not ZARR_V3_ALLOW_DATETIME:
                raise ValueError(
                    "datetime64 and timedelta64 arrays not supported"
                )
        elif dtype.kind == 'O':
            if not ZARR_V3_ALLOW_OBJECTARRAY:
                raise ValueError("object arrays not supported")
        elif dtype.kind == 'V':
            if not ZARR_V3_ALLOW_STRUCTURED:
                raise ValueError("structured arrays not supported")
        elif dtype.kind == 'U':
            if not ZARR_V3_ALLOW_UNICODE_ARRAY:
                raise ValueError("unicode arrays not supported")
        elif dtype.kind == 'S':
            if not ZARR_V3_ALLOW_BYTES_ARRAY:
                raise ValueError("bytes arrays not supported")
        else:
            assert d in _v3_core_type
        return dtype

    @classmethod
    def encode_dtype(cls, d):
        s = Metadata2.encode_dtype(d)
        if s == "|b1":
            return "bool"
        elif s == "|u1":
            return "u1"
        elif s == "|i1":
            return "i1"
        dtype = np.dtype(d)
        if dtype.kind == "c":
            if not ZARR_V3_ALLOW_COMPLEX:
                raise ValueError(
                    "complex-valued arrays not part of the base v3 spec"
                )
        elif dtype.kind in "mM":
            if not ZARR_V3_ALLOW_DATETIME:
                raise ValueError(
                    "datetime64 and timedelta64 not part of the base v3 "
                    "spec"
                )
        elif dtype.kind == "O":
            if not ZARR_V3_ALLOW_OBJECTARRAY:
                raise ValueError(
                    "object dtypes are not part of the base v3 spec"
                )
        elif dtype.kind == "V":
            if not ZARR_V3_ALLOW_STRUCTURED:
                raise ValueError(
                    "structured arrays are not part of the base v3 spec"
                )
        elif dtype.kind == 'U':
            if not ZARR_V3_ALLOW_UNICODE_ARRAY:
                raise ValueError("unicode dtypes are not part of the base v3 "
                                 "spec")
        elif dtype.kind == 'S':
            if not ZARR_V3_ALLOW_BYTES_ARRAY:
                raise ValueError("bytes dtypes are not part of the base v3 "
                                 "spec")
        else:
            assert s in _v3_core_type
        return s

    @classmethod
    def decode_group_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)
        # 1 / 0
        # # check metadata format version
        # zarr_format = meta.get("zarr_format", None)
        # if zarr_format != cls.ZARR_FORMAT:
        #     raise MetadataError("unsupported zarr format: %s" % zarr_format)

        assert 'attributes' in meta
        # meta = dict(attributes=meta['attributes'])
        return meta

        # return json.loads(s)

    @classmethod
    def encode_group_metadata(cls, meta=None) -> bytes:
        # The ZARR_FORMAT should not be in the group metadata, but in the
        # entry point metadata instead
        # meta = dict(zarr_format=cls.ZARR_FORMAT)
        if meta is None:
            meta = {'attributes': {}}
        meta = dict(attributes=meta.get("attributes", {}))
        return json_dumps(meta)

    @classmethod
    def encode_hierarchy_metadata(cls, meta=None) -> bytes:
        if meta is None:
            meta = _default_entry_point_metadata_v3
        elif set(meta.keys()) != {
                "zarr_format",
                "metadata_encoding",
                "metadata_key_suffix",
                "extensions",
        }:
            raise ValueError(f"Unexpected keys in metadata. meta={meta}")
        return json_dumps(meta)

    @classmethod
    def decode_hierarchy_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)
        # check metadata format
        # zarr_format = meta.get("zarr_format", None)
        # if zarr_format != "https://purl.org/zarr/spec/protocol/core/3.0":
        #     raise MetadataError("unsupported zarr format: %s" % zarr_format)
        if set(meta.keys()) != {
                "zarr_format",
                "metadata_encoding",
                "metadata_key_suffix",
                "extensions",
        }:
            raise ValueError(f"Unexpected keys in metdata. meta={meta}")
        return meta

    @classmethod
    def decode_array_metadata(cls, s: Union[MappingType, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)

        # check metadata format
        zarr_format = meta.get("zarr_format", None)
        if zarr_format != cls.ZARR_FORMAT:
            raise MetadataError("unsupported zarr format: %s" % zarr_format)

        # extract array metadata fields
        try:
            dtype = cls.decode_dtype(meta["data_type"])
            if dtype.hasobject:
                import numcodecs
                object_codec = numcodecs.get_codec(meta['attributes']['filters'][0])
            else:
                object_codec = None
            fill_value = cls.decode_fill_value(meta["fill_value"], dtype, object_codec)
            # TODO: remove dimension_separator?
            meta = dict(
                zarr_format=meta["zarr_format"],
                shape=tuple(meta["shape"]),
                chunk_grid=dict(
                    type=meta["chunk_grid"]["type"],
                    chunk_shape=tuple(meta["chunk_grid"]["chunk_shape"]),
                    separator=meta["chunk_grid"]["separator"],
                ),
                data_type=dtype,
                compressor=meta["compressor"],
                fill_value=fill_value,
                chunk_memory_layout=meta["chunk_memory_layout"],
                dimension_separator=meta.get("dimension_separator", "/"),
                attributes=meta["attributes"],
            )
            # dimension_separator = meta.get("dimension_separator", None)
            # if dimension_separator:
            #     meta["dimension_separator"] = dimension_separator
        except Exception as e:
            raise MetadataError("error decoding metadata: %s" % e)
        else:
            return meta

    @classmethod
    def encode_array_metadata(cls, meta: MappingType[str, Any]) -> bytes:
        dtype = meta["data_type"]
        sdshape = ()
        if dtype.subdtype is not None:
            dtype, sdshape = dtype.subdtype
        dimension_separator = meta.get("dimension_separator")
        if dtype.hasobject:
            import numcodecs
            object_codec = numcodecs.get_codec(meta['attributes']['filters'][0])
        else:
            object_codec = None
        meta = dict(
            zarr_format=cls.ZARR_FORMAT,
            shape=meta["shape"] + sdshape,
            chunk_grid=dict(
                type=meta["chunk_grid"]["type"],
                chunk_shape=tuple(meta["chunk_grid"]["chunk_shape"]),
                separator=meta["chunk_grid"]["separator"],
            ),
            data_type=cls.encode_dtype(dtype),
            compressor=meta["compressor"],
            fill_value=encode_fill_value(meta["fill_value"], dtype, object_codec),
            chunk_memory_layout=meta["chunk_memory_layout"],
            attributes=meta.get("attributes", {}),
        )
        if dimension_separator:
            meta["dimension_separator"] = dimension_separator
        return json_dumps(meta)


parse_metadata = Metadata2.parse_metadata
decode_array_metadata = Metadata2.decode_array_metadata
encode_array_metadata = Metadata2.encode_array_metadata
encode_dtype = Metadata2.encode_dtype
_decode_dtype_descr = Metadata2._decode_dtype_descr
decode_dtype = Metadata2.decode_dtype
decode_group_metadata = Metadata2.decode_group_metadata
encode_group_metadata = Metadata2.encode_group_metadata
decode_fill_value = Metadata2.decode_fill_value
encode_fill_value = Metadata2.encode_fill_value
