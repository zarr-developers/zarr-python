import base64
import itertools
from collections.abc import Mapping

import numpy as np

from zarr.v2.errors import MetadataError
from zarr.v2.util import json_dumps, json_loads

from typing import cast, Union, Any, List, Mapping as MappingType, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


ZARR_FORMAT = 2

# FLOAT_FILLS = {"NaN": np.nan, "Infinity": np.PINF, "-Infinity": np.NINF}

_v3_core_types = set("".join(d) for d in itertools.product("<>", ("u", "i", "f"), ("2", "4", "8")))
_v3_core_types = {"bool", "i1", "u1"} | _v3_core_types

# The set of complex types allowed ({"<c8", "<c16", ">c8", ">c16"})
_v3_complex_types = set(f"{end}c{_bytes}" for end, _bytes in itertools.product("<>", ("8", "16")))

# All dtype.str values corresponding to datetime64 and timedelta64
# see: https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
_date_units = ["Y", "M", "W", "D"]
_time_units = ["h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]
_v3_datetime_types = set(
    f"{end}{kind}8[{unit}]"
    for end, unit, kind in itertools.product("<>", _date_units + _time_units, ("m", "M"))
)


def get_extended_dtype_info(dtype) -> dict:
    if dtype.str in _v3_complex_types:
        return dict(
            extension="https://zarr-specs.readthedocs.io/en/core-protocol-v3.0-dev/protocol/extensions/complex-dtypes/v1.0.html",
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str == "|O":
        return dict(
            extension="TODO: object array protocol URL",
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("|S"):
        return dict(
            extension="TODO: bytestring array protocol URL",
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("<U") or dtype.str.startswith(">U"):
        return dict(
            extension="TODO: unicode array protocol URL",
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("|V"):
        return dict(
            extension="TODO: structured array protocol URL",
            type=dtype.descr,
            fallback=None,
        )
    elif dtype.str in _v3_datetime_types:
        return dict(
            extension="https://zarr-specs.readthedocs.io/en/latest/extensions/data-types/datetime/v1.0.html",
            type=dtype.str,
            fallback=None,
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class Metadata2:
    ZARR_FORMAT = ZARR_FORMAT

    @classmethod
    def parse_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
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
    def decode_array_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
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

                object_codec = numcodecs.get_codec(meta["filters"][0])
            else:
                object_codec = None

            dimension_separator = meta.get("dimension_separator", None)
            fill_value = cls.decode_fill_value(meta["fill_value"], dtype, object_codec)
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
                meta["dimension_separator"] = dimension_separator
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

            object_codec = numcodecs.get_codec(meta["filters"][0])
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
    def decode_group_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
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
        if dtype.kind == "V" and dtype.hasobject:
            if object_codec is None:
                raise ValueError("missing object_codec for object array")
            v = base64.standard_b64decode(v)
            v = object_codec.decode(v)
            v = np.array(v, dtype=dtype)[()]
            return v
        if dtype.kind == "f":
            if v == "NaN":
                return np.nan
            elif v == "Infinity":
                return np.inf
            elif v == "-Infinity":
                return -np.inf
            else:
                return np.array(v, dtype=dtype)[()]
        elif dtype.kind == "c":
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
        if dtype.kind == "V" and dtype.hasobject:
            if object_codec is None:
                raise ValueError("missing object_codec for object array")
            v = object_codec.encode(v)
            v = str(base64.standard_b64encode(v), "ascii")
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
        elif dtype.kind in ("u", "i"):
            return int(v)
        elif dtype.kind == "b":
            return bool(v)
        elif dtype.kind == "c":
            c = cast(np.complex128, np.dtype(complex).type())
            v = (
                cls.encode_fill_value(v.real, c.real.dtype, object_codec),
                cls.encode_fill_value(v.imag, c.imag.dtype, object_codec),
            )
            return v
        elif dtype.kind in ("S", "V"):
            v = str(base64.standard_b64encode(v), "ascii")
            return v
        elif dtype.kind == "U":
            return v
        elif dtype.kind in ("m", "M"):
            return int(v.view("i8"))
        else:
            return v


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
