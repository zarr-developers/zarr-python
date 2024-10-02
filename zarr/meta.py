import base64
import itertools
from collections.abc import Mapping

import numcodecs
import numpy as np
from numcodecs.abc import Codec

from zarr.errors import MetadataError
from zarr.util import json_dumps, json_loads

from typing import cast, Union, Any, List, Mapping as MappingType, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from zarr._storage.store import StorageTransformer


ZARR_FORMAT = 2
ZARR_FORMAT_v3 = 3

# FLOAT_FILLS = {"NaN": np.nan, "Infinity": np.PINF, "-Infinity": np.NINF}

_default_entry_point_metadata_v3 = {
    "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0",
    "metadata_encoding": "https://purl.org/zarr/spec/protocol/core/3.0",
    "metadata_key_suffix": ".json",
    "extensions": [],
}

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
            extension="https://zarr-specs.readthedocs.io/en/core-protocol-v3.0-dev/protocol/extensions/complex-dtypes/v1.0.html",  # noqa
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str == "|O":
        return dict(
            extension="TODO: object array protocol URL",  # noqa
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("|S"):
        return dict(
            extension="TODO: bytestring array protocol URL",  # noqa
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("<U") or dtype.str.startswith(">U"):
        return dict(
            extension="TODO: unicode array protocol URL",  # noqa
            type=dtype.str,
            fallback=None,
        )
    elif dtype.str.startswith("|V"):
        return dict(
            extension="TODO: structured array protocol URL",  # noqa
            type=dtype.descr,
            fallback=None,
        )
    elif dtype.str in _v3_datetime_types:
        return dict(
            extension="https://zarr-specs.readthedocs.io/en/latest/extensions/data-types/datetime/v1.0.html",  # noqa
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
            raise MetadataError(f"unsupported zarr format: {zarr_format}")

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
            raise MetadataError(f"unsupported zarr format: {zarr_format}")

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
        elif dtype.kind in "ui":
            return int(v)
        elif dtype.kind == "b":
            return bool(v)
        elif dtype.kind in "c":
            c = cast(np.complex128, np.dtype(complex).type())
            v = (
                cls.encode_fill_value(v.real, c.real.dtype, object_codec),
                cls.encode_fill_value(v.imag, c.imag.dtype, object_codec),
            )
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
    def decode_dtype(cls, d, validate=True):
        if isinstance(d, dict):
            # extract the type from the extension info
            try:
                d = d["type"]
            except KeyError as e:
                raise KeyError("Extended dtype info must provide a key named 'type'.") from e
        d = cls._decode_dtype_descr(d)
        dtype = np.dtype(d)
        if validate:
            if dtype.str in (_v3_core_types | {"|b1", "|u1", "|i1"}):
                # it is a core dtype of the v3 spec
                pass
            else:
                # will raise if this is not a recognized extended dtype
                get_extended_dtype_info(dtype)
        return dtype

    @classmethod
    def encode_dtype(cls, d):
        s = d.str
        if s == "|b1":
            return "bool"
        elif s == "|u1":
            return "u1"
        elif s == "|i1":
            return "i1"
        elif s in _v3_core_types:
            return Metadata2.encode_dtype(d)
        else:
            # Check if this dtype corresponds to a supported extension to
            # the v3 protocol.
            return get_extended_dtype_info(np.dtype(d))

    @classmethod
    def decode_group_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)
        # 1 / 0
        # # check metadata format version
        # zarr_format = meta.get("zarr_format", None)
        # if zarr_format != cls.ZARR_FORMAT:
        #     raise MetadataError(f"unsupported zarr format: {zarr_format}")

        assert "attributes" in meta
        # meta = dict(attributes=meta['attributes'])
        return meta

        # return json.loads(s)

    @classmethod
    def encode_group_metadata(cls, meta=None) -> bytes:
        # The ZARR_FORMAT should not be in the group metadata, but in the
        # entry point metadata instead
        # meta = dict(zarr_format=cls.ZARR_FORMAT)
        if meta is None:
            meta = {"attributes": {}}
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
    def decode_hierarchy_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)
        # check metadata format
        # zarr_format = meta.get("zarr_format", None)
        # if zarr_format != "https://purl.org/zarr/spec/protocol/core/3.0":
        #     raise MetadataError(f"unsupported zarr format: {zarr_format}")
        if set(meta.keys()) != {
            "zarr_format",
            "metadata_encoding",
            "metadata_key_suffix",
            "extensions",
        }:
            raise ValueError(f"Unexpected keys in metadata. meta={meta}")
        return meta

    @classmethod
    def _encode_codec_metadata(cls, codec: Codec) -> Optional[Mapping]:
        if codec is None:
            return None

        # only support gzip for now
        config = codec.get_config()
        del config["id"]
        uri = "https://purl.org/zarr/spec/codec/"
        if isinstance(codec, numcodecs.GZip):
            uri = uri + "gzip/1.0"
        elif isinstance(codec, numcodecs.Zlib):
            uri = uri + "zlib/1.0"
        elif isinstance(codec, numcodecs.Blosc):
            uri = uri + "blosc/1.0"
        elif isinstance(codec, numcodecs.BZ2):
            uri = uri + "bz2/1.0"
        elif isinstance(codec, numcodecs.LZ4):
            uri = uri + "lz4/1.0"
        elif isinstance(codec, numcodecs.LZMA):
            uri = uri + "lzma/1.0"
        elif isinstance(codec, numcodecs.Zstd):
            uri = uri + "zstd/1.0"
        meta = {
            "codec": uri,
            "configuration": config,
        }
        return meta

    @classmethod
    def _decode_codec_metadata(cls, meta: Optional[Mapping]) -> Optional[Codec]:
        if meta is None:
            return None

        uri = "https://purl.org/zarr/spec/codec/"
        conf = meta["configuration"]
        if meta["codec"].startswith(uri + "gzip/"):
            conf["id"] = "gzip"
        elif meta["codec"].startswith(uri + "zlib/"):
            conf["id"] = "zlib"
        elif meta["codec"].startswith(uri + "blosc/"):
            conf["id"] = "blosc"
        elif meta["codec"].startswith(uri + "bz2/"):
            conf["id"] = "bz2"
        elif meta["codec"].startswith(uri + "lz4/"):
            conf["id"] = "lz4"
        elif meta["codec"].startswith(uri + "lzma/"):
            conf["id"] = "lzma"
        elif meta["codec"].startswith(uri + "zstd/"):
            conf["id"] = "zstd"
        else:
            raise NotImplementedError

        codec = numcodecs.get_codec(conf)

        return codec

    @classmethod
    def _encode_storage_transformer_metadata(
        cls, storage_transformer: "StorageTransformer"
    ) -> Optional[Mapping]:
        return {
            "extension": storage_transformer.extension_uri,
            "type": storage_transformer.type,
            "configuration": storage_transformer.get_config(),
        }

    @classmethod
    def _decode_storage_transformer_metadata(cls, meta: Mapping) -> "StorageTransformer":
        from zarr.tests.test_storage_v3 import DummyStorageTransfomer
        from zarr._storage.v3_storage_transformers import ShardingStorageTransformer

        # This might be changed to a proper registry in the future
        KNOWN_STORAGE_TRANSFORMERS = [DummyStorageTransfomer, ShardingStorageTransformer]

        conf = meta.get("configuration", {})
        extension_uri = meta["extension"]
        transformer_type = meta["type"]

        for StorageTransformerCls in KNOWN_STORAGE_TRANSFORMERS:
            if StorageTransformerCls.extension_uri == extension_uri:
                break
        else:  # pragma: no cover
            raise NotImplementedError

        return StorageTransformerCls.from_config(transformer_type, conf)

    @classmethod
    def decode_array_metadata(cls, s: Union[MappingType, bytes, str]) -> MappingType[str, Any]:
        meta = cls.parse_metadata(s)

        # extract array metadata fields
        try:
            dtype = cls.decode_dtype(meta["data_type"])
            if dtype.hasobject:
                import numcodecs

                object_codec = numcodecs.get_codec(meta["attributes"]["filters"][0])
            else:
                object_codec = None
            fill_value = cls.decode_fill_value(meta["fill_value"], dtype, object_codec)
            # TODO: remove dimension_separator?

            compressor = cls._decode_codec_metadata(meta.get("compressor", None))
            storage_transformers = meta.get("storage_transformers", ())
            storage_transformers = [
                cls._decode_storage_transformer_metadata(i) for i in storage_transformers
            ]
            extensions = meta.get("extensions", [])
            meta = dict(
                shape=tuple(meta["shape"]),
                chunk_grid=dict(
                    type=meta["chunk_grid"]["type"],
                    chunk_shape=tuple(meta["chunk_grid"]["chunk_shape"]),
                    separator=meta["chunk_grid"]["separator"],
                ),
                data_type=dtype,
                fill_value=fill_value,
                chunk_memory_layout=meta["chunk_memory_layout"],
                attributes=meta["attributes"],
                extensions=extensions,
            )
            # compressor field should be absent when there is no compression
            if compressor:
                meta["compressor"] = compressor
            if storage_transformers:
                meta["storage_transformers"] = storage_transformers

        except Exception as e:
            raise MetadataError(f"error decoding metadata: {e}") from e
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

            object_codec = numcodecs.get_codec(meta["attributes"]["filters"][0])
        else:
            object_codec = None

        compressor = cls._encode_codec_metadata(meta.get("compressor", None))
        storage_transformers = meta.get("storage_transformers", ())
        storage_transformers = [
            cls._encode_storage_transformer_metadata(i) for i in storage_transformers
        ]
        extensions = meta.get("extensions", [])
        meta = dict(
            shape=meta["shape"] + sdshape,
            chunk_grid=dict(
                type=meta["chunk_grid"]["type"],
                chunk_shape=tuple(meta["chunk_grid"]["chunk_shape"]),
                separator=meta["chunk_grid"]["separator"],
            ),
            data_type=cls.encode_dtype(dtype),
            fill_value=encode_fill_value(meta["fill_value"], dtype, object_codec),
            chunk_memory_layout=meta["chunk_memory_layout"],
            attributes=meta.get("attributes", {}),
            extensions=extensions,
        )
        if compressor:
            meta["compressor"] = compressor
        if dimension_separator:
            meta["dimension_separator"] = dimension_separator
        if storage_transformers:
            meta["storage_transformers"] = storage_transformers
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
