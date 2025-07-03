import asyncio
from typing import Any, cast

import numcodecs.abc

import zarr
from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec, Codec
from zarr.codecs.blosc import BloscCodec, BloscShuffle
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array import Array
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.common import ZARR_JSON
from zarr.core.dtype.common import HasEndianness
from zarr.core.group import Group, GroupMetadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.registry import get_codec_class
from zarr.storage import StoreLike
from zarr.storage._utils import _join_paths


def convert_v2_to_v3(
    store: StoreLike, path: str | None = None, storage_options: dict[str, Any] | None = None
) -> None:
    """Convert all v2 metadata in a zarr hierarchy to v3. This will create a zarr.json file at each level
    (for every group / array). V2 files (.zarray, .zattrs etc.) will be left as-is.

    Parameters
    ----------
    store : StoreLike
        Store or path to directory in file system or name of zip file.
    path : str | None, optional
        The path within the store to open, by default None
    storage_options : dict | None, optional
        If the store is backed by an fsspec-based implementation, then this dict will be passed to
        the Store constructor for that implementation. Ignored otherwise.
    """

    zarr_v2 = zarr.open(
        store=store, mode="r+", zarr_format=2, path=path, storage_options=storage_options
    )
    convert_array_or_group(zarr_v2)


def convert_array_or_group(zarr_v2: Array | Group) -> None:
    """Convert all v2 metadata in a zarr array/group to v3. Note - if a group is provided, then
    all arrays / groups within this group will also be converted. A zarr.json file will be created
    at each level, with any V2 files (.zarray, .zattrs etc.) left as-is.

    Parameters
    ----------
    zarr_v2 : Array | Group
        An array or group with zarr_format = 2
    """
    if not zarr_v2.metadata.zarr_format == 2:
        raise TypeError("Only arrays / groups with zarr v2 metadata can be converted")

    if isinstance(zarr_v2.metadata, GroupMetadata):
        group_metadata_v3 = GroupMetadata(
            attributes=zarr_v2.metadata.attributes, zarr_format=3, consolidated_metadata=None
        )
        sync(_save_v3_metadata(zarr_v2, group_metadata_v3))

        # process members of the group
        for key in zarr_v2:
            convert_array_or_group(zarr_v2[key])

    else:
        array_metadata_v3 = _convert_array_metadata(zarr_v2.metadata)
        sync(_save_v3_metadata(zarr_v2, array_metadata_v3))


def _convert_array_metadata(metadata_v2: ArrayV2Metadata) -> ArrayV3Metadata:
    chunk_key_encoding = DefaultChunkKeyEncoding(separator=metadata_v2.dimension_separator)

    codecs: list[Codec] = []

    # array-array codecs
    if metadata_v2.order == "F":
        # F is equivalent to order: n-1, ... 1, 0
        codecs.append(TransposeCodec(order=list(range(len(metadata_v2.shape) - 1, -1, -1))))
    codecs.extend(_convert_filters(metadata_v2))

    # array-bytes codecs
    if not isinstance(metadata_v2.dtype, HasEndianness):
        codecs.append(BytesCodec(endian=None))
    else:
        codecs.append(BytesCodec(endian=metadata_v2.dtype.endianness))

    # bytes-bytes codecs
    bytes_bytes_codec = _convert_compressor(metadata_v2)
    if bytes_bytes_codec is not None:
        codecs.append(bytes_bytes_codec)

    return ArrayV3Metadata(
        shape=metadata_v2.shape,
        data_type=metadata_v2.dtype,
        chunk_grid=metadata_v2.chunk_grid,
        chunk_key_encoding=chunk_key_encoding,
        fill_value=metadata_v2.fill_value,
        codecs=codecs,
        attributes=metadata_v2.attributes,
        dimension_names=None,
        storage_transformers=None,
    )


def _convert_filters(metadata_v2: ArrayV2Metadata) -> list[ArrayArrayCodec]:
    if metadata_v2.filters is None:
        return []

    filters_codecs = [_find_numcodecs_zarr3(filter) for filter in metadata_v2.filters]
    for codec in filters_codecs:
        if not isinstance(codec, ArrayArrayCodec):
            raise TypeError(f"Filter {type(codec)} is not an ArrayArrayCodec")

    return cast(list[ArrayArrayCodec], filters_codecs)


def _convert_compressor(metadata_v2: ArrayV2Metadata) -> BytesBytesCodec | None:
    if metadata_v2.compressor is None:
        return None

    compressor_name = metadata_v2.compressor.codec_id

    match compressor_name:
        case "blosc":
            return BloscCodec(
                typesize=metadata_v2.dtype.to_native_dtype().itemsize,
                cname=metadata_v2.compressor.cname,
                clevel=metadata_v2.compressor.clevel,
                shuffle=BloscShuffle.from_int(metadata_v2.compressor.shuffle),
                blocksize=metadata_v2.compressor.blocksize,
            )

        case "zstd":
            return ZstdCodec(
                level=metadata_v2.compressor.level,
                checksum=metadata_v2.compressor.checksum,
            )

        case "gzip":
            return GzipCodec(level=metadata_v2.compressor.level)

        case _:
            # If possible, find matching numcodecs.zarr3 codec
            compressor_codec = _find_numcodecs_zarr3(metadata_v2.compressor)

            if not isinstance(compressor_codec, BytesBytesCodec):
                raise TypeError(f"Compressor {type(compressor_codec)} is not a BytesBytesCodec")

            return compressor_codec


def _find_numcodecs_zarr3(numcodecs_codec: numcodecs.abc.Codec) -> Codec:
    """Find matching numcodecs.zarr3 codec (if it exists)"""

    numcodec_name = f"numcodecs.{numcodecs_codec.codec_id}"
    numcodec_dict = {
        "name": numcodec_name,
        "configuration": numcodecs_codec.get_config(),
    }

    try:
        codec_v3 = get_codec_class(numcodec_name)
    except KeyError as exc:
        raise ValueError(
            f"Couldn't find corresponding numcodecs.zarr3 codec for {numcodecs_codec.codec_id}"
        ) from exc

    return codec_v3.from_dict(numcodec_dict)


async def _save_v3_metadata(
    zarr_v2: Array | Group, metadata_v3: ArrayV3Metadata | GroupMetadata
) -> None:
    zarr_json_path = _join_paths([zarr_v2.path, ZARR_JSON])

    if await zarr_v2.store.exists(zarr_json_path):
        raise ValueError(f"{ZARR_JSON} already exists at {zarr_v2.store_path}")

    to_save = metadata_v3.to_buffer_dict(default_buffer_prototype())
    awaitables = [
        zarr_v2.store.set_if_not_exists(_join_paths([zarr_v2.path, key]), value)
        for key, value in to_save.items()
    ]

    await asyncio.gather(*awaitables)
