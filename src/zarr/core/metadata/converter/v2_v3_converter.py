from typing import cast

import numcodecs.abc

from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec, Codec
from zarr.codecs.blosc import BloscCodec, BloscShuffle
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array import Array
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.dtype.common import HasEndianness
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.registry import get_codec_class


async def convert_v2_to_v3(zarr_v2: Array) -> None:
    if not isinstance(zarr_v2.metadata, ArrayV2Metadata):
        raise TypeError("Only arrays / groups with zarr v2 metadata can be converted")

    # zarr_format = zarr_v2.metadata.zarr_format
    # if zarr_format != 2:
    #     raise ValueError(
    #         f"Input zarr array / group is zarr_format {zarr_format} - only 2 is accepted."
    #     )

    # accept array or group - if group, iterate into it to do the whole hierarchy

    # how are the metadata files currently written? Which function?
    # could add a to_v3() function on to the ArrayV2Metadata / GroupMetadata classes??
    convert_v2_metadata(zarr_v2.metadata)

    # Check for existing zarr json?
    # await zarr_v2._async_array._save_metadata(metadata_v3)


def convert_v2_metadata(metadata_v2: ArrayV2Metadata) -> ArrayV3Metadata:
    chunk_key_encoding = DefaultChunkKeyEncoding(separator=metadata_v2.dimension_separator)

    codecs: list[Codec] = []

    # array-array codecs
    if metadata_v2.order == "F":
        # F is equivalent to order: n-1, ... 1, 0
        codecs.append(TransposeCodec(order=list(range(len(metadata_v2.shape) - 1, -1, -1))))
    codecs.extend(convert_filters(metadata_v2))

    # array-bytes codecs
    if not isinstance(metadata_v2.dtype, HasEndianness):
        codecs.append(BytesCodec(endian=None))
    else:
        codecs.append(BytesCodec(endian=metadata_v2.dtype.endianness))

    # bytes-bytes codecs
    bytes_bytes_codec = convert_compressor(metadata_v2)
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


def convert_filters(metadata_v2: ArrayV2Metadata) -> list[ArrayArrayCodec]:
    if metadata_v2.filters is None:
        return []

    filters_codecs = [find_numcodecs_zarr3(filter) for filter in metadata_v2.filters]
    for codec in filters_codecs:
        if not isinstance(codec, ArrayArrayCodec):
            raise TypeError(f"Filter {type(codec)} is not an ArrayArrayCodec")

    return cast(list[ArrayArrayCodec], filters_codecs)


def convert_compressor(metadata_v2: ArrayV2Metadata) -> BytesBytesCodec | None:
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
            compressor_codec = find_numcodecs_zarr3(metadata_v2.compressor)

            if not isinstance(compressor_codec, BytesBytesCodec):
                raise TypeError(f"Compressor {type(compressor_codec)} is not a BytesBytesCodec")

            return compressor_codec


def find_numcodecs_zarr3(numcodecs_codec: numcodecs.abc.Codec) -> Codec:
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
