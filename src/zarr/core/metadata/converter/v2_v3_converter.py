from typing import cast

import numcodecs.abc

from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec, Codec
from zarr.codecs.blosc import BloscCodec, BloscShuffle
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array import Array
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
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

    # Handle C vs F? (see gist)
    convert_filters(metadata_v2)
    convert_compressor(metadata_v2)

    codecs: list[Codec] = []

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


def convert_compressor(metadata_v2: ArrayV2Metadata) -> list[BytesBytesCodec]:
    compressor_codecs: list[BytesBytesCodec] = []

    if metadata_v2.compressor is None:
        return compressor_codecs

    compressor_name = metadata_v2.compressor.codec_id

    match compressor_name:
        case "blosc":
            compressor_codecs.append(
                BloscCodec(
                    typesize=metadata_v2.dtype.to_native_dtype().itemsize,
                    cname=metadata_v2.compressor.cname,
                    clevel=metadata_v2.compressor.clevel,
                    shuffle=BloscShuffle.from_int(metadata_v2.compressor.shuffle),
                    blocksize=metadata_v2.compressor.blocksize,
                )
            )

        case "zstd":
            compressor_codecs.append(
                ZstdCodec(
                    level=metadata_v2.compressor.level,
                    checksum=metadata_v2.compressor.checksum,
                )
            )

        case "gzip":
            compressor_codecs.append(GzipCodec(level=metadata_v2.compressor.level))

        case _:
            # If possible, find matching numcodecs.zarr3 codec
            compressor_codec = find_numcodecs_zarr3(metadata_v2.compressor)

            if not isinstance(compressor_codec, BytesBytesCodec):
                raise TypeError(f"Compressor {type(compressor_codec)} is not a BytesBytesCodec")

            compressor_codecs.append(compressor_codec)

    return compressor_codecs


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
