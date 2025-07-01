from zarr.abc.codec import Codec
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
    codecs = convert_compressor(metadata_v2)

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


def convert_compressor(metadata_v2: ArrayV2Metadata) -> list[Codec]:
    compressor_codecs: list[Codec] = []

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
            numcodec_name = f"numcodecs.{compressor_name}"
            numcodec_dict = {
                "name": numcodec_name,
                "configuration": metadata_v2.compressor.get_config(),
            }

            try:
                compressor_codec = get_codec_class(numcodec_name)
            except KeyError as exc:
                raise ValueError(
                    f"Couldn't find corresponding numcodecs.zarr3 codec for {compressor_name}"
                ) from exc

            compressor_codecs.append(compressor_codec.from_dict(numcodec_dict))

    return compressor_codecs
