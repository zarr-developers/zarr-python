from __future__ import annotations
from zarr.v3.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.v3.common import BytesLike, RuntimeConfiguration


import numpy as np


from typing import List, Optional


async def decode(
    codecs: List[Codec], chunk_bytes: BytesLike, runtime_configuration: RuntimeConfiguration
) -> np.ndarray:
    # todo: increase the arity of the function signature with
    # positions for array_array, array_bytes, and bytes_bytes
    _array_array_codecs = [codec for codec in codecs if isinstance(codec, ArrayArrayCodec)]
    _array_bytes_codec = next(codec for codec in codecs if isinstance(codec, ArrayBytesCodec))
    _bytes_bytes_codecs = [codec for codec in codecs if isinstance(codec, BytesBytesCodec)]

    for bb_codec in _bytes_bytes_codecs[::-1]:
        chunk_bytes = await bb_codec.decode(
            chunk_bytes, runtime_configuration=runtime_configuration
        )

    chunk_array = await _array_bytes_codec.decode(
        chunk_bytes, runtime_configuration=runtime_configuration
    )

    for aa_codec in _array_array_codecs[::-1]:
        chunk_array = await aa_codec.decode(
            chunk_array, runtime_configuration=runtime_configuration
        )

    return chunk_array


async def encode(
    codecs: List[Codec], chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
) -> Optional[BytesLike]:
    # todo: increase the arity of the function signature
    # with positions for array_array, array_bytes, and bytes_bytes
    _array_array_codecs = [codec for codec in codecs if isinstance(codec, ArrayArrayCodec)]
    _array_bytes_codec = next(codec for codec in codecs if isinstance(codec, ArrayBytesCodec))
    _bytes_bytes_codecs = [codec for codec in codecs if isinstance(codec, BytesBytesCodec)]

    for aa_codec in _array_array_codecs:
        chunk_array_maybe = await aa_codec.encode(
            chunk_array, runtime_configuration=runtime_configuration
        )
        if chunk_array_maybe is None:
            return None
        chunk_array = chunk_array_maybe

    chunk_bytes_maybe = await _array_bytes_codec.encode(
        chunk_array, runtime_configuration=runtime_configuration
    )
    if chunk_bytes_maybe is None:
        return None
    chunk_bytes = chunk_bytes_maybe

    for bb_codec in _bytes_bytes_codecs:
        chunk_bytes_maybe = await bb_codec.encode(
            chunk_bytes, runtime_configuration=runtime_configuration
        )
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

    return chunk_bytes
