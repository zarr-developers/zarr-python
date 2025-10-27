from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.abc.codec import Codec
from zarr.abc.numcodec import Numcodec
from zarr.codecs._v2 import NumcodecWrapper
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.numcodecs._codecs import _NumcodecsCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import _check_codecjson_v2, _check_codecjson_v3
from zarr.registry import get_codec

if TYPE_CHECKING:
    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import ZDType


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return data


def _parse_codec(data: object, *, dtype: ZDType[Any, Any]) -> Codec | NumcodecWrapper:
    """
    Resolve a potential codec.
    """
    if _check_codecjson_v2(data) or _check_codecjson_v3(data):
        return _parse_codec(get_codec(data), dtype=dtype)

    # This must come before the isinstance(data, Codec) check because _NumcodecsCodec instances
    # are generally subclasses of Codec via multiple inheritance
    if isinstance(data, _NumcodecsCodec):
        # If the input is a NumcodecsCodec, then it's wrapping a numcodecs codec, and in many cases
        # there is a better codec available from the registry.
        return get_codec(data.codec_config)  # type: ignore[arg-type]

    if isinstance(data, (Codec, NumcodecWrapper)):
        # TERRIBLE HACK
        # This is necessary because the Blosc codec defaults create a broken state.
        # We need to provide dtype information here to convert a potentially broken blosc codec to a valid one
        # https://github.com/zarr-developers/zarr-python/issues/3427
        if isinstance(data, BloscCodec):
            return data.evolve_from_array_spec(
                ArraySpec(
                    shape=(1,),
                    dtype=dtype,
                    fill_value=None,
                    config=ArrayConfig.from_dict({}),
                    prototype=default_buffer_prototype(),
                )
            )
        return data

    if isinstance(data, Numcodec):
        try:
            # attempt to get a v3-api compatible version of this codec from the registry
            return get_codec(data.get_config())
        except KeyError:
            # if we could not find a v3-api compatible version of this codec, wrap it
            # in a NumcodecsWrapper
            return NumcodecWrapper(codec=data)

    raise TypeError(
        f"Invalid compressor. Expected None, a numcodecs.abc.Codec, or a dict representation of codec. Got {type(data)} instead."
    )
