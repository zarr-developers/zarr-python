import numcodecs.registry as numcodecs_registry

from zarr.abc.codec import CodecJSON_V2
from zarr.codecs._v2 import Numcodec


def get_numcodec(data: CodecJSON_V2[str]) -> Numcodec:
    """
    Resolve a numcodec codec from the numcodecs registry.

    This requires the Numcodecs package to be installed.

    Parameters
    ----------
    data : CodecJSON_V2
        The JSON metadata for the codec.

    Returns
    -------
    codec : Numcodec

    Examples
    --------

    >>> codec = get_numcodec({'id': 'zlib', 'level': 1})
    >>> codec
    Zlib(level=1)
    """

    codec_id = data["id"]
    cls = numcodecs_registry.codec_registry.get(codec_id)
    if cls is None and data in numcodecs_registry.entries:
        cls = numcodecs_registry.entries[data].load()
        numcodecs_registry.register_codec(cls, codec_id=data)
    if cls is not None:
        return cls.from_config({k: v for k, v in data.items() if k != "id"})  # type: ignore[no-any-return]
    raise KeyError(data)
