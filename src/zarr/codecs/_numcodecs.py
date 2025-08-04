from zarr.abc.codec import CodecJSON_V2, Numcodec


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

    >>> codec = get_codec({'id': 'zlib', 'level': 1})
    >>> codec
    Zlib(level=1)
    """

    from numcodecs.registry import get_codec

    return get_codec(data)  # type: ignore[no-any-return]
