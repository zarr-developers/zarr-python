from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer


def normalize_path(path: str | bytes | Path | None) -> str:
    # handle bytes
    if path is None:
        result = ""
    elif isinstance(path, bytes):
        result = str(path, "ascii")
    # ensure str
    # handle pathlib.Path and upath.Path
    elif isinstance(path, Path):
        result = str(path)

    elif isinstance(path, str):
        result = path

    else:
        raise TypeError(f'Object {path} has an invalid type for "path": {type(path).__name__}')

    # convert backslash to forward slash
    result = result.replace("\\", "/")

    # ensure no leading slash
    while len(result) > 0 and result[0] == "/":
        result = result[1:]

    # ensure no trailing slash
    while len(result) > 0 and result[-1] == "/":
        result = result[:-1]

    # collapse any repeated slashes
    previous_char = None
    collapsed = ""
    for char in result:
        if char == "/" and previous_char == "/":
            pass
        else:
            collapsed += char
        previous_char = char
    result = collapsed

    # disallow path segments with just '.' or '..'
    segments = result.split("/")
    if any(s in {".", ".."} for s in segments):
        raise ValueError(
            f"The path {path!r} is invalid because its string representation contains '.' or '..' segments."
        )

    return result


def _normalize_interval_index(
    data: Buffer, interval: None | tuple[int | None, int | None]
) -> tuple[int, int]:
    """
    Convert an implicit interval into an explicit start and length
    """
    if interval is None:
        start = 0
        length = len(data)
    else:
        maybe_start, maybe_len = interval
        if maybe_start is None:
            start = 0
        else:
            start = maybe_start

        if maybe_len is None:
            length = len(data) - start
        else:
            length = maybe_len

    return (start, length)
