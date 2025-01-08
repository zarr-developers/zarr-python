from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store import ExplicitByteRequest, OffsetByteRequest, SuffixByteRequest

if TYPE_CHECKING:
    from zarr.abc.store import ByteRangeRequest
    from zarr.core.buffer import Buffer


def normalize_path(path: str | bytes | Path | None) -> str:
    if path is None:
        result = ""
    elif isinstance(path, bytes):
        result = str(path, "ascii")

    # handle pathlib.Path
    elif isinstance(path, Path):
        result = str(path)

    elif isinstance(path, str):
        result = path

    else:
        raise TypeError(f'Object {path} has an invalid type for "path": {type(path).__name__}')

    # convert backslash to forward slash
    result = result.replace("\\", "/")

    # remove leading and trailing slashes
    result = result.strip("/")

    # collapse any repeated slashes
    pat = re.compile(r"//+")
    result = pat.sub("/", result)

    # disallow path segments with just '.' or '..'
    segments = result.split("/")
    if any(s in {".", ".."} for s in segments):
        raise ValueError(
            f"The path {path!r} is invalid because its string representation contains '.' or '..' segments."
        )

    return result


def _normalize_byte_range_index(
    data: Buffer, byte_range: ByteRangeRequest | None
) -> tuple[int, int]:
    """
    Convert an ByteRangeRequest into an explicit start and stop
    """
    match byte_range:
        case None:
            return (0, len(data) + 1)
        case ExplicitByteRequest(start, end):
            return (start, end)
        case OffsetByteRequest(offset):
            return (offset, len(data) + 1)
        case SuffixByteRequest(suffix):
            return (len(data) - suffix, len(data) + 1)
        case _:
            raise ValueError(f"Unexpected byte_range, got {byte_range}.")
