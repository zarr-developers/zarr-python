from __future__ import annotations

from zarr.buffer import Buffer
from zarr.common import BytesLike


def assert_bytes_equal(b1: Buffer | BytesLike | None, b2: Buffer | BytesLike | None) -> None:
    """Help function to assert if two bytes-like or Buffers are equal

    Warnings
    --------
    Always copies data, only use for testing and debugging
    """
    if isinstance(b1, Buffer):
        b1 = b1.to_bytes()
    if isinstance(b2, Buffer):
        b2 = b2.to_bytes()
    assert b1 == b2
