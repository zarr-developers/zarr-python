from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr.core.buffer import Buffer

if TYPE_CHECKING:
    from zarr.core.common import BytesLike

__all__ = ["assert_bytes_equal"]


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


def has_cupy() -> bool:
    try:
        import cupy

        return cast(bool, cupy.cuda.runtime.getDeviceCount() > 0)
    except ImportError:
        return False
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


# Decorator for GPU tests
def gpu_test(func: Any) -> Any:
    return pytest.mark.gpu(
        pytest.mark.skipif(not has_cupy(), reason="CuPy not installed or no GPU available")(func)
    )
