from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr.testing._deps import missing_dependency

try:
    import pytest
except ImportError as e:
    raise missing_dependency("pytest", __name__) from e

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

        return cast("bool", cupy.cuda.runtime.getDeviceCount() > 0)
    except ImportError:
        return False
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


gpu_mark = pytest.mark.gpu
skip_if_no_gpu = pytest.mark.skipif(not has_cupy(), reason="CuPy not installed or no GPU available")


# Decorator for GPU tests
def gpu_test[T](func: T) -> T:
    return cast(T, gpu_mark(skip_if_no_gpu(func)))
