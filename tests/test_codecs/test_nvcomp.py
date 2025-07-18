import typing

import pytest

import zarr
from zarr.abc.store import Store
from zarr.buffer.gpu import buffer_prototype
from zarr.codecs import NvcompZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.storage import StorePath
from zarr.testing.utils import gpu_test

if typing.TYPE_CHECKING:
    from zarr.core.common import JSON


# the type-ignores here are here thanks to not reliably having GPU
# libraries in the pre-commit mypy environment.


@gpu_test  # type: ignore[misc,unused-ignore]
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    "checksum",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None)),  # everything
        (slice(1, None), slice(1, None)),  # top-left chunk is empty
    ],
)
def test_nvcomp_zstd(store: Store, checksum: bool, selection: tuple[slice, slice]) -> None:
    import cupy as cp

    with zarr.config.enable_gpu():
        data = cp.arange(0, 256, dtype="uint16").reshape((16, 16))

        a = zarr.create_array(
            StorePath(store, path="nvcomp_zstd"),
            shape=data.shape,
            chunks=(4, 4),
            dtype=data.dtype,
            fill_value=0,
            compressors=NvcompZstdCodec(level=0, checksum=checksum),
        )

        a[*selection] = data

        if selection == (slice(None), slice(None)):
            cp.testing.assert_array_equal(data[*selection], a[*selection])
            cp.testing.assert_array_equal(data[:, :], a[:, :])
        else:
            expected = cp.full(data.shape, a.fill_value)
            expected[*selection] = data[*selection]
            cp.testing.assert_array_equal(expected, a)
            cp.testing.assert_array_equal(expected[:, :], a[:, :])


def test_invalid_raises() -> None:
    with pytest.raises(ValueError):
        NvcompZstdCodec(level=100, checksum=False)

    with pytest.raises(TypeError):
        NvcompZstdCodec(level="100", checksum=False)  # type: ignore[arg-type,unused-ignore]

    with pytest.raises(TypeError):
        NvcompZstdCodec(checksum="False")  # type: ignore[arg-type,unused-ignore]


@gpu_test  # type: ignore[misc,unused-ignore]
def test_uses_default_codec() -> None:
    with zarr.config.enable_gpu():
        a = zarr.create_array(
            StorePath(zarr.storage.MemoryStore(), path="nvcomp_zstd"),
            shape=(10, 10),
            chunks=(10, 10),
            dtype="int32",
        )
        assert a.metadata.zarr_format == 3
        assert isinstance(a.metadata.codecs[-1], NvcompZstdCodec)


@gpu_test  # type: ignore[misc,unused-ignore]
def test_nvcomp_from_dict() -> None:
    config: dict[str, JSON] = {
        "name": "zstd",
        "configuration": {
            "level": 0,
            "checksum": False,
        },
    }
    codec = NvcompZstdCodec.from_dict(config)
    assert codec.level == 0
    assert codec.checksum is False


@gpu_test  # type: ignore[misc,unused-ignore]
def test_compute_encoded_chunk_size() -> None:
    codec = NvcompZstdCodec(level=0, checksum=False)
    with pytest.raises(NotImplementedError):
        codec.compute_encoded_size(
            _input_byte_length=0,
            _chunk_spec=ArraySpec(
                shape=(10, 10),
                dtype=zarr.core.dtype.npy.int.Int32,
                fill_value=0,
                config=ArrayConfig(order="C", write_empty_chunks=False),
                prototype=buffer_prototype,
            ),
        )
