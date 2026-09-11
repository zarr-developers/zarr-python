from __future__ import annotations

from dataclasses import dataclass

import pytest
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CpuBuffer
from zarr.core.dtype import UInt8

from zarr_codec.legacy import BytesBytesCodec, CodecPipeline


@dataclass(frozen=True)
class ReverseCodec(BytesBytesCodec):
    name: str = "reverse"

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        if len(chunk_data) != chunk_spec.shape[0]:
            raise ValueError("unexpected chunk size")
        return CpuBuffer.from_bytes(chunk_data.to_bytes()[::-1])

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await self._encode_single(chunk_data, chunk_spec)


@pytest.fixture
def spec() -> ArraySpec:
    return ArraySpec((3,), UInt8(), 0, ArrayConfig("C", False), default_buffer_prototype())


@pytest.mark.parametrize("values", [[], [None], [b"abc", None, b"xyz"], [None, b"abc"]])
async def test_batch_roundtrip(values: list[bytes | None], spec: ArraySpec) -> None:
    codec = ReverseCodec()
    inputs = [CpuBuffer.from_bytes(value) if value is not None else None for value in values]
    encoded = list(await codec.encode((value, spec) for value in inputs))
    assert [value.to_bytes() if value is not None else None for value in encoded] == [
        value[::-1] if value is not None else None for value in values
    ]
    decoded = await codec.decode((value, spec) for value in encoded)
    assert [value.to_bytes() if value is not None else None for value in decoded] == values
    assert codec.resolve_metadata(spec) is spec
    assert codec.evolve_from_array_spec(spec) is codec
    assert codec.to_dict() == {"name": "reverse"}
    assert ReverseCodec.from_dict({"name": "other"}).name == "other"


async def test_batch_propagates_codec_error(spec: ArraySpec) -> None:
    with pytest.raises(ValueError, match="unexpected chunk size"):
        await ReverseCodec().encode([(CpuBuffer.from_bytes(b"too long"), spec)])


def test_pipeline_requires_explicit_metadata_factory() -> None:
    # The inherited method must still reject this construction path; it must
    # not silently create an uninitialized pipeline.
    with pytest.raises(NotImplementedError, match="from_array_metadata_and_store"):
        CodecPipeline.from_array_metadata_and_store(None, None)  # type: ignore[arg-type]
