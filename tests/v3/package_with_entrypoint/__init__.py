from numpy import ndarray

from zarr.abc.codec import ArrayBytesCodec
from zarr.array_spec import ArraySpec
from zarr.common import BytesLike


class TestCodec(ArrayBytesCodec):
    is_fixed_size = True

    async def encode(
        self,
        chunk_array: ndarray,
        chunk_spec: ArraySpec,
    ) -> BytesLike | None:
        pass

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
    ) -> ndarray:
        pass

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length
