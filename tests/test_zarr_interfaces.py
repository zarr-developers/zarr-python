"""Tests that externally-defined codecs and data types (subclassing
zarr_interfaces, not zarr) are recognized by zarr's internal machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

import numpy as np
from zarr_interfaces.codec.v1 import ArrayArrayCodec as IArrayArrayCodec
from zarr_interfaces.codec.v1 import ArrayBytesCodec as IArrayBytesCodec
from zarr_interfaces.codec.v1 import BytesBytesCodec as IBytesBytes
from zarr_interfaces.data_type.v1 import ZDType as IZDType
from zarr_interfaces.metadata.v1 import Metadata as IMetadata

# ---------------------------------------------------------------------------
# Verify zarr's classes satisfy the interfaces
# ---------------------------------------------------------------------------


class TestZarrClassesSatisfyInterfaces:
    def test_array_array_codec(self) -> None:
        """zarr's ArrayArrayCodec is a subclass of the interface."""
        from zarr.abc.codec import ArrayArrayCodec

        assert issubclass(ArrayArrayCodec, IArrayArrayCodec)

    def test_array_bytes_codec(self) -> None:
        """zarr's ArrayBytesCodec is a subclass of the interface."""
        from zarr.abc.codec import ArrayBytesCodec

        assert issubclass(ArrayBytesCodec, IArrayBytesCodec)

    def test_bytes_bytes_codec(self) -> None:
        """zarr's BytesBytesCodec is a subclass of the interface."""
        from zarr.abc.codec import BytesBytesCodec

        assert issubclass(BytesBytesCodec, IBytesBytes)

    def test_zdtype(self) -> None:
        """zarr's ZDType is a subclass of the interface."""
        from zarr.core.dtype.wrapper import ZDType

        assert issubclass(ZDType, IZDType)

    def test_concrete_dtype_is_interface_instance(self) -> None:
        """A concrete zarr dtype is an instance of the interface ZDType."""
        from zarr.core.dtype.npy.float import Float64

        assert isinstance(Float64(), IZDType)

    def test_metadata_protocol(self) -> None:
        """zarr's Metadata class satisfies the Metadata protocol."""
        from zarr.abc.metadata import Metadata

        assert isinstance(Metadata, type)
        # Metadata instances should satisfy the protocol
        from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

        enc = DefaultChunkKeyEncoding(separator="/")
        assert isinstance(enc, IMetadata)

    def test_concrete_codec_is_interface_instance(self) -> None:
        """A concrete zarr codec is an instance of the interface ABC."""
        from zarr.codecs.bytes import BytesCodec

        assert isinstance(BytesCodec(), IArrayBytesCodec)

    def test_concrete_bb_codec_is_interface_instance(self) -> None:
        """A concrete zarr BytesBytesCodec is an instance of the interface ABC."""
        from zarr.codecs.gzip import GzipCodec

        assert isinstance(GzipCodec(), IBytesBytes)


# ---------------------------------------------------------------------------
# External codec defined using only zarr_interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalIdentityCodec(IArrayArrayCodec):
    """An array-to-array codec defined using only zarr_interfaces.
    Simulates what a third-party package would do.
    """

    is_fixed_size: ClassVar[bool] = True

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: Any) -> int:
        return input_byte_length

    def _decode_sync(self, chunk_data: Any, chunk_spec: Any) -> Any:
        return chunk_data

    async def _decode_single(self, chunk_data: Any, chunk_spec: Any) -> Any:
        return self._decode_sync(chunk_data, chunk_spec)

    def _encode_sync(self, chunk_data: Any, chunk_spec: Any) -> Any:
        return chunk_data

    async def _encode_single(self, chunk_data: Any, chunk_spec: Any) -> Any:
        return self._encode_sync(chunk_data, chunk_spec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"name": "external_identity"}


class TestExternalCodecCompatibility:
    def test_isinstance_zarr_abc(self) -> None:
        """External codec passes isinstance against zarr's ABC."""
        from zarr.abc.codec import ArrayArrayCodec

        codec = ExternalIdentityCodec()
        assert isinstance(codec, ArrayArrayCodec)

    def test_isinstance_interface(self) -> None:
        """External codec passes isinstance against the interface."""
        codec = ExternalIdentityCodec()
        assert isinstance(codec, IArrayArrayCodec)

    def test_codecs_from_list(self) -> None:
        """External codec is correctly classified by codecs_from_list."""
        from zarr.codecs.bytes import BytesCodec
        from zarr.core.codec_pipeline import codecs_from_list

        aa, ab, bb = codecs_from_list([ExternalIdentityCodec(), BytesCodec()])
        assert len(aa) == 1
        assert isinstance(aa[0], IArrayArrayCodec)
        assert isinstance(ab, IArrayBytesCodec)
        assert len(bb) == 0

    def test_roundtrip_through_array(self) -> None:
        """External codec works in a real zarr array encode/decode cycle."""
        import zarr
        from zarr.registry import register_codec

        register_codec("external_identity", ExternalIdentityCodec)

        arr = zarr.create_array(
            store={},
            shape=(10,),
            dtype="float64",
            chunks=(10,),
            filters=[ExternalIdentityCodec()],
            compressors=None,
            fill_value=0,
        )
        data = np.arange(10, dtype="float64")
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)


# ---------------------------------------------------------------------------
# External dtype defined using only zarr_interfaces
# ---------------------------------------------------------------------------


class TestExternalDtypeCompatibility:
    def test_isinstance_zarr_zdtype(self) -> None:
        """A class subclassing the interface ZDType passes isinstance against zarr's ZDType."""
        from zarr.core.dtype.wrapper import ZDType

        # We can't easily instantiate an abstract ZDType subclass without
        # implementing all methods, but we can verify the class hierarchy
        @dataclass(frozen=True, kw_only=True, slots=True)
        class ExternalDType(IZDType[np.dtype[np.float32], np.float32]):
            dtype_cls: ClassVar[type] = np.dtype
            _zarr_v3_name: ClassVar[str] = "external_float32"

            @classmethod
            def from_native_dtype(cls, dtype: Any) -> Self:
                return cls()

            def to_native_dtype(self) -> np.dtype[np.float32]:
                return np.dtype(np.float32)

            @classmethod
            def _from_json_v2(cls, data: Any) -> Self:
                return cls()

            @classmethod
            def _from_json_v3(cls, data: Any) -> Self:
                return cls()

            def to_json(self, zarr_format: Literal[2, 3]) -> Any:
                return "external_float32"

            def _check_scalar(self, data: object) -> bool:
                return isinstance(data, float | int | np.floating)

            def cast_scalar(self, data: object) -> np.float32:
                return np.float32(data)  # type: ignore[arg-type]

            def default_scalar(self) -> np.float32:
                return np.float32(0)

            def from_json_scalar(self, data: Any, *, zarr_format: Literal[2, 3]) -> np.float32:
                return np.float32(data)

            def to_json_scalar(self, data: object, *, zarr_format: Literal[2, 3]) -> Any:
                return float(data)  # type: ignore[arg-type]

        ext = ExternalDType()
        assert isinstance(ext, IZDType)
        assert isinstance(ext, ZDType)
