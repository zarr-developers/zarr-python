from typing import Self, TypeGuard

from typing_extensions import Protocol

from zarr.abc.codec import CodecJSON_V2
from zarr.core.buffer import Buffer, NDBuffer


class Numcodec(Protocol):
    """
    A protocol that models the ``numcodecs.abc.Codec`` interface.
    """

    codec_id: str

    def encode(self, buf: Buffer | NDBuffer) -> Buffer | NDBuffer: ...

    def decode(
        self, buf: Buffer | NDBuffer, out: Buffer | NDBuffer | None = None
    ) -> Buffer | NDBuffer: ...

    def get_config(self) -> CodecJSON_V2[str]: ...

    @classmethod
    def from_config(cls, config: CodecJSON_V2[str]) -> Self: ...


def _is_numcodec_cls(obj: object) -> TypeGuard[type[Numcodec]]:
    """
    Check if the given object is a class implements the Numcodec protocol.

    The @runtime_checkable decorator does not allow issubclass checks for protocols with non-method
    members (i.e., attributes), so we use this function to manually check for the presence of the
    required attributes and methods on a given object.
    """
    return (
        isinstance(obj, type)
        and hasattr(obj, "codec_id")
        and isinstance(obj.codec_id, str)
        and hasattr(obj, "encode")
        and callable(obj.encode)
        and hasattr(obj, "decode")
        and callable(obj.decode)
        and hasattr(obj, "get_config")
        and callable(obj.get_config)
        and hasattr(obj, "from_config")
        and callable(obj.from_config)
    )


def _is_numcodec(obj: object) -> TypeGuard[Numcodec]:
    """
    Check if the given object implements the Numcodec protocol.

    The @runtime_checkable decorator does not allow issubclass checks for protocols with non-method
    members (i.e., attributes), so we use this function to manually check for the presence of the
    required attributes and methods on a given object.
    """
    return _is_numcodec_cls(type(obj))
