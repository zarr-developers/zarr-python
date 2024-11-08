from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from types import TracebackType
    from typing import Any, Self

    from zarr.abc.store import ByteRangeRequest
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import AccessModeLiteral, BytesLike

from zarr.abc.store import AccessMode, Store

T_Wrapped = TypeVar("T_Wrapped", bound=Store)


class WrapperStore(Store, Generic[T_Wrapped]):
    """
    A store class that wraps an existing ``Store`` instance.
    By default all of the store methods are delegated to the wrapped store instance, which is
    accessible via the ``._wrapped`` attribute of this class.

    Use this class to modify or extend the behavior of the other store classes.
    """

    _wrapped: T_Wrapped

    def __init__(self, wrapped: T_Wrapped) -> None:
        self._wrapped = wrapped

    @classmethod
    async def open(
        cls: type[Self], wrapped_class: type[T_Wrapped], *args: Any, **kwargs: Any
    ) -> Self:
        wrapped = wrapped_class(*args, **kwargs)
        await wrapped._open()
        return cls(wrapped=wrapped)

    def __enter__(self) -> Self:
        return type(self)(self._wrapped.__enter__())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self._wrapped.__exit__(exc_type, exc_value, traceback)

    async def _open(self) -> None:
        await self._wrapped._open()

    async def _ensure_open(self) -> None:
        await self._wrapped._ensure_open()

    async def empty(self) -> bool:
        return await self._wrapped.empty()

    async def clear(self) -> None:
        return await self._wrapped.clear()

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        return type(self)(wrapped=self._wrapped.with_mode(mode=mode))

    @property
    def mode(self) -> AccessMode:
        return self._wrapped._mode

    def _check_writable(self) -> None:
        return self._wrapped._check_writable()

    def __eq__(self, value: object) -> bool:
        return type(self) is type(value) and self._wrapped.__eq__(value)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRangeRequest | None = None
    ) -> Buffer | None:
        return await self._wrapped.get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        return await self._wrapped.get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        return await self._wrapped.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        await self._wrapped.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        return await self._wrapped.set_if_not_exists(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        await self._wrapped._set_many(values)

    @property
    def supports_writes(self) -> bool:
        return self._wrapped.supports_writes

    @property
    def supports_deletes(self) -> bool:
        return self._wrapped.supports_deletes

    async def delete(self, key: str) -> None:
        await self._wrapped.delete(key)

    @property
    def supports_partial_writes(self) -> bool:
        return self._wrapped.supports_partial_writes

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        return await self._wrapped.set_partial_values(key_start_values)

    @property
    def supports_listing(self) -> bool:
        return self._wrapped.supports_listing

    def list(self) -> AsyncGenerator[str]:
        return self._wrapped.list()

    def list_prefix(self, prefix: str) -> AsyncGenerator[str]:
        return self._wrapped.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncGenerator[str]:
        return self._wrapped.list_dir(prefix)

    async def delete_dir(self, prefix: str) -> None:
        return await self._wrapped.delete_dir(prefix)

    def close(self) -> None:
        self._wrapped.close()

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRangeRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        async for req in self._wrapped._get_many(requests):
            yield req
