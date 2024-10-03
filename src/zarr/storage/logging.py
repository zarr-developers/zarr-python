from __future__ import annotations

import inspect
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Self

from zarr.abc.store import AccessMode, ByteRangeRequest, Store
from zarr.core.buffer import Buffer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import AccessModeLiteral


class LoggingStore(Store):
    _store: Store
    counter: defaultdict[str, int]

    def __init__(
        self,
        store: Store,
        log_level: str = "DEBUG",
        log_handler: logging.Handler | None = None,
    ) -> None:
        self._store = store
        self.counter = defaultdict(int)
        self.log_level = log_level
        self.log_handler = log_handler

        self._configure_logger(log_level, log_handler)

    def _configure_logger(
        self, log_level: str = "DEBUG", log_handler: logging.Handler | None = None
    ) -> None:
        self.log_level = log_level
        self.logger = logging.getLogger(f"LoggingStore({self._store!s})")
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            if not log_handler:
                log_handler = self._default_handler()
            # Add handler to logger
            self.logger.addHandler(log_handler)

    def _default_handler(self) -> logging.Handler:
        """Define a default log handler"""
        handler = logging.StreamHandler()
        handler.setLevel(self.log_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        return handler

    @contextmanager
    def log(self) -> Generator[None, None, None]:
        method = inspect.stack()[2].function
        op = f"{type(self._store).__name__}.{method}"
        self.logger.info(f"Calling {op}")
        start_time = time.time()
        try:
            self.counter[method] += 1
            yield
        finally:
            end_time = time.time()
            self.logger.info(f"Finished {op} in {end_time - start_time:.2f} seconds")

    @property
    def supports_writes(self) -> bool:
        with self.log():
            return self._store.supports_writes

    @property
    def supports_deletes(self) -> bool:
        with self.log():
            return self._store.supports_deletes

    @property
    def supports_partial_writes(self) -> bool:
        with self.log():
            return self._store.supports_partial_writes

    @property
    def supports_listing(self) -> bool:
        with self.log():
            return self._store.supports_listing

    @property
    def _mode(self) -> AccessMode:  # type: ignore[override]
        with self.log():
            return self._store._mode

    @property
    def _is_open(self) -> bool:  # type: ignore[override]
        with self.log():
            return self._store._is_open

    async def _open(self) -> None:
        with self.log():
            return await self._store._open()

    async def _ensure_open(self) -> None:
        with self.log():
            return await self._store._ensure_open()

    async def empty(self) -> bool:
        with self.log():
            return await self._store.empty()

    async def clear(self) -> None:
        with self.log():
            return await self._store.clear()

    def __str__(self) -> str:
        return f"logging-{self._store!s}"

    def __repr__(self) -> str:
        return f"LoggingStore({repr(self._store)!r})"

    def __eq__(self, other: object) -> bool:
        with self.log():
            return self._store == other

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        with self.log():
            return await self._store.get(key=key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        with self.log():
            return await self._store.get_partial_values(prototype=prototype, key_ranges=key_ranges)

    async def exists(self, key: str) -> bool:
        with self.log():
            return await self._store.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        with self.log():
            return await self._store.set(key=key, value=value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        with self.log():
            return await self._store.set_if_not_exists(key=key, value=value)

    async def delete(self, key: str) -> None:
        with self.log():
            return await self._store.delete(key=key)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        with self.log():
            return await self._store.set_partial_values(key_start_values=key_start_values)

    async def list(self) -> AsyncGenerator[str, None]:
        with self.log():
            async for key in self._store.list():
                yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        with self.log():
            async for key in self._store.list_prefix(prefix=prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        with self.log():
            async for key in self._store.list_dir(prefix=prefix):
                yield key

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        with self.log():
            return type(self)(
                self._store.with_mode(mode),
                log_level=self.log_level,
                log_handler=self.log_handler,
            )
