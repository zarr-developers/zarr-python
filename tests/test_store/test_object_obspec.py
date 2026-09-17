"""Tests for ``ObjectStore`` over a store that is not obstore.

``ObjectStore`` accepts any object implementing the async obspec protocols. The store
defined here is a minimal pure-Python implementation of those protocols: it inherits
from nothing, and it raises its own exception classes, named after the obspec
exceptions but not derived from them or from the builtins, so these tests fail if
``ObjectStore`` ever falls back to nominal checks.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING, Any, TypedDict, cast

import pytest

pytest.importorskip("obspec")

from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.storage import ObjectStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Iterable,
        Iterator,
        Sequence,
    )
    from collections.abc import (
        Buffer as BufferLike,
    )
    from pathlib import Path

    from obspec import Attributes, GetOptions, ListResult, ObjectMeta, PutMode, PutResult


class NotFoundError(Exception):
    """Deliberately not a ``FileNotFoundError``: only the name matches obspec."""


class NotSupportedError(Exception):
    pass


class AlreadyExistsError(Exception):
    pass


class PermissionDeniedError(Exception):
    """An obspec-named error that ``ObjectStore`` has no special handling for."""


class _GetResult:
    def __init__(self, path: str, data: bytes, byte_range: tuple[int, int]) -> None:
        self._path = path
        self._data = data
        self._range = byte_range

    @property
    def attributes(self) -> Attributes:
        return {}

    @property
    def meta(self) -> ObjectMeta:
        return _meta(self._path, self._data)

    @property
    def range(self) -> tuple[int, int]:
        return self._range

    async def buffer_async(self) -> BufferLike:
        return self._data

    async def __aiter__(self) -> AsyncIterator[BufferLike]:
        yield self._data


def _meta(path: str, data: bytes) -> ObjectMeta:
    return {
        "path": path,
        "last_modified": datetime.now(UTC),
        "size": len(data),
        "e_tag": None,
        "version": None,
    }


class DictObspecStore:
    """An in-memory implementation of the async obspec protocols ``ObjectStore`` uses.

    Suffix range requests raise ``NotSupportedError`` (as Azure does) unless
    ``supports_suffix`` is set, so that ``ObjectStore`` has to take its fallback path.
    """

    def __init__(self, *, supports_suffix: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.supports_suffix = supports_suffix

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DictObspecStore)
            and other.data == self.data
            and other.supports_suffix == self.supports_suffix
        )

    def _read(self, path: str) -> bytes:
        try:
            return self.data[path]
        except KeyError:
            raise NotFoundError(path) from None

    async def head_async(self, path: str) -> ObjectMeta:
        return _meta(path, self._read(path))

    async def get_async(self, path: str, *, options: GetOptions | None = None) -> _GetResult:
        data = self._read(path)
        byte_range = (options or {}).get("range")
        if byte_range is None:
            start, end = 0, len(data)
        elif isinstance(byte_range, dict):
            range_dict = cast("dict[str, int]", byte_range)
            if "suffix" in range_dict:
                if not self.supports_suffix:
                    raise NotSupportedError("suffix range requests are not supported")
                start, end = max(len(data) - range_dict["suffix"], 0), len(data)
            else:
                start, end = range_dict["offset"], len(data)
        else:
            start, end = byte_range[0], byte_range[1]
        return _GetResult(path, data[start:end], (start, end))

    async def get_range_async(
        self, path: str, *, start: int, end: int | None = None, length: int | None = None
    ) -> BufferLike:
        if end is None:
            assert length is not None
            end = start + length
        return self._read(path)[start:end]

    async def get_ranges_async(
        self,
        path: str,
        *,
        starts: Sequence[int],
        ends: Sequence[int] | None = None,
        lengths: Sequence[int] | None = None,
    ) -> Sequence[BufferLike]:
        if ends is None:
            assert lengths is not None
            ends = [start + length for start, length in zip(starts, lengths, strict=True)]
        data = self._read(path)
        return [data[start:end] for start, end in zip(starts, ends, strict=True)]

    async def put_async(
        self,
        path: str,
        file: IO[bytes]
        | Path
        | bytes
        | BufferLike
        | AsyncIterator[BufferLike]
        | AsyncIterable[BufferLike]
        | Iterator[BufferLike]
        | Iterable[BufferLike],
        *,
        attributes: Attributes | None = None,
        tags: dict[str, str] | None = None,
        mode: PutMode | None = None,
        use_multipart: bool | None = None,
        chunk_size: int = 5 * 1024 * 1024,
        max_concurrency: int = 12,
    ) -> PutResult:
        if mode == "create" and path in self.data:
            raise AlreadyExistsError(path)
        self.data[path] = bytes(memoryview(file))  # type: ignore[arg-type]
        return {"e_tag": None, "version": None}

    async def delete_async(self, paths: str | Sequence[str]) -> None:
        for path in [paths] if isinstance(paths, str) else paths:
            if path not in self.data:
                raise NotFoundError(path)
            del self.data[path]

    async def list_async(
        self, prefix: str | None = None, *, offset: str | None = None
    ) -> AsyncIterator[Sequence[ObjectMeta]]:
        # obspec evaluates prefixes per path segment: "c" matches "c/0" but not "cc/0".
        base = (prefix or "").strip("/")
        yield [
            _meta(k, v)
            for k, v in sorted(self.data.items())
            if not base or k == base or k.startswith(base + "/")
        ]

    async def list_with_delimiter_async(
        self, prefix: str | None = None
    ) -> ListResult[Sequence[ObjectMeta]]:
        base = (prefix or "").strip("/")
        base = base + "/" if base else ""
        common_prefixes: set[str] = set()
        objects: list[ObjectMeta] = []
        for key, value in sorted(self.data.items()):
            if not key.startswith(base):
                continue
            child, sep, _ = key[len(base) :].partition("/")
            if sep:
                common_prefixes.add(base + child)
            else:
                objects.append(_meta(key, value))
        return {"common_prefixes": sorted(common_prefixes), "objects": objects}


class StoreKwargs(TypedDict):
    store: DictObspecStore
    read_only: bool


class TestObspecObjectStore(StoreTests[ObjectStore[DictObspecStore], cpu.Buffer]):
    # store_cls is needed to do an isinstance check, so can't be a subscripted generic
    store_cls = ObjectStore  # type: ignore[assignment]
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self) -> StoreKwargs:
        return {"store": DictObspecStore(), "read_only": False}

    @pytest.fixture
    def store(self, store_kwargs: StoreKwargs) -> ObjectStore[DictObspecStore]:
        return self.store_cls(**store_kwargs)

    async def get(self, store: ObjectStore[DictObspecStore], key: str) -> Buffer:
        return self.buffer_cls.from_bytes(store.store.data[key])

    async def set(self, store: ObjectStore[DictObspecStore], key: str, value: Buffer) -> None:
        store.store.data[key] = value.to_bytes()

    def test_store_repr(self, store: ObjectStore[DictObspecStore]) -> None:
        assert repr(store).startswith("ObjectStore(object_store://")

    def test_store_supports_writes(self, store: ObjectStore[DictObspecStore]) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ObjectStore[DictObspecStore]) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: ObjectStore[DictObspecStore]) -> None:
        assert store.supports_listing


@pytest.mark.parametrize("supports_suffix", [True, False])
async def test_suffix_requests(supports_suffix: bool) -> None:
    """Suffix reads give the same result whether or not the store supports them natively."""
    store = ObjectStore(DictObspecStore(supports_suffix=supports_suffix))
    await store.set("key", cpu.Buffer.from_bytes(b"0123456789"))
    prototype = default_buffer_prototype()

    single = await store.get("key", prototype, SuffixByteRequest(3))
    assert single is not None
    assert single.to_bytes() == b"789"

    ranges: list[tuple[str, ByteRequest | None]] = [
        ("key", SuffixByteRequest(2)),
        ("key", RangeByteRequest(1, 3)),
        ("key", OffsetByteRequest(8)),
        ("key", None),
    ]
    observed = await store.get_partial_values(prototype, ranges)
    assert [buf.to_bytes() for buf in observed if buf is not None] == [
        b"89",
        b"12",
        b"89",
        b"0123456789",
    ]


async def test_set_if_not_exists_keeps_existing_value() -> None:
    store = ObjectStore(DictObspecStore())
    await store.set("key", cpu.Buffer.from_bytes(b"first"))
    await store.set_if_not_exists("key", cpu.Buffer.from_bytes(b"second"))
    assert store.store.data["key"] == b"first"


async def test_wrapping_an_obstore_store() -> None:
    """A wrapper that delegates to an obstore store is accepted, not just obstore itself."""
    pytest.importorskip("obstore")
    from obstore.store import MemoryStore

    class LoggingStore:
        def __init__(self, inner: MemoryStore) -> None:
            self.inner = inner
            self.calls: list[str] = []

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self.inner, name)
            if callable(attr):
                self.calls.append(name)
            return attr

    wrapper = LoggingStore(MemoryStore())
    store = ObjectStore(wrapper)
    await store.set("a", cpu.Buffer.from_bytes(b"x"))
    result = await store.get("a", default_buffer_prototype())
    assert result is not None
    assert result.to_bytes() == b"x"
    assert not await store.exists("b")
    assert "put_async" in wrapper.calls
    assert "head_async" in wrapper.calls


def test_init_rejects_object_without_obspec_methods() -> None:
    class PartialStore:
        async def get_async(self, path: str) -> None: ...

    with pytest.raises(TypeError, match=r"missing the following methods: delete_async, "):
        ObjectStore(PartialStore())  # type: ignore[type-var]


def test_init_requires_obspec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "obspec", None)
    with pytest.raises(ImportError, match="ObjectStore requires the obspec package"):
        ObjectStore(DictObspecStore())


async def test_unhandled_store_errors_propagate_unchanged() -> None:
    """Errors that ObjectStore does not handle are re-raised as-is, not as obspec copies."""
    store = ObjectStore(DictObspecStore())
    error = PermissionDeniedError("nope")

    async def deny(*args: Any, **kwargs: Any) -> Any:
        raise error

    store.store.head_async = deny  # type: ignore[method-assign]
    store.store.get_async = deny  # type: ignore[method-assign]
    store.store.delete_async = deny  # type: ignore[method-assign]
    store.store.put_async = deny  # type: ignore[method-assign]

    with pytest.raises(PermissionDeniedError) as info:
        await store.exists("key")
    assert info.value is error
    with pytest.raises(PermissionDeniedError) as info:
        await store.get("key", default_buffer_prototype())
    assert info.value is error
    with pytest.raises(PermissionDeniedError) as info:
        await store.delete("key")
    assert info.value is error
    with pytest.raises(PermissionDeniedError) as info:
        await store.set_if_not_exists("key", cpu.Buffer.from_bytes(b""))
    assert info.value is error


@pytest.mark.parametrize(
    "byte_range", [None, RangeByteRequest(0, 2), OffsetByteRequest(1), SuffixByteRequest(1)]
)
async def test_get_missing_key_returns_none(byte_range: ByteRequest | None) -> None:
    """A store's own NotFoundError is treated like the builtin FileNotFoundError."""
    store = ObjectStore(DictObspecStore())
    assert await store.get("missing", default_buffer_prototype(), byte_range) is None


async def test_get_partial_values_rejects_unknown_range() -> None:
    store = ObjectStore(DictObspecStore())
    with pytest.raises(ValueError, match="Unsupported range input"):
        await store.get_partial_values(
            default_buffer_prototype(),
            [("key", (0, 1))],  # type: ignore[list-item]
        )
