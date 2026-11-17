from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import FunctionType
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast

import pytest

from zarr.abc.store import ByteRequest, RangeByteRequest, Store, _store_supports_sync_io
from zarr.core.buffer import Buffer
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.core.buffer.cpu import buffer_prototype
from zarr.experimental.cache_store import CacheStore
from zarr.storage import LocalStore, LoggingStore, MemoryStore, WrapperStore, ZipStore
from zarr.testing.store import LatencyStore, StoreTests

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
    from pathlib import Path
    from types import TracebackType

    from zarr.core.buffer.core import BufferPrototype


class StoreKwargs(TypedDict):
    store: LocalStore


class OpenKwargs(TypedDict):
    store_cls: type[LocalStore]
    root: str


# TODO: fix this warning
@pytest.mark.filterwarnings(
    "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
)
class TestWrapperStore(StoreTests[WrapperStore[Any], Buffer]):
    store_cls = WrapperStore
    buffer_cls = CPUBuffer

    async def get(self, store: WrapperStore[LocalStore], key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store._store.root / key).read_bytes())

    async def set(self, store: WrapperStore[LocalStore], key: str, value: Buffer) -> None:
        parent = (store._store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store._store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmp_path: Path) -> StoreKwargs:
        return {"store": LocalStore(str(tmp_path))}

    @pytest.fixture
    def open_kwargs(self, tmp_path: Path) -> OpenKwargs:
        return {"store_cls": LocalStore, "root": str(tmp_path)}

    def test_store_supports_writes(self, store: WrapperStore[LocalStore]) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: WrapperStore[LocalStore]) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: WrapperStore[LocalStore]) -> None:
        assert f"{store!r}" == f"WrapperStore(LocalStore, 'file://{store._store.root.as_posix()}')"

    def test_store_str(self, store: WrapperStore[LocalStore]) -> None:
        assert str(store) == f"wrapping-file://{store._store.root.as_posix()}"

    def test_check_writeable(self, store: WrapperStore[LocalStore]) -> None:
        """
        Test _check_writeable() runs without errors.
        """
        store._check_writable()

    def test_close(self, store: WrapperStore[LocalStore]) -> None:
        "Test store can be closed"
        store.close()
        assert not store._is_open

    def test_is_open_setter_raises(self, store: WrapperStore[LocalStore]) -> None:
        """
        Test that a user cannot change `_is_open` without opening the underlying store.
        """
        with pytest.raises(
            NotImplementedError, match="WrapperStore must be opened via the `_open` method"
        ):
            store._is_open = True


# TODO: work out where warning is coming from and fix
@pytest.mark.filterwarnings(
    "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
)
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_set(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisySetter(WrapperStore[Store]):
        async def set(self, key: str, value: Buffer) -> None:
            print(f"setting {key}")
            await super().set(key, value)

    key = "foo"
    value = CPUBuffer.from_bytes(b"bar")
    store_wrapped = NoisySetter(store)
    await store_wrapped.set(key, value)
    captured = capsys.readouterr()
    assert f"setting {key}" in captured.out
    assert await store_wrapped.get(key, buffer_prototype) == value


@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_get(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisyGetter(WrapperStore[Any]):
        async def get(
            self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
        ) -> None:
            print(f"getting {key}")
            await super().get(key, prototype=prototype, byte_range=byte_range)

    key = "foo"
    value = CPUBuffer.from_bytes(b"bar")
    store_wrapped = NoisyGetter(store)
    await store_wrapped.set(key, value)
    await store_wrapped.get(key, buffer_prototype)
    captured = capsys.readouterr()
    assert f"getting {key}" in captured.out


@pytest.mark.parametrize(
    ("store_factory", "expected"),
    [
        (lambda tmp: MemoryStore(), True),
        (lambda tmp: LocalStore(str(tmp)), True),
        (lambda tmp: WrapperStore(MemoryStore()), True),
        (lambda tmp: LatencyStore(MemoryStore()), True),
        (lambda tmp: ZipStore(tmp / "store.zip", mode="w"), False),
        (lambda tmp: WrapperStore(ZipStore(tmp / "store.zip", mode="w")), False),
    ],
    ids=[
        "memory",
        "local",
        "wrapper-of-memory",
        "latency-wrapper-of-memory",
        "zip",
        "wrapper-of-zip",
    ],
)
def test_supports_sync_io(store_factory: Any, expected: bool, tmp_path: Path | Any) -> None:
    """`_store_supports_sync_io` is True only for stores implementing the full
    sync surface (get_sync + set_sync + delete_sync); wrappers forward the
    wrapped store's capability via `_supports_sync_io`."""
    assert _store_supports_sync_io(store_factory(tmp_path)) is expected


def test_wrapper_get_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous get"):
        store.get_sync("key")


def test_wrapper_set_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous set"):
        store.set_sync("key", CPUBuffer.from_bytes(b"data"))


def test_wrapper_delete_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous delete"):
        store.delete_sync("key")


# --- Every wrapper forwards every Store method -------------------------------
#
# `RecordingStore` overrides every method of `Store` (and the sync protocols) to
# record its arguments and return a distinct value. Wrapping it and calling a
# method on the wrapper must produce exactly one call of the same method, with
# the same arguments, on the wrapped store, and must return what that call
# returned. A method the wrapper doesn't forward falls back to a `Store`
# default, which reaches the wrapped store through some other method (e.g.
# `getsize` through `get`), and the check fails.


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]


def _make_result(name: str) -> Any:
    """A value the wrapped store returns from `name`, distinct from every other."""
    member = vars(RecordingStore)[name]
    if isinstance(member, property):
        return object()
    if inspect.signature(member).return_annotation == "None":
        return None
    if name in {"get", "get_sync"}:
        # `CacheStore` caches what `get` returns, so it must be a Buffer.
        return CPUBuffer.from_bytes(name.encode())
    if name in {"with_read_only", "__enter__"}:
        # Wrappers re-wrap what these return, so it must be a Store.
        return MemoryStore()
    return object()


class RecordingStore(Store):
    """A store that records each call and returns a distinct value for each method."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.results: dict[str, Any] = {}

    def _record(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(Call(name, {k: v for k, v in arguments.items() if k != "self"}))
        return self.results.setdefault(name, _make_result(name))

    def with_read_only(self, read_only: bool = False) -> Store:
        return cast(Store, self._record("with_read_only", locals()))

    def __enter__(self) -> Self:
        return cast(Self, self._record("__enter__", locals()))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return cast(None, self._record("__exit__", locals()))

    async def _open(self) -> None:
        return cast(None, self._record("_open", locals()))

    async def _ensure_open(self) -> None:
        return cast(None, self._record("_ensure_open", locals()))

    async def is_empty(self, prefix: str) -> bool:
        return cast(bool, self._record("is_empty", locals()))

    async def clear(self) -> None:
        return cast(None, self._record("clear", locals()))

    @property
    def read_only(self) -> bool:
        return cast(bool, self._record("read_only", {}))

    def _check_writable(self) -> None:
        return cast(None, self._record("_check_writable", locals()))

    def __eq__(self, value: object) -> bool:
        return self is value

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        return cast(Buffer, self._record("get", locals()))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return cast(list[Buffer | None], self._record("get_partial_values", locals()))

    async def exists(self, key: str) -> bool:
        return cast(bool, self._record("exists", locals()))

    @property
    def supports_writes(self) -> bool:
        return cast(bool, self._record("supports_writes", {}))

    async def set(self, key: str, value: Buffer) -> None:
        return cast(None, self._record("set", locals()))

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        return cast(None, self._record("set_if_not_exists", locals()))

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        return cast(None, self._record("_set_many", locals()))

    @property
    def supports_consolidated_metadata(self) -> bool:
        return cast(bool, self._record("supports_consolidated_metadata", {}))

    @property
    def supports_deletes(self) -> bool:
        return cast(bool, self._record("supports_deletes", {}))

    async def delete(self, key: str) -> None:
        return cast(None, self._record("delete", locals()))

    @property
    def supports_listing(self) -> bool:
        return cast(bool, self._record("supports_listing", {}))

    async def list(self) -> AsyncIterator[str]:
        yield self._record("list", locals())

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        yield self._record("list_prefix", locals())

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        yield self._record("list_dir", locals())

    async def delete_dir(self, prefix: str) -> None:
        return cast(None, self._record("delete_dir", locals()))

    def close(self) -> None:
        return cast(None, self._record("close", locals()))

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        yield self._record("_get_many", locals())

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int = 10,
        max_gap_bytes: int = 1 << 20,
        max_coalesced_bytes: int = 16 << 20,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        yield self._record("get_ranges", locals())

    def get_ranges_sync(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_gap_bytes: int = 1 << 20,
        max_coalesced_bytes: int = 16 << 20,
    ) -> Sequence[tuple[int, Buffer | None]]:
        return cast(
            "Sequence[tuple[int, Buffer | None]]", self._record("get_ranges_sync", locals())
        )

    async def getsize(self, key: str) -> int:
        return cast(int, self._record("getsize", locals()))

    async def getsize_prefix(self, prefix: str) -> int:
        return cast(int, self._record("getsize_prefix", locals()))

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        return cast(Buffer, self._record("get_sync", locals()))

    def set_sync(self, key: str, value: Buffer) -> None:
        return cast(None, self._record("set_sync", locals()))

    def delete_sync(self, key: str) -> None:
        return cast(None, self._record("delete_sync", locals()))


@dataclass(frozen=True)
class Recipe:
    """How to call a method on a wrapper, and what kind of value it gives back."""

    kind: Literal["property", "sync", "async", "aiter", "rewrap"]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


VALUE = CPUBuffer.from_bytes(b"value")
RANGES = [RangeByteRequest(0, 1), RangeByteRequest(4, 6)]
# Coalescing settings that differ from every default, so a wrapper that drops
# them is caught.
COALESCING = {"max_gap_bytes": 5, "max_coalesced_bytes": 7}

RECIPES: dict[str, Recipe] = {
    "with_read_only": Recipe("rewrap", (True,)),
    "__enter__": Recipe("rewrap"),
    "__exit__": Recipe("sync", (None, None, None)),
    "_open": Recipe("async"),
    "_ensure_open": Recipe("async"),
    "is_empty": Recipe("async", ("a/b",)),
    "clear": Recipe("async"),
    "read_only": Recipe("property"),
    "_check_writable": Recipe("sync"),
    "get": Recipe("async", ("a/b", buffer_prototype, RangeByteRequest(1, 3))),
    "get_partial_values": Recipe("async", (buffer_prototype, [("a/b", RangeByteRequest(1, 3))])),
    "exists": Recipe("async", ("a/b",)),
    "supports_writes": Recipe("property"),
    "set": Recipe("async", ("a/b", VALUE)),
    "set_if_not_exists": Recipe("async", ("a/b", VALUE)),
    "_set_many": Recipe("async", ([("a/b", VALUE)],)),
    "supports_consolidated_metadata": Recipe("property"),
    "supports_deletes": Recipe("property"),
    "delete": Recipe("async", ("a/b",)),
    "supports_listing": Recipe("property"),
    "list": Recipe("aiter"),
    "list_prefix": Recipe("aiter", ("a/",)),
    "list_dir": Recipe("aiter", ("a/",)),
    "delete_dir": Recipe("async", ("a",)),
    "close": Recipe("sync"),
    "_get_many": Recipe("aiter", ([("a/b", buffer_prototype, None)],)),
    "get_ranges": Recipe(
        "aiter",
        ("a/b", RANGES),
        {"prototype": buffer_prototype, "max_concurrency": 3, **COALESCING},
    ),
    "get_ranges_sync": Recipe(
        "sync", ("a/b", RANGES), {"prototype": buffer_prototype, **COALESCING}
    ),
    "getsize": Recipe("async", ("a/b",)),
    "getsize_prefix": Recipe("async", ("a/",)),
    "get_sync": Recipe("sync", ("a/b",), {"prototype": buffer_prototype, "byte_range": None}),
    "set_sync": Recipe("sync", ("a/b", VALUE)),
    "delete_sync": Recipe("sync", ("a/b",)),
}

# Members of `Store` that are not a call a wrapper passes through.
NOT_FORWARDED: dict[str, str] = {
    "__init__": "constructs the wrapper",
    "open": "constructs the wrapper",
    "__eq__": "compares the wrapped stores of two wrappers, so the argument changes",
    "supports_partial_writes": "always False, on every store",
}

# Methods a wrapper deliberately implements itself instead of forwarding. Each
# entry must hold: the test fails if the wrapper starts forwarding the method.
REROUTED: dict[tuple[str, str], str] = {
    ("latency", "get_partial_values"): "reads through the wrapper's own get, to add latency",
    ("latency", "get_ranges"): "reads through the wrapper's own get, to add latency",
    ("latency", "get_ranges_sync"): "reads through the wrapper's own get_sync, to add latency",
    ("cache", "__enter__"): "CacheStore can't re-wrap a store, so it raises NotImplementedError",
}

WRAPPERS: dict[str, Callable[[Store], WrapperStore[Any]]] = {
    "wrapper": WrapperStore,
    "logging": LoggingStore,
    "latency": LatencyStore,
    "cache": lambda store: CacheStore(store, cache_store=MemoryStore()),
}

# Every member of the `Store` interface, including the sync protocol a wrapper
# presents.
STORE_MEMBERS = {
    name
    for name, member in vars(Store).items()
    if isinstance(member, FunctionType | property | classmethod)
} | {"get_sync", "set_sync", "delete_sync"}


def _arguments(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The arguments a call binds to, with defaults filled in."""
    bound = inspect.signature(getattr(RecordingStore, name)).bind(None, *args, **kwargs)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k != "self"}


async def _forwarded(wrapper_name: str, name: str) -> tuple[bool, list[Call]]:
    """Whether the wrapper passes one call of `name` straight to the store it
    wraps, and the calls the wrapped store saw."""
    inner = RecordingStore()
    wrapper = WRAPPERS[wrapper_name](inner)
    recipe = RECIPES[name]
    try:
        if recipe.kind == "property":
            result = getattr(wrapper, name)
        elif recipe.kind == "aiter":
            result = [item async for item in getattr(wrapper, name)(*recipe.args, **recipe.kwargs)]
        elif recipe.kind == "async":
            result = await getattr(wrapper, name)(*recipe.args, **recipe.kwargs)
        else:
            result = getattr(wrapper, name)(*recipe.args, **recipe.kwargs)
    except NotImplementedError:
        return False, inner.calls
    expected = inner.results.get(name)
    if recipe.kind == "aiter":
        returned = len(result) == 1 and result[0] is expected
    elif recipe.kind == "rewrap":
        returned = type(result) is type(wrapper) and result._store is expected
    else:
        returned = result is expected
    arguments = {} if recipe.kind == "property" else _arguments(name, recipe.args, recipe.kwargs)
    return returned and inner.calls == [Call(name, arguments)], inner.calls


def test_every_store_member_is_checked() -> None:
    """Each `Store` member has a recipe or a reason it isn't forwarded, and
    `RecordingStore` records every member with a recipe."""
    assert set(RECIPES) | set(NOT_FORWARDED) == STORE_MEMBERS
    assert set(RECIPES) <= set(vars(RecordingStore))


@pytest.mark.parametrize("name", sorted(RECIPES))
@pytest.mark.parametrize("wrapper_name", sorted(WRAPPERS))
async def test_wrappers_forward_every_store_method(wrapper_name: str, name: str) -> None:
    """A wrapper forwards each `Store` method to the store it wraps, with the
    same arguments, unless `REROUTED` says why it doesn't."""
    forwarded, calls = await _forwarded(wrapper_name, name)
    expected = (wrapper_name, name) not in REROUTED
    assert forwarded is expected, f"{wrapper_name}.{name}: the wrapped store saw {calls}"
