"""Differential stateful test: every built-in store must agree with one reference model.

A Hypothesis state machine applies each operation to all stores at once and compares every result with
the contract below. A store that disagrees fails the test, unless the disagreement is a known divergence:
an entry in `KNOWN_DIVERGENCES` that records what that store does instead. `test_divergence_ledger_is_exact`
runs each entry's witness against every store, so each entry has to name exactly the stores that diverge
on its witness: a fixed store must be removed from the entry, and a newly diverging one must be added. A
divergence with no witness yet is found only when the state machine reaches it. Some entries apply on one
platform only, because the filesystem stores raise the operating system's own errors.

The contract (what the `Store` docstrings should promise):

- Keys use the grammar zarr produces through `StorePath`: `/`-separated segments, no empty, `.` or `..`
  segment, no leading or trailing `/`. A key is never also a directory of another key.
- A prefix names a directory: `"a"` and `"a/"` mean the same directory, and `""` means the root.
  `list_prefix`, `list_dir`, `getsize_prefix`, `is_empty` and `delete_dir` see only keys inside that
  directory: `list_prefix("lat")` never returns `lat_bnds/...`, and `list_prefix("a")` never returns a
  key named `a`.
- `list_dir` returns the names of the directory's direct children that contain keys.
- `exists`, `get` and `getsize` concern keys only. For any other name (a directory, a name below a key)
  they report a missing key.
- A byte range that reaches past the end of a value returns the bytes that exist, as Python slicing does.

The state machine never generates a case whose divergence would leave the stores in different states:
`delete` on a directory, `delete_dir` on a key, writing a key where a directory used to be, and keys outside
the grammar. Those divergences are in `UNMODELED_DIVERGENCES`, checked only by their witnesses.

Paths that leave a store's root (a leading `/`, `..`) are never generated: `LocalStore` joins them as real
paths, so `delete_dir("/")` runs `shutil.rmtree("/")`. `test_delete_stays_in_root` checks escapes against a
sentinel directory inside the test's own temporary directory.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, note, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import (
    FsspecStore,
    LocalStore,
    LoggingStore,
    ManagedMemoryStore,
    MemoryStore,
    ObjectStore,
    ZipStore,
)
from zarr.testing.store import LatencyStore

WINDOWS = sys.platform == "win32"

PROTOTYPE = default_buffer_prototype()

type Model = dict[str, bytes]
type Outcome = tuple[str, object]
"""`("ok", value)` or `("raised", exception class name)`."""

# --- stores -------------------------------------------------------------------------------------

type Cleanup = Callable[[], None]


@dataclass
class StoreUnderTest:
    name: str
    store: Store
    cleanup: Cleanup


def _no_cleanup() -> None:
    return None


def _remove_fsspec_memory_root(root: str) -> None:
    import fsspec

    fs = fsspec.filesystem("memory")
    if fs.exists(root):
        fs.rm(root, recursive=True)


def _make_memory(tmp: Path) -> tuple[Store, Cleanup]:
    return MemoryStore(), _no_cleanup


def _make_managed_memory(tmp: Path) -> tuple[Store, Cleanup]:
    return ManagedMemoryStore(name=uuid.uuid4().hex, path="sub/dir"), _no_cleanup


def _make_local(tmp: Path) -> tuple[Store, Cleanup]:
    return LocalStore(tmp / "local"), _no_cleanup


def _make_zip(tmp: Path) -> tuple[Store, Cleanup]:
    store = ZipStore(tmp / "store.zip", mode="w")
    return store, store.close


def _make_fsspec_local(tmp: Path) -> tuple[Store, Cleanup]:
    (tmp / "fsspec").mkdir()
    url = f"file://{tmp / 'fsspec'}"
    return FsspecStore.from_url(url, storage_options={"auto_mkdir": True}), _no_cleanup


def _make_fsspec_memory(tmp: Path) -> tuple[Store, Cleanup]:
    root = f"/{uuid.uuid4().hex}"
    return FsspecStore.from_url(f"memory://{root}"), partial(_remove_fsspec_memory_root, root)


def _make_obstore_local(tmp: Path) -> tuple[Store, Cleanup]:
    import obstore.store

    (tmp / "obstore").mkdir()
    return ObjectStore(obstore.store.LocalStore(str(tmp / "obstore"))), _no_cleanup


def _make_obstore_memory(tmp: Path) -> tuple[Store, Cleanup]:
    import obstore.store

    return ObjectStore(obstore.store.MemoryStore()), _no_cleanup


def _make_logging_memory(tmp: Path) -> tuple[Store, Cleanup]:
    return LoggingStore(MemoryStore(), log_level="WARNING"), _no_cleanup


def _make_latency_local(tmp: Path) -> tuple[Store, Cleanup]:
    return LatencyStore(LocalStore(tmp / "latency")), _no_cleanup


STORE_FACTORIES: dict[str, Callable[[Path], tuple[Store, Cleanup]]] = {
    "memory": _make_memory,
    "managed-memory": _make_managed_memory,
    "local": _make_local,
    "zip": _make_zip,
    "fsspec-local": _make_fsspec_local,
    "fsspec-memory": _make_fsspec_memory,
    "obstore-local": _make_obstore_local,
    "obstore-memory": _make_obstore_memory,
    "logging-memory": _make_logging_memory,
    "latency-local": _make_latency_local,
}


def _fsspec_unavailable() -> str | None:
    if importlib.util.find_spec("fsspec") is None:
        return "fsspec is not installed"
    import fsspec
    from packaging.version import parse as parse_version

    if parse_version(fsspec.__version__) < parse_version("2024.12.0"):
        # Wrapping the sync `file` and `memory` filesystems needs `AsyncFileSystemWrapper`.
        return "fsspec < 2024.12.0 has no AsyncFileSystemWrapper"
    return None


def _obstore_unavailable() -> str | None:
    return "obstore is not installed" if importlib.util.find_spec("obstore") is None else None


UNAVAILABLE: dict[str, str] = {
    name: reason
    for names, reason in (
        (("fsspec-local", "fsspec-memory"), _fsspec_unavailable()),
        (("obstore-local", "obstore-memory"), _obstore_unavailable()),
    )
    if reason is not None
    for name in names
}
"""Stores this environment can't build, and why."""

WRITABLE_STORES = tuple(name for name in STORE_FACTORIES if name not in {"zip", *UNAVAILABLE})
"""Stores that support every operation, including deletes and overwrites."""

APPEND_ONLY_STORES = ("zip",)
"""`ZipStore` can't delete or overwrite a key."""


def _build(name: str, tmp: Path) -> StoreUnderTest:
    if name in UNAVAILABLE:
        pytest.skip(UNAVAILABLE[name])
    store, cleanup = STORE_FACTORIES[name](tmp)
    target = StoreUnderTest(name, store, cleanup)
    try:
        sync(store._ensure_open())
    except BaseException:
        cleanup()
        raise
    return target


# --- the contract -------------------------------------------------------------------------------


def _directory(prefix: str) -> str:
    return prefix.strip("/")


def _is_under(key: str, directory: str) -> bool:
    return directory == "" or key.startswith(directory + "/")


def _keys_under(model: Model, prefix: str) -> list[str]:
    directory = _directory(prefix)
    return sorted(key for key in model if _is_under(key, directory))


def _children(model: Model, prefix: str) -> list[str]:
    directory = _directory(prefix)
    start = len(directory) + 1 if directory else 0
    return sorted({key[start:].split("/")[0] for key in _keys_under(model, prefix)})


def _directories(model: Model) -> set[str]:
    return {
        "/".join(key.split("/")[:depth]) for key in model for depth in range(1, key.count("/") + 1)
    }


@dataclass
class State:
    """The reference model, and every directory it has ever had."""

    model: Model = field(default_factory=dict)
    past_directories: set[str] = field(default_factory=set)
    """Filesystem stores keep a directory after its last key is deleted."""

    def record(self, operation: str, args: tuple[Any, ...]) -> None:
        """Apply a write as the contract says a store applies it."""
        match operation, args:
            case "set", (key, value):
                self.model[key] = value
            case "set_if_not_exists", (key, value):
                self.model.setdefault(key, value)
            case "delete", (key,):
                self.model.pop(key, None)
            case "delete_dir", (prefix,):
                for key in _keys_under(self.model, prefix):
                    del self.model[key]
        self.past_directories |= _directories(self.model)


def _slice(value: bytes, byte_range: ByteRequest | None) -> bytes:
    match byte_range:
        case None:
            return value
        case RangeByteRequest(start, end):
            return value[start:end]
        case OffsetByteRequest(offset):
            return value[offset:]
        case SuffixByteRequest(suffix):
            return value[max(0, len(value) - suffix) :]
    raise TypeError(byte_range)


def _expect_list(model: Model) -> Outcome:
    return ("ok", sorted(model))


def _expect_list_prefix(model: Model, prefix: str) -> Outcome:
    return ("ok", _keys_under(model, prefix))


def _expect_list_dir(model: Model, prefix: str) -> Outcome:
    return ("ok", _children(model, prefix))


def _expect_getsize_prefix(model: Model, prefix: str) -> Outcome:
    return ("ok", sum(len(model[key]) for key in _keys_under(model, prefix)))


def _expect_is_empty(model: Model, prefix: str) -> Outcome:
    return ("ok", not _keys_under(model, prefix))


def _expect_exists(model: Model, name: str) -> Outcome:
    return ("ok", name in model)


def _expect_get(model: Model, key: str, byte_range: ByteRequest | None) -> Outcome:
    if key not in model:
        return ("ok", None)
    return ("ok", _slice(model[key], byte_range))


def _expect_getsize(model: Model, key: str) -> Outcome:
    if key not in model:
        return ("raised", "FileNotFoundError")
    return ("ok", len(model[key]))


def _expect_none(model: Model, *args: Any) -> Outcome:
    return ("ok", None)


# --- operations ---------------------------------------------------------------------------------


async def _collect(iterator: AsyncIterator[str]) -> list[str]:
    return sorted([item async for item in iterator])


async def _call_list(store: Store) -> list[str]:
    return await _collect(store.list())


async def _call_list_prefix(store: Store, prefix: str) -> list[str]:
    return await _collect(store.list_prefix(prefix))


async def _call_list_dir(store: Store, prefix: str) -> list[str]:
    return await _collect(store.list_dir(prefix))


async def _call_get(store: Store, key: str, byte_range: ByteRequest | None) -> bytes | None:
    value = await store.get(key, PROTOTYPE, byte_range=byte_range)
    return None if value is None else value.to_bytes()


async def _call_set(store: Store, key: str, value: bytes) -> None:
    await store.set(key, PROTOTYPE.buffer.from_bytes(value))


async def _call_set_if_not_exists(store: Store, key: str, value: bytes) -> None:
    await store.set_if_not_exists(key, PROTOTYPE.buffer.from_bytes(value))


async def _call_method(method: str, store: Store, *args: Any) -> Any:
    return await getattr(store, method)(*args)


@dataclass(frozen=True)
class Operation:
    call: Callable[..., Any]
    """A coroutine function taking the store and the operation's arguments."""
    contract: Callable[..., Outcome]
    """The contract's outcome, given the model and the operation's arguments."""


OPERATIONS: dict[str, Operation] = {
    "list": Operation(_call_list, _expect_list),
    "list_prefix": Operation(_call_list_prefix, _expect_list_prefix),
    "list_dir": Operation(_call_list_dir, _expect_list_dir),
    "getsize_prefix": Operation(partial(_call_method, "getsize_prefix"), _expect_getsize_prefix),
    "is_empty": Operation(partial(_call_method, "is_empty"), _expect_is_empty),
    "exists": Operation(partial(_call_method, "exists"), _expect_exists),
    "get": Operation(_call_get, _expect_get),
    "getsize": Operation(partial(_call_method, "getsize"), _expect_getsize),
    "set": Operation(_call_set, _expect_none),
    "set_if_not_exists": Operation(_call_set_if_not_exists, _expect_none),
    "delete": Operation(partial(_call_method, "delete"), _expect_none),
    "delete_dir": Operation(partial(_call_method, "delete_dir"), _expect_none),
}


async def _outcome(operation: str, store: Store, args: tuple[Any, ...]) -> Outcome:
    # Call inside the event loop: `ObjectStore.list_dir` starts a coroutine as soon as it is called.
    try:
        return ("ok", await OPERATIONS[operation].call(store, *args))
    except Exception as e:  # noqa: BLE001
        return ("raised", type(e).__name__)


def _run(operation: str, store: Store, args: tuple[Any, ...]) -> Outcome:
    return sync(_outcome(operation, store, args))


# --- known divergences --------------------------------------------------------------------------

type Step = tuple[str, tuple[Any, ...]]
type Accepts = Callable[[Outcome, State, tuple[Any, ...]], bool]


@dataclass(frozen=True)
class Divergence:
    stores: frozenset[str]
    summary: str
    witness: tuple[Step, ...]
    """Steps whose last one breaks the contract on exactly the listed stores."""


@dataclass(frozen=True)
class KnownDivergence(Divergence):
    accepts: Accepts
    """Whether an outcome of the witness's last operation is this store's known behavior."""


def _raw_string_prefix(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (prefix,) = args
    return observed == ("ok", sorted(key for key in state.model if key.startswith(prefix)))


def _keys_under_or_named(model: Model, prefix: str) -> list[str]:
    """Directory semantics, except that a key named by the prefix itself counts as inside it."""
    named = [_directory(prefix)] if _directory(prefix) in model else []
    return sorted(_keys_under(model, prefix) + named)


def _list_prefix_with_named_key(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (prefix,) = args
    return observed == ("ok", _keys_under_or_named(state.model, prefix))


def _getsize_prefix_with_named_key(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (prefix,) = args
    keys = _keys_under_or_named(state.model, prefix)
    return observed == ("ok", sum(len(state.model[key]) for key in keys))


def _is_empty_with_named_key(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (prefix,) = args
    return observed == ("ok", not _keys_under_or_named(state.model, prefix))


def _lists_a_key_as_its_own_child(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (prefix,) = args
    name = _directory(prefix)
    return name in state.model and observed == ("ok", [name.rsplit("/", 1)[-1]])


def _lists_empty_directories(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    """Every extra child is a directory that held keys before deletes emptied it."""
    (prefix,) = args
    directory = _directory(prefix)
    match observed:
        case ("ok", list(names)):
            extra = set(names) - set(_children(state.model, prefix))
            paths = {f"{directory}/{name}" if directory else name for name in extra}
            return (
                bool(extra)
                and set(_children(state.model, prefix)) <= set(names)
                and (paths <= state.past_directories)
            )
    return False


def _is_a_directory(state: State, name: str) -> bool:
    return name not in state.model and name in state.past_directories


def _exists_for_a_directory(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (name,) = args
    return observed == ("ok", True) and _is_a_directory(state, name)


def _size_for_a_directory(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    (name,) = args
    return observed[0] == "ok" and isinstance(observed[1], int) and _is_a_directory(state, name)


def _is_below_a_key(state: State, name: str) -> bool:
    return any(_is_under(name, key) for key in state.model)


def _raises_below_a_key(error: str) -> Accepts:
    return partial(_raised_below_a_key, error)


def _raised_below_a_key(error: str, observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    name = args[0]
    return observed == ("raised", error) and _is_below_a_key(state, name)


def _raises_past_end(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    """A byte range that starts at or after the end of the value raises."""
    key, byte_range = args
    if key not in state.model:
        return False
    match byte_range:
        case RangeByteRequest(start, _) | OffsetByteRequest(start):
            return observed == ("raised", "GenericError") and start >= len(state.model[key])
    return False


def _rejects_empty_range(observed: Outcome, state: State, args: tuple[Any, ...]) -> bool:
    """A zero-length `RangeByteRequest` raises, even for a missing key."""
    _, byte_range = args
    match byte_range:
        case RangeByteRequest(start, end):
            return observed == ("raised", "ValueError") and start == end
    return False


def _below_a_key(stores: frozenset[str], error: str, operation: str) -> KnownDivergence:
    """`operation` on `k/x`, where `k` is a key, raises `error`."""
    args: tuple[Any, ...] = ("k/x", None) if operation == "get" else ("k/x",)
    return KnownDivergence(
        stores=stores,
        summary=f"{operation}() on a name below a key raises {error}",
        accepts=_raises_below_a_key(error),
        witness=(("set", ("k", b"v")), (operation, args)),
    )


def _raises_for_a_directory(error: str) -> Accepts:
    return partial(_raised_for_a_directory, error)


def _raised_for_a_directory(
    error: str, observed: Outcome, state: State, args: tuple[Any, ...]
) -> bool:
    name = args[0]
    return observed == ("raised", error) and _is_a_directory(state, name)


def _posix_only(stores: frozenset[str]) -> frozenset[str]:
    """Stores whose divergence comes from POSIX filesystem errors, which Windows doesn't raise."""
    return frozenset() if WINDOWS else stores


def _windows_only(stores: frozenset[str]) -> frozenset[str]:
    return stores if WINDOWS else frozenset()


FSSPEC = frozenset({"fsspec-local", "fsspec-memory"})
FILESYSTEMS = frozenset({"local", "latency-local", "fsspec-local", "obstore-local"})

KNOWN_DIVERGENCES: tuple[KnownDivergence, ...] = (
    KnownDivergence(
        stores=frozenset({"memory", "logging-memory", "zip", "managed-memory"}),
        summary="matches keys as strings: list_prefix('lat') returns lat_bnds/... (#3773, #4645)",
        accepts=_raw_string_prefix,
        witness=(
            ("set", ("lat/c", b"v")),
            ("set", ("lat_bnds/c", b"v")),
            ("list_prefix", ("lat",)),
        ),
    ),
    KnownDivergence(
        stores=FSSPEC,
        summary="a prefix naming a key returns that key",
        accepts=_list_prefix_with_named_key,
        witness=(("set", ("d/k", b"v")), ("list_prefix", ("d/k",))),
    ),
    KnownDivergence(
        stores=FSSPEC,
        summary="a prefix naming a key counts that key",
        accepts=_getsize_prefix_with_named_key,
        witness=(("set", ("d/k", b"v")), ("getsize_prefix", ("d/k",))),
    ),
    KnownDivergence(
        stores=FSSPEC,
        summary="a prefix naming a key is not empty",
        accepts=_is_empty_with_named_key,
        witness=(("set", ("d/k", b"v")), ("is_empty", ("d/k",))),
    ),
    KnownDivergence(
        stores=FSSPEC,
        summary="list_dir() on a top-level key's name returns the key itself",
        accepts=_lists_a_key_as_its_own_child,
        witness=(("set", ("k", b"v")), ("list_dir", ("k",))),
    ),
    KnownDivergence(
        stores=FILESYSTEMS,
        summary="lists directories left empty by deletes (members() then requests them, #4161)",
        accepts=_lists_empty_directories,
        witness=(("set", ("d/e/k", b"v")), ("delete", ("d/e/k",)), ("list_dir", ("d",))),
    ),
    KnownDivergence(
        stores=_windows_only(frozenset({"obstore-local"})),
        summary="exists() on a directory raises GenericError (Windows)",
        accepts=_raises_for_a_directory("GenericError"),
        witness=(("set", ("d/k", b"v")), ("exists", ("d",))),
    ),
    KnownDivergence(
        stores=_windows_only(frozenset({"obstore-local"})),
        summary="getsize() on a directory raises GenericError (Windows)",
        accepts=_raises_for_a_directory("GenericError"),
        witness=(("set", ("d/k", b"v")), ("getsize", ("d",))),
    ),
    KnownDivergence(
        stores=_windows_only(frozenset({"obstore-local"})),
        summary="get() on a directory raises GenericError (Windows)",
        accepts=_raises_for_a_directory("GenericError"),
        witness=(("set", ("d/k", b"v")), ("get", ("d", None))),
    ),
    KnownDivergence(
        stores=_windows_only(frozenset({"local", "latency-local", "fsspec-local"})),
        summary="get() on a directory raises PermissionError (Windows opens it as a file)",
        accepts=_raises_for_a_directory("PermissionError"),
        witness=(("set", ("d/k", b"v")), ("get", ("d", None))),
    ),
    KnownDivergence(
        stores=FSSPEC,
        summary="exists() is true for a directory",
        accepts=_exists_for_a_directory,
        witness=(("set", ("d/k", b"v")), ("exists", ("d",))),
    ),
    KnownDivergence(
        stores=frozenset({"local", "latency-local"} | FSSPEC),
        summary="getsize() returns a number for a directory instead of raising",
        accepts=_size_for_a_directory,
        witness=(("set", ("d/k", b"v")), ("getsize", ("d",))),
    ),
    KnownDivergence(
        stores=frozenset({"obstore-local", "obstore-memory"}),
        summary="a byte range starting at or after the end raises GenericError",
        accepts=_raises_past_end,
        witness=(("set", ("k", b"v")), ("get", ("k", OffsetByteRequest(1)))),
    ),
    KnownDivergence(
        stores=frozenset({"obstore-local", "obstore-memory"}),
        summary="a zero-length RangeByteRequest raises ValueError",
        accepts=_rejects_empty_range,
        witness=(("set", ("k", b"v")), ("get", ("k", RangeByteRequest(0, 0)))),
    ),
)
"""Divergences the state machine meets and accepts. Each witness ends with the divergent operation."""

KNOWN_DIVERGENCES += tuple(
    _below_a_key(_posix_only(frozenset({"obstore-local"})), "GenericError", operation)
    for operation in (
        "exists",
        "get",
        "getsize",
        "list_prefix",
        "list_dir",
        "getsize_prefix",
        "is_empty",
        "delete",
        "delete_dir",
    )
) + (
    _below_a_key(
        _posix_only(frozenset({"local", "latency-local", "fsspec-local"})),
        "NotADirectoryError",
        "getsize",
    ),
    _below_a_key(
        _posix_only(frozenset({"local", "latency-local"})), "NotADirectoryError", "delete"
    ),
    _below_a_key(_posix_only(frozenset({"fsspec-local"})), "NotADirectoryError", "list_dir"),
)

UNMODELED_DIVERGENCES: tuple[Divergence, ...] = (
    Divergence(
        stores=frozenset({"local", "latency-local"}),
        summary="delete_dir() on a key raises ValueError",
        witness=(("set", ("a", b"v")), ("delete_dir", ("a",))),
    ),
    Divergence(
        stores=FSSPEC,
        summary="delete_dir('a') deletes the key 'a'",
        witness=(("set", ("a", b"v")), ("delete_dir", ("a",)), ("list", ())),
    ),
    Divergence(
        stores=FILESYSTEMS,
        summary="a key can't be written where deletes left an empty directory of that name",
        witness=(("set", ("d/k", b"v")), ("delete", ("d/k",)), ("set", ("d", b"v"))),
    ),
)
"""Divergences that would leave stores in different states, so the state machine avoids them."""


def _known_divergence(
    store: str, operation: str, observed: Outcome, state: State, args: tuple[Any, ...]
) -> KnownDivergence | None:
    """The ledger entry that explains an outcome breaking the contract, if there is one."""
    for divergence in KNOWN_DIVERGENCES:
        if (
            store in divergence.stores
            and divergence.witness[-1][0] == operation
            and divergence.accepts(observed, state, args)
        ):
            return divergence
    return None


def _accepted(
    store: str, operation: str, observed: Outcome, state: State, args: tuple[Any, ...]
) -> bool:
    if observed == OPERATIONS[operation].contract(state.model, *args):
        return True
    return _known_divergence(store, operation, observed, state, args) is not None


# --- strategies ---------------------------------------------------------------------------------

SEGMENTS = ("a", "b", "la", "lat", "lat_bnds", "zarr.json", "0")
KEYS_BY_DEPTH = tuple(
    tuple("/".join(parts) for parts in product(SEGMENTS, repeat=depth)) for depth in (1, 2, 3)
)
VALUES = st.binary(max_size=12)


def _conflicts(key: str, model: Model) -> bool:
    """Whether `key` would be a directory of an existing key, or have one as a directory."""
    return any(_is_under(other, key) or _is_under(key, other) for other in model if other != key)


def _prefix_candidates(model: Model) -> list[str]:
    """Directories, keys, names cut short, names below a key, the root, and a missing name."""
    names = _directories(model) | set(model)
    cut_short = {name[:-1] for name in names if len(name) > 1}
    below_keys = {key + "/x" for key in model}
    trailing = {name + "/" for name in names}
    return sorted({"", "missing"} | names | cut_short | below_keys | trailing)


def _name_candidates(model: Model) -> list[str]:
    """The same names, restricted to the key grammar."""
    return [name for name in _prefix_candidates(model) if name and not name.endswith("/")]


def _byte_ranges(size: int) -> st.SearchStrategy[ByteRequest | None]:
    starts = st.integers(min_value=0, max_value=size + 3)
    lengths = st.integers(min_value=0, max_value=size + 3)
    return st.one_of(
        st.none(),
        st.builds(_range_request, starts, lengths),
        st.builds(OffsetByteRequest, starts),
        st.builds(SuffixByteRequest, st.integers(min_value=1, max_value=size + 3)),
    )


def _range_request(start: int, length: int) -> RangeByteRequest:
    return RangeByteRequest(start, start + length)


MOSTLY = st.sampled_from([True, True, True, False])
"""Three times out of four."""


# --- the state machine --------------------------------------------------------------------------


class StoreConformanceMachine(RuleBasedStateMachine):
    """Apply every operation to every store and compare each result with the contract."""

    store_names: tuple[str, ...]
    supports_deletes: bool

    def __init__(self) -> None:
        super().__init__()
        self.tmp = Path(tempfile.mkdtemp())
        self.stores: list[StoreUnderTest] = []
        self.state = State()

    @initialize(data=st.data())
    def open_stores(self, data: st.DataObject) -> None:
        self.stores = [_build(name, self.tmp) for name in self.store_names]
        # Start from a populated tree, so directory operations meet siblings and nesting early.
        self._write_some(data, st.integers(min_value=2, max_value=8))

    def teardown(self) -> None:
        for target in self.stores:
            target.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @property
    def model(self) -> Model:
        return self.state.model

    def _check(self, operation: str, *args: Any) -> None:
        note(f"{operation}{args!r}")
        failures = []
        for target in self.stores:
            observed = _run(operation, target.store, args)
            expected = OPERATIONS[operation].contract(self.model, *args)
            if observed == expected:
                continue
            divergence = _known_divergence(target.name, operation, observed, self.state, args)
            if divergence is None:
                failures.append(f"  {target.name}: {observed!r}, expected {expected!r}")
            else:
                event(f"known divergence: {target.name}: {divergence.summary}")
        assert not failures, f"{operation}{args!r} breaks the contract:\n" + "\n".join(failures)
        self.state.record(operation, args)

    def _draw_new_key(self, data: st.DataObject) -> str | None:
        """A key that doesn't clash with the tree; depth first, so shallow keys are common."""
        by_depth = [
            [
                key
                for key in keys
                if not _conflicts(key, self.model)
                and key not in self.state.past_directories
                and (self.supports_deletes or key not in self.model)
            ]
            for keys in KEYS_BY_DEPTH
        ]
        depths = [candidates for candidates in by_depth if candidates]
        if not depths:
            return None
        candidates = data.draw(st.sampled_from(depths), label="depth")
        return data.draw(st.sampled_from(candidates), label="key")

    def _write_some(self, data: st.DataObject, count: st.SearchStrategy[int]) -> None:
        for _ in range(data.draw(count, label="count")):
            key = self._draw_new_key(data)
            if key is None:
                return
            self._check("set", key, data.draw(VALUES, label="value"))

    # writes

    @rule(data=st.data())
    def set(self, data: st.DataObject) -> None:
        self._write_some(data, st.integers(min_value=1, max_value=3))

    @rule(data=st.data(), value=VALUES)
    def set_if_not_exists(self, data: st.DataObject, value: bytes) -> None:
        existing = data.draw(st.booleans(), label="existing") if self.model else False
        if existing:
            key: str | None = data.draw(st.sampled_from(sorted(self.model)), label="key")
        else:
            key = self._draw_new_key(data)
        if key is not None:
            self._check("set_if_not_exists", key, value)

    @precondition(lambda self: self.supports_deletes)
    @rule(data=st.data())
    def delete(self, data: st.DataObject) -> None:
        candidates = [*sorted(self.model), "missing"]
        self._check("delete", data.draw(st.sampled_from(candidates), label="key"))

    @precondition(lambda self: self.supports_deletes)
    @rule(data=st.data())
    def delete_dir(self, data: st.DataObject) -> None:
        candidates = [
            prefix
            for prefix in _prefix_candidates(self.model)
            if prefix and _directory(prefix) not in self.model
        ]
        self._check("delete_dir", data.draw(st.sampled_from(candidates), label="prefix"))

    @precondition(lambda self: self.supports_deletes and len(self.model) >= 6)
    @rule()
    def delete_everything(self) -> None:
        self._check("delete_dir", "")

    # reads

    @rule(
        data=st.data(),
        operation=st.sampled_from(["list_prefix", "list_dir", "getsize_prefix", "is_empty"]),
    )
    def read_directory(self, data: st.DataObject, operation: str) -> None:
        prefix = data.draw(st.sampled_from(_prefix_candidates(self.model)), label="prefix")
        self._check(operation, prefix)

    @rule(data=st.data(), operation=st.sampled_from(["exists", "getsize"]))
    def read_name(self, data: st.DataObject, operation: str) -> None:
        name = data.draw(st.sampled_from(_name_candidates(self.model)), label="name")
        self._check(operation, name)

    @rule(data=st.data())
    def get(self, data: st.DataObject) -> None:
        existing = data.draw(MOSTLY, label="existing") if self.model else False
        if existing:
            key = data.draw(st.sampled_from(sorted(self.model)), label="key")
        else:
            key = data.draw(st.sampled_from(_name_candidates(self.model)), label="key")
        size = len(self.model.get(key, b""))
        self._check("get", key, data.draw(_byte_ranges(size), label="byte_range"))

    @invariant()
    def keys_agree(self) -> None:
        if self.stores:
            self._check("list")


class WritableStoresMachine(StoreConformanceMachine):
    store_names = WRITABLE_STORES
    supports_deletes = True


class AppendOnlyStoresMachine(StoreConformanceMachine):
    store_names = APPEND_ONLY_STORES
    supports_deletes = False


# --- tests --------------------------------------------------------------------------------------

QUICK = settings(max_examples=25, stateful_step_count=25)


def test_writable_stores_agree_quick() -> None:
    run_state_machine_as_test(WritableStoresMachine, settings=QUICK)


def test_append_only_stores_agree_quick() -> None:
    run_state_machine_as_test(AppendOnlyStoresMachine, settings=QUICK)


@pytest.mark.slow_hypothesis
def test_writable_stores_agree() -> None:
    run_state_machine_as_test(WritableStoresMachine)


@pytest.mark.slow_hypothesis
def test_append_only_stores_agree() -> None:
    run_state_machine_as_test(AppendOnlyStoresMachine)


def _explained(store: str, steps: tuple[Step, ...], observed: Outcome, state: State) -> bool:
    """Whether the outcome of the last of `steps` is a ledger entry for this store.

    An unmodeled divergence explains it only when `steps` end with that divergence's whole witness.
    """
    operation, args = steps[-1]
    if _accepted(store, operation, observed, state, args):
        return True
    return any(
        store in divergence.stores and steps[-len(divergence.witness) :] == divergence.witness
        for divergence in UNMODELED_DIVERGENCES
    )


ALL_DIVERGENCES: tuple[Divergence, ...] = (*KNOWN_DIVERGENCES, *UNMODELED_DIVERGENCES)


@pytest.mark.parametrize("store_name", list(STORE_FACTORIES))
@pytest.mark.parametrize(
    "divergence", [pytest.param(d, id=d.summary[:48]) for d in ALL_DIVERGENCES]
)
def test_divergence_ledger_is_exact(
    store_name: str, divergence: Divergence, tmp_path: Path
) -> None:
    """A witness breaks the contract on exactly the stores its entry lists.

    A listed store that now follows the contract has been fixed: remove it from the entry. An unlisted
    store that breaks the contract has a divergence the ledger doesn't record: add it.
    """
    if store_name in APPEND_ONLY_STORES and any(
        operation.startswith("delete") for operation, _ in divergence.witness
    ):
        pytest.skip("the store can't delete")
    *setup, (operation, args) = divergence.witness
    state = State()
    target = _build(store_name, tmp_path)
    try:
        for index, (step, step_args) in enumerate(setup):
            observed = _run(step, target.store, step_args)
            if not _explained(store_name, divergence.witness[: index + 1], observed, state):
                pytest.fail(f"setup step {step}{step_args!r} breaks the contract: {observed!r}")
            state.record(step, step_args)
        observed = _run(operation, target.store, args)
    finally:
        target.cleanup()
    diverges = observed != OPERATIONS[operation].contract(state.model, *args)
    if store_name in divergence.stores:
        assert diverges, f"{store_name} now follows the contract; remove it from this entry"
        if isinstance(divergence, KnownDivergence):
            assert divergence.accepts(observed, state, args), observed
    else:
        assert not diverges or _explained(store_name, divergence.witness, observed, state), (
            f"{store_name} breaks the contract here too ({observed!r}); add it to this entry"
        )


ROOT_ESCAPES = {
    ("local", "relative"),
    ("local", "absolute"),
    ("latency-local", "relative"),
    ("latency-local", "absolute"),
    ("fsspec-local", "relative"),
}
"""Stores whose `delete_dir` reaches a directory outside their root, by path form."""


@pytest.mark.parametrize(
    ("store_name", "form"),
    [
        pytest.param(
            store_name,
            form,
            marks=[pytest.mark.xfail(strict=True, raises=AssertionError)]
            if (store_name, form) in ROOT_ESCAPES
            else [],
        )
        for store_name in sorted(FILESYSTEMS)
        for form in ("relative", "absolute")
    ],
)
def test_delete_stays_in_root(store_name: str, form: str, tmp_path: Path) -> None:
    """`delete_dir` must not reach a directory outside the store's root.

    The target is a sentinel directory beside the store, inside this test's temporary directory, so an
    escape can only delete the sentinel.
    """
    sentinel = tmp_path / "outside"
    sentinel.mkdir()
    (sentinel / "keep").write_bytes(b"x")
    (tmp_path / "stores").mkdir()
    target = _build(store_name, tmp_path / "stores")
    path = "../../outside" if form == "relative" else str(sentinel)
    try:
        _run("delete_dir", target.store, (path,))
    finally:
        target.cleanup()
    assert (sentinel / "keep").exists()
