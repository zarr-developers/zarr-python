"""The whole option space of `zarr.open`, checked against the invariants its contract implies.

`_Scenario` names everything that can vary: what is at the path before the call,
and the arguments to `open`. `_expected` is the contract written as an oracle,
independent of the implementation. `check_open` builds the scene, calls `open`,
and checks every invariant. `test_open_option_space` walks every discrete
combination; `test_open_properties` lets Hypothesis vary the parts that are not
discrete (path names, attribute contents, the consolidated-metadata key) and
shrink whatever it finds.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING, Any, Literal

import pytest

import zarr
from zarr import Array, Group
from zarr.abc.store import Store
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON
from zarr.core.sync import sync
from zarr.errors import (
    ArrayNotFoundError,
    ContainsArrayError,
    ContainsGroupError,
    NodeNotFoundError,
)
from zarr.storage import MemoryStore
from zarr.storage._wrapper import WrapperStore

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import event, given

from zarr.testing.strategies import _attr_keys, _attr_values, node_names

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Self

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import AccessModeLiteral, ZarrFormat

Existing = Literal["nothing", "array", "group"]
Kind = Literal["array", "group"]

MODES: tuple[AccessModeLiteral | None, ...] = ("r", "r+", "a", "w", "w-", None)
FORMATS: tuple[ZarrFormat | None, ...] = (2, 3, None)


class _RecordingStore(WrapperStore[Store]):
    """A store that counts every `get` per key and every write, so a test can see I/O."""

    get_counts: dict[str, int]
    writes: int

    def __init__(self, store: Store) -> None:
        super().__init__(store)
        self.get_counts = {}
        self.writes = 0

    def _with_store(self, store: Store) -> Self:
        # opening in mode "r" makes a read-only copy; it must count into the same tally
        new = type(self)(store)
        new.get_counts = self.get_counts
        new.__dict__["writes_owner"] = self
        return new

    def _record_write(self) -> None:
        owner = self.__dict__.get("writes_owner", self)
        owner.writes += 1

    def reset(self) -> None:
        self.get_counts.clear()
        self.writes = 0

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.get_counts[key] = self.get_counts.get(key, 0) + 1
        return await self._store.get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        self._record_write()
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._record_write()
        await self._store.set_if_not_exists(key, value)

    async def delete(self, key: str) -> None:
        self._record_write()
        await self._store.delete(key)

    async def delete_dir(self, prefix: str) -> None:
        self._record_write()
        await self._store.delete_dir(prefix)


@dataclasses.dataclass(frozen=True)
class _Scenario:
    existing: Existing
    existing_format: ZarrFormat
    consolidated: bool
    """For a group: whether consolidated metadata was written for it."""
    path: str
    mode: AccessModeLiteral | None
    zarr_format: ZarrFormat | None
    shape: bool
    """Whether `shape` (and `dtype`) are passed, which makes the call describe an array."""
    use_consolidated: bool | str | None
    """Only passed without `shape`; a str is the key the consolidated document is stored at."""
    attributes: dict[str, Any]

    @property
    def effective_mode(self) -> AccessModeLiteral:
        return self.mode or "a"  # the store is writable, so None means "a"

    @property
    def visible(self) -> bool:
        """Whether the existing node is one the requested format can see."""
        return self.existing != "nothing" and self.zarr_format in (None, self.existing_format)

    @property
    def created_format(self) -> ZarrFormat:
        return self.zarr_format or 3

    @property
    def consolidated_key(self) -> str:
        if isinstance(self.use_consolidated, str):
            return self.use_consolidated
        return ZMETADATA_V2_JSON


# ---------------------------------------------------------------------------
# the contract, as an oracle
# ---------------------------------------------------------------------------


def _create_outcome(sc: _Scenario, kind: Kind) -> Kind | type[Exception]:
    """What creating `kind` yields once `open` has decided to create.

    Creation checks the path in the format being created, and only that format,
    so an existing node of the other format goes unnoticed and the new node is
    made alongside it.
    """
    if sc.existing != "nothing" and sc.existing_format == sc.created_format:
        return ContainsArrayError if sc.existing == "array" else ContainsGroupError
    return kind


def _expected(sc: _Scenario) -> Kind | type[Exception]:
    """The contract of `open`: what comes back, or what is raised."""
    mode = sc.effective_mode
    if mode == "w-" and sc.existing != "nothing":
        return FileExistsError  # anything under the path is enough
    if mode in ("w", "w-"):
        # "w" empties the path before anything is read; "w-" found it empty
        return "array" if sc.shape else "group"
    if sc.shape:
        # the call describes an array: open_array semantics
        if sc.visible and sc.existing == "array":
            return "array"
        if sc.visible and sc.existing == "group" and sc.existing_format == 3:
            return ContainsGroupError  # format 3 says what it is in the one document read
        # nothing visible as an array: a format 2 group is not looked for
        if mode != "a":
            return ArrayNotFoundError
        return _create_outcome(sc, "array")
    if sc.visible:
        if sc.existing == "array":
            return "array"
        if sc.existing_format == 3 and isinstance(sc.use_consolidated, str):
            return TypeError
        if sc.use_consolidated and not sc.consolidated:
            return ValueError
        return "group"
    if mode != "a":
        return NodeNotFoundError
    return _create_outcome(sc, "group")


def _opened(sc: _Scenario) -> bool:
    """Whether the outcome is the existing node, as opposed to a new one."""
    expected = _expected(sc)
    if not isinstance(expected, str):
        return False
    return sc.visible and sc.effective_mode in ("r", "r+", "a") and (sc.existing == expected)


def _expected_reads(sc: _Scenario) -> set[str]:
    """The keys `open` reads, each exactly once, when it opens the existing node."""
    if sc.existing_format == 3:
        return {ZARR_JSON}
    if sc.shape:
        keys = {ZARRAY_JSON, ZATTRS_JSON}
    else:
        keys = {ZARRAY_JSON, ZGROUP_JSON, ZATTRS_JSON}
        if sc.use_consolidated is not False:
            keys.add(sc.consolidated_key)
    if sc.zarr_format is None:
        keys.add(ZARR_JSON)  # detecting the format costs the look that finds nothing
    return keys


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


def _snapshot(store: Store) -> dict[str, bytes]:
    from zarr.core.buffer import default_buffer_prototype

    async def _read_all() -> dict[str, bytes]:
        out = {}
        async for key in store.list():
            buf = await store.get(key, prototype=default_buffer_prototype())
            assert buf is not None
            out[key] = buf.to_bytes()
        return out

    return sync(_read_all())


def _build_scene(sc: _Scenario) -> _RecordingStore:
    store = _RecordingStore(MemoryStore())
    prefix = f"{sc.path}/" if sc.path else ""
    if sc.existing == "array":
        zarr.create_array(
            store,
            name=sc.path or None,
            shape=(3,),
            dtype="uint8",
            attributes=sc.attributes,
            zarr_format=sc.existing_format,
        )
    elif sc.existing == "group":
        zarr.create_group(
            store, path=sc.path, attributes=sc.attributes, zarr_format=sc.existing_format
        )
        if sc.consolidated:
            zarr.consolidate_metadata(store, path=sc.path)
            if sc.existing_format == 2 and isinstance(sc.use_consolidated, str):
                # the caller will ask for the document at a custom key: move it there
                from zarr.core.buffer import default_buffer_prototype

                doc = sync(store.get(prefix + ZMETADATA_V2_JSON, default_buffer_prototype()))
                assert doc is not None
                sync(store.set(prefix + sc.use_consolidated, doc))
                sync(store.delete(prefix + ZMETADATA_V2_JSON))
    return store


def check_open(sc: _Scenario, record: Callable[[str], None] = lambda label: None) -> None:
    """Build the scene for `sc`, call `open`, and check every invariant; `record` notes coverage."""
    store = _build_scene(sc)
    prefix = f"{sc.path}/" if sc.path else ""
    before = _snapshot(store)
    store.reset()
    kwargs: dict[str, Any] = (
        {"shape": (3,), "dtype": "uint8"} if sc.shape else {"use_consolidated": sc.use_consolidated}
    )

    def call() -> Array[Any] | Group:
        return zarr.open(
            store=store, mode=sc.mode, zarr_format=sc.zarr_format, path=sc.path, **kwargs
        )

    expected = _expected(sc)
    record(f"outcome={expected if isinstance(expected, str) else expected.__name__}")

    if not isinstance(expected, str):
        with pytest.raises(expected):
            call()
        if sc.effective_mode != "w":
            # a failed open leaves the store as it was
            assert store.writes == 0
            assert _snapshot(store) == before
        return

    node = call()
    assert isinstance(node, Array if expected == "array" else Group)
    assert node.path == sc.path

    if _opened(sc):
        record("opened")
        assert dict(node.attrs) == sc.attributes
        assert node.metadata.zarr_format == sc.existing_format
        # only the documents the node needs, each exactly once, and nothing written
        assert store.get_counts == {prefix + key: 1 for key in _expected_reads(sc)}
        assert store.writes == 0
        assert _snapshot(store) == before
        # opening again gives the same node, and so does the opener for that kind
        assert call().metadata == node.metadata
        same: Array[Any] | Group
        if isinstance(node, Array):
            same = zarr.open_array(
                store=store, mode=sc.effective_mode, zarr_format=sc.zarr_format, path=sc.path
            )
        else:
            same = zarr.open_group(
                store,
                mode=sc.effective_mode,
                zarr_format=sc.zarr_format,
                path=sc.path,
                use_consolidated=sc.use_consolidated,
            )
            assert (node.metadata.consolidated_metadata is not None) == (
                sc.consolidated and sc.use_consolidated is not False
            )
        assert same.metadata == node.metadata
    else:
        record("created")
        assert dict(node.attrs) == {}
        assert node.metadata.zarr_format == sc.created_format
        assert store.writes > 0
        # and it is there to be opened, in the format it was made in
        again = zarr.open(store=store, mode="r", zarr_format=sc.created_format, path=sc.path)
        assert type(again) is type(node)
        assert again.metadata == node.metadata


# ---------------------------------------------------------------------------
# every discrete combination
# ---------------------------------------------------------------------------


def _discrete_scenarios() -> Iterator[_Scenario]:
    existing_nodes: list[tuple[Existing, ZarrFormat, bool]] = [
        ("nothing", 3, False),
        *[("array", fmt, False) for fmt in (2, 3)],
        *[("group", fmt, cons) for fmt in (2, 3) for cons in (False, True)],
    ]
    for (existing, fmt, cons), path, mode, zarr_format in itertools.product(
        existing_nodes, ("", "outer/inner"), MODES, FORMATS
    ):
        calls: list[tuple[bool, bool | str | None]] = [
            (True, None),
            *[(False, uc) for uc in (None, False, True, "custom")],
        ]
        for shape, use_consolidated in calls:
            yield _Scenario(
                existing=existing,
                existing_format=fmt,
                consolidated=cons,
                path=path,
                mode=mode,
                zarr_format=zarr_format,
                shape=shape,
                use_consolidated=use_consolidated,
                attributes={"old": True},
            )


def _scenario_id(sc: _Scenario) -> str:
    node = sc.existing if sc.existing == "nothing" else f"{sc.existing}v{sc.existing_format}"
    if sc.consolidated:
        node += "c"
    call = "shape" if sc.shape else f"uc={sc.use_consolidated}"
    return f"{node}-{sc.path or 'root'}-mode={sc.mode}-fmt={sc.zarr_format}-{call}"


@pytest.mark.parametrize("sc", list(_discrete_scenarios()), ids=_scenario_id)
def test_open_option_space(sc: _Scenario) -> None:
    """`open` follows its contract for every combination of what is there and what is asked."""
    check_open(sc)


# ---------------------------------------------------------------------------
# and the parts that are not discrete
# ---------------------------------------------------------------------------

_names = node_names.filter(lambda name: not name.startswith("."))
_paths = st.just("") | st.lists(_names, min_size=1, max_size=3).map("/".join)
_attributes = st.dictionaries(_attr_keys, _attr_values, max_size=3)


@st.composite
def scenarios(draw: st.DrawFn) -> _Scenario:
    existing: Existing = draw(st.sampled_from(["nothing", "array", "group"]))
    shape = draw(st.booleans())
    return _Scenario(
        existing=existing,
        existing_format=draw(st.sampled_from([2, 3])),
        consolidated=existing == "group" and draw(st.booleans()),
        path=draw(_paths),
        mode=draw(st.sampled_from(MODES)),
        zarr_format=draw(st.sampled_from(FORMATS)),
        shape=shape,
        use_consolidated=None if shape else draw(st.none() | st.booleans() | _names),
        attributes=draw(_attributes),
    )


@given(sc=scenarios())
def test_open_properties(sc: _Scenario) -> None:
    """`open` follows its contract whatever the path, the attributes, and the consolidated key."""
    check_open(sc, record=event)
