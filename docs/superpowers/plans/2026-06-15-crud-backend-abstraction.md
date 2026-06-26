# Backend-agnostic CRUD layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the low-level functional CRUD API into a backend-agnostic `zarr.crud` package with a pure-Python reference backend and the existing zarrs bindings as a second, interchangeable backend.

**Architecture:** A narrow async `CrudBackend` protocol (byte/metadata level) plus a shared `zarr.crud` facade that holds all backend-neutral logic (selection normalization, numpy assembly, dtype handling, `read_encoded_chunk` via `store.get`). Two backends conform: `ReferenceBackend` (pure Python, wraps zarr-python's own codec pipeline / indexer / metadata machinery) and `ZarrsBackend` (wraps `_zarrs_bindings`). A registry + `zarr.config` key `crud.backend` (default `"reference"`) selects one; every facade function also takes `backend=`.

**Tech Stack:** Python 3.12+, numpy, zarr-python internals (`BatchedCodecPipeline`, `AsyncArray`, `save_metadata`, `ArrayConfig`/`ArraySpec`, chunk-key encoding), the existing `_zarrs_bindings` Rust extension (unchanged — no Rust build needed).

Spec: `docs/superpowers/specs/2026-06-15-crud-backend-abstraction-design.md`.

---

## Environment notes (read first)

- **Run python/pytest/mypy via `uv run`.** The zarrs backend needs the extension: `uv run --group zarrs pytest ...`. The reference backend works under plain `uv run pytest ...`.
- The Claude Code bash sandbox is broken on this host (`bwrap: loopback` error). Run commands with the sandbox **disabled**.
- **No Rust changes in this plan.** The `_zarrs_bindings` pyfunctions keep their existing names (`retrieve_chunk`, `store_chunk`, `erase_chunk`, `retrieve_array_subset`, `retrieve_encoded_chunk`, `create_array`, `create_group`, `read_metadata`, `delete_node`, `list_children`); `ZarrsBackend` adapts them to the contract's verb names. No `cargo` build or `uv sync --reinstall` is required, but the `zarrs` group must already be installed (`uv sync --group zarrs`) to run the zarrs-parametrized tests.
- Pre-commit hooks (ruff strict, mypy strict over `src`+`tests`, codespell) run on `git commit`. If a hook rewrites a file, `git add` and commit again.
- Docstrings use markdown (single backticks), not RST.
- pytest is configured with `asyncio_mode = "auto"` — async tests/fixtures need no decorator.

## File structure

```
src/zarr/crud/
  __init__.py        # public exports; registers the reference backend at import
  _backend.py        # CrudBackend Protocol + NodeExistsError
  _registry.py       # register_backend / get_backend + config default resolution
  _reference.py      # ReferenceBackend (pure Python)
  _api.py            # shared async facade (the public functions) + neutral helpers
src/zarr/zarrs/
  __init__.py        # SHRINKS: version + register ZarrsBackend; no _api re-exports
  _backend.py        # ZarrsBackend (wraps _zarrs_bindings) — NEW
  _bridge.py         # unchanged
  _api.py            # DELETED
src/zarr/core/config.py   # add  "crud": {"backend": "reference"}
tests/crud/
  __init__.py
  conftest.py        # store fixture, backend fixture (reference+zarrs), metadata helpers
  test_registry.py   # registry + default + override
  test_reference_backend.py   # direct reference-backend smoke tests
  test_crud.py       # full differential suite, parametrized over backend x store
tests/zarrs/
  __init__.py        # unchanged
  conftest.py        # unchanged (still used by test_bridge/test_cache)
  test_bridge.py     # unchanged
  test_cache.py      # imports updated to zarr.crud read_chunk/write_chunk, backend="zarrs"
  test_node.py       # DELETED (covered by tests/crud/test_crud.py)
  test_chunk.py      # DELETED (covered by tests/crud/test_crud.py)
  test_api.py        # DELETED (replaced by tests/crud import coverage)
changes/+zarrs-bindings.feature.md   # reworded for zarr.crud
.github/workflows/zarrs.yml          # run tests/crud tests/zarrs
```

---

### Task 1: `zarr.crud` skeleton — protocol, exceptions, registry, config

**Files:**
- Create: `src/zarr/crud/__init__.py`
- Create: `src/zarr/crud/_backend.py`
- Create: `src/zarr/crud/_registry.py`
- Modify: `src/zarr/core/config.py`
- Create: `tests/crud/__init__.py` (empty)
- Test: `tests/crud/test_registry.py`

- [ ] **Step 1: Write the failing test** — `tests/crud/test_registry.py`

```python
from __future__ import annotations

import pytest

from zarr.crud import CrudBackend, NodeExistsError, get_backend, register_backend


def test_node_exists_error_is_value_error() -> None:
    assert issubclass(NodeExistsError, ValueError)


def test_default_backend_is_reference() -> None:
    # the reference backend is registered at import and is the configured default
    be = get_backend()
    assert be is get_backend("reference")


def test_get_unknown_backend_raises() -> None:
    with pytest.raises(KeyError, match="no CRUD backend"):
        get_backend("does-not-exist")


def test_register_and_resolve_instance() -> None:
    class Dummy:
        pass

    dummy = Dummy()
    register_backend("dummy-test", dummy)  # type: ignore[arg-type]
    try:
        assert get_backend("dummy-test") is dummy
    finally:
        from zarr.crud import _registry

        _registry._BACKENDS.pop("dummy-test", None)


def test_protocol_is_runtime_checkable() -> None:
    # ReferenceBackend (registered as "reference") structurally satisfies the protocol
    assert isinstance(get_backend("reference"), CrudBackend)
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/crud/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zarr.crud'`

- [ ] **Step 3: Create `tests/crud/__init__.py`** (empty file)

- [ ] **Step 4: Create `src/zarr/crud/_backend.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


class NodeExistsError(ValueError):
    """Raised when a node already exists at a path and overwrite was not requested."""


@runtime_checkable
class CrudBackend(Protocol):
    """The byte/metadata-level contract a CRUD backend must implement.

    Methods take neutral types: the metadata document as a `dict`, a zarr
    `Store`, and plain zarr paths (`""`, `"foo/bar"`). They return raw bytes,
    parsed JSON documents, or `None`. The shared `zarr.crud` facade builds the
    numpy- and selection-level API on top of these.

    `create_*` raise `zarr.crud.NodeExistsError` when a node exists and
    `overwrite` is false. `read_metadata`/`delete_node`/`list_children` raise
    `zarr.errors.NodeNotFoundError` when the target is missing.
    """

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None: ...

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None: ...

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]: ...

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes: ...

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes: ...

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None: ...

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None: ...

    async def delete_node(self, store: Store, path: str) -> None: ...

    async def list_children(
        self, store: Store, path: str
    ) -> list[tuple[str, dict[str, JSON]]]: ...
```

- [ ] **Step 5: Create `src/zarr/crud/_registry.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.config import config

if TYPE_CHECKING:
    from zarr.crud._backend import CrudBackend

_BACKENDS: dict[str, CrudBackend] = {}


def register_backend(name: str, backend: CrudBackend) -> None:
    """Register a CRUD backend instance under `name`."""
    _BACKENDS[name] = backend


def get_backend(name: str | None = None) -> CrudBackend:
    """Resolve a backend by name, or the configured default when `name` is None.

    Selecting `"zarrs"` imports `zarr.zarrs` if needed so it can self-register.
    """
    if name is None:
        name = config.get("crud.backend")
    if name not in _BACKENDS and name == "zarrs":
        import zarr.zarrs  # noqa: F401  (import registers the zarrs backend)
    if name not in _BACKENDS:
        raise KeyError(
            f"no CRUD backend registered as {name!r}; registered: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]
```

- [ ] **Step 6: Create `src/zarr/crud/__init__.py`** (reference backend is added in Task 2; for now register nothing)

```python
"""
Backend-agnostic low-level functional CRUD API for zarr hierarchies.

The public functions delegate byte- and metadata-level work to a `CrudBackend`.
Two backends ship: a pure-Python reference backend (the default) and a
zarrs-accelerated backend (`zarr.zarrs`, requires the `zarrs-bindings`
extension). Select one with the `crud.backend` config key or a per-call
`backend=` argument.

Array routines take an explicit metadata document (a `dict` matching the
`zarr.json` / `.zarray` document) rather than reading it from the store, which
makes read-only and virtual views possible.
"""

from zarr.crud._backend import CrudBackend, NodeExistsError
from zarr.crud._registry import get_backend, register_backend

__all__ = [
    "CrudBackend",
    "NodeExistsError",
    "get_backend",
    "register_backend",
]
```

- [ ] **Step 7: Add the config default** — `src/zarr/core/config.py`

Find the defaults mapping passed to the `Config(...)` constructor (it contains the `"codec_pipeline"` key). Add a sibling entry:

```python
        "crud": {"backend": "reference"},
```

Run to confirm it loads: `uv run python -c "from zarr.core.config import config; print(config.get('crud.backend'))"`
Expected: `reference`

- [ ] **Step 8: Run the test (note: `test_default_backend_is_reference` and the protocol test still fail — reference backend arrives in Task 2)**

Run: `uv run pytest tests/crud/test_registry.py -v`
Expected: `test_node_exists_error_is_value_error`, `test_get_unknown_backend_raises`, `test_register_and_resolve_instance` PASS; `test_default_backend_is_reference` and `test_protocol_is_runtime_checkable` FAIL (KeyError: no backend `reference`). That is expected at this task boundary; they pass after Task 2.

- [ ] **Step 9: Commit**

```bash
git add src/zarr/crud/_backend.py src/zarr/crud/_registry.py src/zarr/crud/__init__.py src/zarr/core/config.py tests/crud/__init__.py tests/crud/test_registry.py
git commit -m "feat: zarr.crud skeleton — CrudBackend protocol, registry, config"
```

End every commit body in this plan with:
```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
```

---

### Task 2: `ReferenceBackend` (pure Python)

**Files:**
- Create: `src/zarr/crud/_reference.py`
- Modify: `src/zarr/crud/__init__.py`
- Test: `tests/crud/test_reference_backend.py`

All snippets below are verified against the installed zarr-python.

- [ ] **Step 1: Write the failing test** — `tests/crud/test_reference_backend.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import zarr
from zarr.crud import NodeExistsError, get_backend
from zarr.errors import NodeNotFoundError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    pass

import pytest


def _array_meta() -> dict:
    arr = zarr.create_array(store=MemoryStore(), shape=(8, 8), chunks=(4, 4), dtype="uint16")
    return dict(arr.metadata.to_dict())


async def test_reference_round_trip_chunk() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    meta = _array_meta()
    await be.create_array(store, "a", meta, overwrite=False)
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await be.write_chunk(store, "a", meta, (0, 1), value.tobytes())
    raw = await be.read_chunk(store, "a", meta, (0, 1))
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="uint16").reshape(4, 4), value)


async def test_reference_read_subset_spans_chunks() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    arr = zarr.create_array(store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    data = np.arange(64, dtype="uint16").reshape(8, 8)
    arr[:, :] = data
    meta = dict(arr.metadata.to_dict())
    raw = await be.read_subset(store, "a", meta, (2, 1), (5, 4))
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="uint16").reshape(5, 4), data[2:7, 1:5]
    )


async def test_reference_create_exists_raises() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    meta = _array_meta()
    await be.create_array(store, "a", meta, overwrite=False)
    with pytest.raises(NodeExistsError):
        await be.create_array(store, "a", meta, overwrite=False)


async def test_reference_read_metadata_missing_raises() -> None:
    be = get_backend("reference")
    with pytest.raises(NodeNotFoundError):
        await be.read_metadata(MemoryStore(), "nope")
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/crud/test_reference_backend.py -v`
Expected: FAIL — `KeyError: no CRUD backend registered as 'reference'`

- [ ] **Step 3: Create `src/zarr/crud/_reference.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.core.array import AsyncArray, create_codec_pipeline
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer.core import NDBuffer, default_buffer_prototype
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.group import GroupMetadata
from zarr.core.metadata.io import save_metadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata
from zarr.crud._backend import NodeExistsError
from zarr.errors import NodeNotFoundError
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


def _parse_array_metadata(
    metadata: Mapping[str, JSON],
) -> ArrayV3Metadata | ArrayV2Metadata:
    """Parse a metadata document into a v2 or v3 array metadata object."""
    data = dict(metadata)
    if data.get("zarr_format") == 3:
        return ArrayV3Metadata.from_dict(data)
    return ArrayV2Metadata.from_dict(data)


def _native_dtype(meta_obj: ArrayV3Metadata | ArrayV2Metadata) -> np.dtype[Any]:
    """Numpy dtype in native byte order (zarrs and the facade assume native)."""
    return meta_obj.dtype.to_native_dtype().newbyteorder("=")


def _chunk_shape(meta_obj: ArrayV3Metadata | ArrayV2Metadata) -> tuple[int, ...]:
    if isinstance(meta_obj, ArrayV3Metadata):
        grid = meta_obj.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        return tuple(grid.chunk_shape)
    return tuple(meta_obj.chunks)


def _array_spec(
    meta_obj: ArrayV3Metadata | ArrayV2Metadata, shape: tuple[int, ...]
) -> ArraySpec:
    return ArraySpec(
        shape=shape,
        dtype=meta_obj.dtype,
        fill_value=meta_obj.fill_value,
        config=ArrayConfig.from_dict({}),
        prototype=default_buffer_prototype(),
    )


def _meta_key(path: str, zarr_format: int) -> str:
    fname = ZARR_JSON if zarr_format == 3 else ZARRAY_JSON
    p = path.strip("/")
    return f"{p}/{fname}" if p else fname


class ReferenceBackend:
    """Pure-Python CRUD backend wrapping zarr-python's own machinery.

    Constructs no high-level `Array` for chunk operations (it drives the codec
    pipeline directly); it does reuse `AsyncArray.getitem` for multi-chunk
    subset reads, which is exactly the `BasicIndexer` + codec-pipeline read path.
    """

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        meta_obj = _parse_array_metadata(metadata)
        await self._create(store, path, meta_obj, meta_obj.zarr_format, overwrite=overwrite)

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        meta_obj = GroupMetadata.from_dict(dict(metadata))
        await self._create(store, path, meta_obj, meta_obj.zarr_format, overwrite=overwrite)

    async def _create(
        self, store: Store, path: str, meta_obj: Any, zarr_format: int, *, overwrite: bool
    ) -> None:
        sp = StorePath(store, path.strip("/"))
        proto = default_buffer_prototype()
        if overwrite:
            await store.delete_dir(path.strip("/"))
        else:
            key = _meta_key(path, zarr_format)
            if await store.get(key, prototype=proto) is not None:
                raise NodeExistsError(f"a node already exists at path {path!r}")
        await save_metadata(sp, meta_obj, ensure_parents=True)

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]:
        from zarr.core._json import buffer_to_json_object

        proto = default_buffer_prototype()
        p = path.strip("/")
        sp = StorePath(store, p)
        buf = await (sp / ZARR_JSON).get(prototype=proto)
        if buf is not None:
            return buffer_to_json_object(buf)
        buf2 = await (sp / ZARRAY_JSON).get(prototype=proto)
        if buf2 is not None:
            doc = buffer_to_json_object(buf2)
            zattrs = await (sp / ZATTRS_JSON).get(prototype=proto)
            if zattrs is not None:
                doc["attributes"] = buffer_to_json_object(zattrs)
            return doc
        raise NodeNotFoundError(f"no node found at path {path!r}")

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes:
        meta_obj = _parse_array_metadata(metadata)
        shape = _chunk_shape(meta_obj)
        np_dtype = _native_dtype(meta_obj)
        sp = StorePath(store, path.strip("/"))
        chunk_key = meta_obj.encode_chunk_key(coords)
        buf = await (sp / chunk_key).get(prototype=default_buffer_prototype())
        if buf is None:
            arr = np.full(shape, meta_obj.fill_value, dtype=np_dtype)
        else:
            pipeline = create_codec_pipeline(meta_obj)
            spec = _array_spec(meta_obj, shape)
            decoded = list(await pipeline.decode_batch([(buf, spec)]))
            arr = np.asarray(decoded[0].as_numpy_array(), dtype=np_dtype)
        return np.ascontiguousarray(arr).tobytes()

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        meta_obj = _parse_array_metadata(metadata)
        np_dtype = _native_dtype(meta_obj)
        async_arr = AsyncArray(metadata=meta_obj, store_path=StorePath(store, path.strip("/")))
        selection = tuple(slice(s, s + length) for s, length in zip(start, shape, strict=True))
        result = await async_arr.getitem(selection)
        return np.ascontiguousarray(np.asarray(result, dtype=np_dtype)).tobytes()

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        meta_obj = _parse_array_metadata(metadata)
        shape = _chunk_shape(meta_obj)
        np_dtype = _native_dtype(meta_obj)
        sp = StorePath(store, path.strip("/"))
        chunk_key = meta_obj.encode_chunk_key(coords)
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        pipeline = create_codec_pipeline(meta_obj)
        spec = _array_spec(meta_obj, shape)
        encoded = list(await pipeline.encode_batch([(NDBuffer.from_ndarray_like(arr), spec)]))
        buf = encoded[0]
        if buf is None:
            await (sp / chunk_key).delete()
        else:
            await (sp / chunk_key).set(buf)

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        meta_obj = _parse_array_metadata(metadata)
        sp = StorePath(store, path.strip("/"))
        await (sp / meta_obj.encode_chunk_key(coords)).delete()

    async def delete_node(self, store: Store, path: str) -> None:
        proto = default_buffer_prototype()
        p = path.strip("/")
        sp = StorePath(store, p)
        present = (
            await (sp / ZARR_JSON).get(prototype=proto) is not None
            or await (sp / ZARRAY_JSON).get(prototype=proto) is not None
        )
        if not present:
            raise NodeNotFoundError(f"no node found at path {path!r}")
        await store.delete_dir(p)

    async def list_children(
        self, store: Store, path: str
    ) -> list[tuple[str, dict[str, JSON]]]:
        proto = default_buffer_prototype()
        p = path.strip("/")
        sp = StorePath(store, p)
        if (
            await (sp / ZARR_JSON).get(prototype=proto) is None
            and await (sp / ZARRAY_JSON).get(prototype=proto) is None
        ):
            raise NodeNotFoundError(f"no node found at path {path!r}")
        prefix = f"{p}/" if p else ""
        children: list[tuple[str, dict[str, JSON]]] = []
        async for name in store.list_dir(prefix):
            child_path = f"{p}/{name}" if p else name
            child_sp = StorePath(store, child_path)
            if (
                await (child_sp / ZARR_JSON).get(prototype=proto) is not None
                or await (child_sp / ZARRAY_JSON).get(prototype=proto) is not None
            ):
                children.append((name, await self.read_metadata(store, child_path)))
        return children
```

Notes for the implementer:
- `decode_batch`/`encode_batch` are async and return iterables — wrap in `list(...)`.
- `ArraySpec.dtype` is the `ZDType` object (`meta_obj.dtype`), **not** a numpy dtype.
- `_native_dtype` byte-swaps to native order so both backends return identical
  bytes through the facade (the facade reads them with a native dtype).
- `AsyncArray(metadata=meta_obj, store_path=...)` constructs from an explicit
  document without reading the store.

- [ ] **Step 4: Register the reference backend** — append to `src/zarr/crud/__init__.py` (after the imports, before `__all__`)

```python
from zarr.crud._reference import ReferenceBackend

register_backend("reference", ReferenceBackend())
```

and add `"ReferenceBackend"` to `__all__`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/crud/test_reference_backend.py tests/crud/test_registry.py -v`
Expected: all PASS (the two previously-failing registry tests now pass too).

- [ ] **Step 6: Commit**

```bash
git add src/zarr/crud/_reference.py src/zarr/crud/__init__.py tests/crud/test_reference_backend.py
git commit -m "feat: pure-Python ReferenceBackend for zarr.crud"
```

---

### Task 3: shared facade `zarr.crud._api` + differential suite (reference backend)

**Files:**
- Create: `src/zarr/crud/_api.py`
- Modify: `src/zarr/crud/__init__.py`
- Create: `tests/crud/conftest.py`
- Test: `tests/crud/test_crud.py`

- [ ] **Step 1: Create `tests/crud/conftest.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from zarr.abc.store import Store


def _zarrs_available() -> bool:
    try:
        import _zarrs_bindings  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(
    params=[
        "reference",
        pytest.param(
            "zarrs",
            marks=pytest.mark.skipif(
                not _zarrs_available(), reason="zarrs-bindings is not installed"
            ),
        ),
    ]
)
def backend(request: pytest.FixtureRequest) -> str:
    """A CRUD backend name. The zarrs param is skipped when the extension is absent."""
    import zarr.crud  # noqa: F401  (ensures reference is registered)

    if request.param == "zarrs":
        import zarr.zarrs  # noqa: F401  (registers the zarrs backend)
    return request.param


@pytest.fixture(params=["memory", "local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    if request.param == "memory":
        s: Store = await MemoryStore.open()
    else:
        s = await LocalStore.open(root=tmp_path / "store")
    try:
        yield s
    finally:
        s.close()


def array_metadata(**kwargs: Any) -> dict[str, Any]:
    """An array metadata document built via zarr-python itself."""
    params: dict[str, Any] = {
        "shape": (8, 8),
        "chunks": (4, 4),
        "dtype": "uint16",
        "zarr_format": 3,
    } | kwargs
    arr = zarr.create_array(store=MemoryStore(), **params)
    doc = dict(arr.metadata.to_dict())
    if params["zarr_format"] == 2:
        doc.pop("attributes", None)
    return doc


def filled(store: Store, **kwargs: Any) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
    """Create an 8x8 array 'a', fill it with a ramp, return (data, metadata)."""
    params: dict[str, Any] = {"shape": (8, 8), "chunks": (4, 4), "dtype": "uint16"} | kwargs
    arr = zarr.create_array(store=store, name="a", **params)
    data = np.arange(64, dtype=params["dtype"]).reshape(8, 8)
    arr[:, :] = data
    doc = dict(arr.metadata.to_dict())
    if params.get("zarr_format") == 2:
        doc.pop("attributes", None)
    return data, doc
```

- [ ] **Step 2: Write the failing test** — `tests/crud/test_crud.py`

```python
from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from tests.crud.conftest import array_metadata, filled
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.crud import (
    NodeExistsError,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    delete_chunk,
    delete_node,
    list_children,
    read_chunk,
    read_encoded_chunk,
    read_metadata,
    read_region,
    write_chunk,
)
from zarr.errors import NodeNotFoundError

if TYPE_CHECKING:
    from zarr.abc.store import Store

GROUP_META: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {"answer": 42}}


# --- node lifecycle ---

async def test_create_new_group(backend: str, store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo", backend=backend)
    assert dict(zarr.open_group(store=store, path="foo", mode="r").attrs) == {"answer": 42}


async def test_create_new_group_existing_raises(backend: str, store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo", backend=backend)
    with pytest.raises(NodeExistsError):
        await create_new_group(GROUP_META, store, "foo", backend=backend)


async def test_create_overwrite_group_replaces_array(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="foo", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    await create_overwrite_group(GROUP_META, store, "foo", backend=backend)
    assert dict(zarr.open_group(store=store, path="foo", mode="r").attrs) == {"answer": 42}
    assert not await store.exists("foo/c/0")


async def test_create_new_array(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr", backend=backend)
    a = zarr.open_array(store=store, path="arr", mode="r")
    assert a.shape == (8, 8)
    assert a.dtype == np.dtype("uint16")


async def test_create_new_array_v2(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(zarr_format=2), store, "arr", backend=backend)
    assert zarr.open_array(store=store, path="arr", mode="r").metadata.zarr_format == 2


async def test_create_overwrite_array(backend: str, store: Store) -> None:
    zarr.create_group(store=store, path="arr")
    await create_overwrite_array(array_metadata(), store, "arr", backend=backend)
    assert zarr.open_array(store=store, path="arr", mode="r").shape == (8, 8)


async def test_read_metadata(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr", backend=backend)
    observed = await read_metadata(store, "arr", backend=backend)
    raw = await store.get("arr/zarr.json", prototype=default_buffer_prototype())
    assert raw is not None
    assert observed == json.loads(raw.to_bytes())


async def test_read_metadata_missing(backend: str, store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await read_metadata(store, "nope", backend=backend)


async def test_delete_node(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="doomed", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    await delete_node(store, "doomed", backend=backend)
    assert not await store.exists("doomed/zarr.json")
    assert not await store.exists("doomed/c/0")


async def test_delete_node_missing(backend: str, store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await delete_node(store, "nope", backend=backend)


async def test_list_children(backend: str, store: Store) -> None:
    root = zarr.create_group(store=store)
    root.create_group("sub_group", attributes={"kind": "group"})
    root.create_array("sub_array", shape=(4,), chunks=(2,), dtype="uint8")
    by_path = dict(await list_children(store, "", backend=backend))
    assert set(by_path) == {"sub_group", "sub_array"}
    assert by_path["sub_group"]["node_type"] == "group"
    assert by_path["sub_array"]["node_type"] == "array"
    assert not any(p.startswith("/") for p in by_path)


# --- chunk I/O ---

@pytest.mark.parametrize("dtype", ["uint8", "int32", "float64", "<u2", ">u2"])
async def test_read_chunk_differential(backend: str, store: Store, dtype: str) -> None:
    data, meta = filled(store, dtype=dtype)
    observed = await read_chunk(meta, store, "a", (1, 0), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 0:4])


@pytest.mark.parametrize(
    "compressors", [None, (GzipCodec(),), (ZstdCodec(),), (BloscCodec(cname="lz4"),)]
)
async def test_read_chunk_codecs(backend: str, store: Store, compressors: Any) -> None:
    data, meta = filled(store, compressors=compressors)
    observed = await read_chunk(meta, store, "a", (0, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[0:4, 4:8])


async def test_read_chunk_v2(backend: str, store: Store) -> None:
    data, meta = filled(store, dtype="<u2", zarr_format=2)
    observed = await read_chunk(meta, store, "a", (1, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_read_chunk_sharding(backend: str, store: Store) -> None:
    data, meta = filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await read_chunk(meta, store, "a", (1, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_read_chunk_missing_is_fill(backend: str, store: Store) -> None:
    arr = zarr.create_array(
        store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", fill_value=7
    )
    meta = dict(arr.metadata.to_dict())
    observed = await read_chunk(meta, store, "a", (0, 0), backend=backend)
    np.testing.assert_array_equal(observed, np.full((4, 4), 7, dtype="uint16"))


async def test_read_chunk_metadata_view(backend: str, store: Store) -> None:
    data, meta = filled(store, dtype="uint16", compressors=None)
    view = copy.deepcopy(meta)
    view["data_type"] = "uint8"
    view["shape"] = [8, 16]
    view["chunk_grid"]["configuration"]["chunk_shape"] = [4, 8]
    observed = await read_chunk(view, store, "a", (1, 0), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 0:4].view("uint8"))


async def test_read_chunk_readonly(backend: str, store: Store) -> None:
    _, meta = filled(store)
    observed = await read_chunk(meta, store, "a", (0, 0), backend=backend)
    assert not observed.flags.writeable


async def test_write_chunk_differential(backend: str, store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a", backend=backend)
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await write_chunk(meta, store, "a", (0, 1), value, backend=backend)
    np.testing.assert_array_equal(zarr.open_array(store=store, path="a", mode="r")[0:4, 4:8], value)


async def test_write_chunk_shape_mismatch(backend: str, store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a", backend=backend)
    with pytest.raises(ValueError, match="chunk shape"):
        await write_chunk(meta, store, "a", (0, 0), np.zeros((2, 2), dtype="uint16"), backend=backend)


async def test_delete_chunk(backend: str, store: Store) -> None:
    data, meta = filled(store)
    assert await store.exists("a/c/0/0")
    await delete_chunk(meta, store, "a", (0, 0), backend=backend)
    assert not await store.exists("a/c/0/0")


async def test_read_encoded_chunk_matches_store(backend: str, store: Store) -> None:
    _, meta = filled(store)
    raw = await read_encoded_chunk(meta, store, "a", (0, 0), backend=backend)
    expected = await store.get("a/c/0/0", prototype=default_buffer_prototype())
    assert expected is not None
    assert raw == expected.to_bytes()


async def test_read_encoded_chunk_missing_is_none(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="e", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    meta = dict(arr.metadata.to_dict())
    assert await read_encoded_chunk(meta, store, "e", (0, 0), backend=backend) is None


# --- region I/O ---

SELECTIONS: list[Any] = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(1, 5)),
    (slice(None), 3),
    (5, slice(None)),
    (3, 4),
    (slice(1, 8, 2), slice(None)),
    (slice(None), slice(6, 1, -2)),
    (slice(-3, None), slice(None, -1)),
    ...,
    (..., slice(2, 4)),
    (slice(0, 0), slice(None)),
    (slice(2, 6),),
]


@pytest.mark.parametrize("sel", SELECTIONS)
async def test_read_region_differential(backend: str, store: Store, sel: Any) -> None:
    data, meta = filled(store)
    observed = await read_region(meta, store, "a", sel, backend=backend)
    np.testing.assert_array_equal(observed, data[sel])


async def test_read_region_sharding(backend: str, store: Store) -> None:
    data, meta = filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await read_region(meta, store, "a", (slice(1, 7), slice(3, 8)), backend=backend)
    np.testing.assert_array_equal(observed, data[1:7, 3:8])


async def test_read_region_too_many_indices(backend: str, store: Store) -> None:
    _, meta = filled(store)
    with pytest.raises(IndexError, match="too many indices"):
        await read_region(meta, store, "a", (0, 0, 0), backend=backend)


async def test_read_region_fancy_rejected(backend: str, store: Store) -> None:
    _, meta = filled(store)
    with pytest.raises(TypeError, match="only integers, slices"):
        await read_region(meta, store, "a", ([0, 1], slice(None)), backend=backend)  # type: ignore[arg-type]
```

- [ ] **Step 3: Run it to verify failure**

Run: `uv run pytest tests/crud/test_crud.py -q`
Expected: collection error — `ImportError: cannot import name 'read_chunk' from 'zarr.crud'`

- [ ] **Step 4: Create `src/zarr/crud/_api.py`**

```python
from __future__ import annotations

import operator
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from zarr.core.buffer.core import default_buffer_prototype
from zarr.crud._registry import get_backend

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from zarr.abc.store import Store
    from zarr.core.common import JSON
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata
    from zarr.crud._backend import CrudBackend


@dataclass(frozen=True, slots=True)
class CrudOptions:
    """Options for CRUD operations.

    Currently empty: fields (concurrency limits, checksum validation) arrive in
    a later phase. Accepting it now keeps signatures stable.
    """


BasicIndex = int | slice | types.EllipsisType
BasicSelection = BasicIndex | tuple[BasicIndex, ...]


def _resolve_backend(backend: CrudBackend | str | None) -> CrudBackend:
    if backend is None or isinstance(backend, str):
        return get_backend(backend)
    return backend


def _parse_array_metadata(
    metadata: Mapping[str, JSON],
) -> ArrayV3Metadata | ArrayV2Metadata:
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata

    data = dict(metadata)
    if data.get("zarr_format") == 3:
        return ArrayV3Metadata.from_dict(data)
    return ArrayV2Metadata.from_dict(data)


def _chunk_dtype_and_shape(
    metadata: Mapping[str, JSON],
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Resolve native-byte-order numpy dtype and regular chunk shape.

    Backends decode to (and encode from) the native in-memory representation,
    applying any byte-order codec themselves, so the dtype is coerced to native.
    """
    from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

    meta_obj = _parse_array_metadata(metadata)
    if isinstance(meta_obj, ArrayV3Metadata):
        grid = meta_obj.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        chunk_shape = tuple(grid.chunk_shape)
    else:
        chunk_shape = tuple(meta_obj.chunks)
    return meta_obj.dtype.to_native_dtype().newbyteorder("="), chunk_shape


def _array_shape(metadata: Mapping[str, JSON]) -> tuple[int, ...]:
    shape = metadata.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise TypeError("metadata document has no valid 'shape'")
    result: list[int] = []
    for s in shape:
        if not isinstance(s, (int, float)):
            raise TypeError(f"shape element {s!r} is not a number")
        if isinstance(s, float) and not s.is_integer():
            raise TypeError(f"shape element {s!r} is not an integer")
        result.append(int(s))
    return tuple(result)


def _chunk_key(metadata: Mapping[str, JSON], path: str, coords: tuple[int, ...]) -> str:
    meta_obj = _parse_array_metadata(metadata)
    rel = meta_obj.encode_chunk_key(coords)
    p = path.strip("/")
    return f"{p}/{rel}" if p else rel


def _normalize_selection(
    selection: BasicSelection, shape: tuple[int, ...]
) -> tuple[list[int], list[int], tuple[slice | int, ...]]:
    """Normalize a numpy basic-indexing selection to a step-1 bounding box.

    Returns `(start, bounding_shape, post_index)`: the box to fetch and the
    numpy index to apply to it (strides, reversals, integer-axis removal). Only
    integers, slices, and `Ellipsis` are supported; fancy indexing raises.
    """
    sel_tuple = selection if isinstance(selection, tuple) else (selection,)

    n_ellipsis = sum(1 for s in sel_tuple if s is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        i = sel_tuple.index(Ellipsis)
        n_fill = len(shape) - (len(sel_tuple) - 1)
        if n_fill < 0:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional")
        sel_tuple = sel_tuple[:i] + (slice(None),) * n_fill + sel_tuple[i + 1 :]
    if len(sel_tuple) > len(shape):
        raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional")
    sel_tuple = sel_tuple + (slice(None),) * (len(shape) - len(sel_tuple))

    starts: list[int] = []
    lengths: list[int] = []
    post: list[slice | int] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            n = len(range(start, stop, step))
            if n == 0:
                starts.append(0)
                lengths.append(0)
                post.append(slice(None))
            elif step > 0:
                last = start + (n - 1) * step
                starts.append(start)
                lengths.append(last - start + 1)
                post.append(slice(None, None, step))
            else:
                last = start + (n - 1) * step
                starts.append(last)
                lengths.append(start - last + 1)
                post.append(slice(None, None, step))
        else:
            assert not isinstance(sel, types.EllipsisType), "Ellipsis already expanded above"
            try:
                idx = operator.index(sel)
            except TypeError:
                raise TypeError(
                    "unsupported selection element "
                    f"{sel!r}: only integers, slices, and Ellipsis are supported"
                ) from None
            if idx < 0:
                idx += size
            if not 0 <= idx < size:
                raise IndexError(f"index {sel} is out of bounds for axis {dim} with size {size}")
            starts.append(idx)
            lengths.append(1)
            post.append(0)
    return starts, lengths, tuple(post)


# --- node lifecycle ---

async def create_new_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Create a group from a group metadata document. Raises `NodeExistsError`
    if a node already exists at `path`. Not atomic against concurrent writers."""
    await _resolve_backend(backend).create_group(store, path, metadata, overwrite=False)


async def create_overwrite_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Create a group, deleting any existing node (and children) first. Not
    atomic against concurrent writers."""
    await _resolve_backend(backend).create_group(store, path, metadata, overwrite=True)


async def create_new_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Create an array from a v2 or v3 metadata document. Raises
    `NodeExistsError` if a node already exists. Not atomic against concurrent
    writers."""
    await _resolve_backend(backend).create_array(store, path, metadata, overwrite=False)


async def create_overwrite_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Create an array, deleting any existing node (and children) first. Not
    atomic against concurrent writers."""
    await _resolve_backend(backend).create_array(store, path, metadata, overwrite=True)


async def read_metadata(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> dict[str, JSON]:
    """Read the metadata document of the array or group at `path`. Raises
    `zarr.errors.NodeNotFoundError` if no node exists there."""
    return await _resolve_backend(backend).read_metadata(store, path)


async def delete_node(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Delete the node at `path` and everything under it. Raises
    `zarr.errors.NodeNotFoundError` if absent. `path=""` clears the store."""
    await _resolve_backend(backend).delete_node(store, path)


async def list_children(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> list[tuple[str, dict[str, JSON]]]:
    """List the direct children of the group at `path` as
    `(path, metadata_document)` pairs (store-relative, no leading `/`). Raises
    `zarr.errors.NodeNotFoundError` if no group exists there."""
    return await _resolve_backend(backend).list_children(store, path)


# --- chunk I/O ---

async def read_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode the whole chunk at `chunk_coords`. The metadata document
    is authoritative; missing chunks decode to the fill value. The result is a
    read-only view (`.copy()` for a writable array)."""
    be = _resolve_backend(backend)
    raw = await be.read_chunk(store, path, metadata, tuple(chunk_coords))
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)


async def read_encoded_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> bytes | None:
    """Read the raw, still-encoded bytes of the chunk at `chunk_coords`, or
    `None` if absent. Pure store I/O (`store.get` on the chunk key): the
    `backend` argument is accepted for signature uniformity but unused."""
    key = _chunk_key(metadata, path, tuple(chunk_coords))
    buf = await store.get(key, prototype=default_buffer_prototype())
    return None if buf is None else buf.to_bytes()


async def write_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    value: npt.ArrayLike,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Encode `value` with the codecs in `metadata` and store it as the chunk at
    `chunk_coords`. `value` must match the chunk shape exactly."""
    be = _resolve_backend(backend)
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.shape != chunk_shape:
        raise ValueError(f"value shape {arr.shape} does not match chunk shape {chunk_shape}")
    await be.write_chunk(store, path, metadata, tuple(chunk_coords), arr.tobytes())


async def delete_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> None:
    """Delete the chunk at `chunk_coords`. Deleting a missing chunk is a no-op."""
    await _resolve_backend(backend).delete_chunk(store, path, metadata, tuple(chunk_coords))


# --- region I/O ---

async def read_region(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    selection: BasicSelection,
    *,
    options: CrudOptions | None = None,
    backend: CrudBackend | str | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode a region given by a numpy basic-indexing `selection`
    (integers, slices with steps, `Ellipsis`). One backend call fetches the
    step-1 bounding box; strides/reversals/integer-axis removal are applied as
    numpy views. Missing chunks decode to the fill value. Fancy indexing raises
    `TypeError`. The result is a read-only view.

    Note: a `slice(0, N, step)` reads `O(N)` bytes even though `O(N / step)` are
    returned; for sparse selections over large arrays prefer `read_chunk`."""
    be = _resolve_backend(backend)
    dtype, _ = _chunk_dtype_and_shape(metadata)
    shape = _array_shape(metadata)
    starts, lengths, post_index = _normalize_selection(selection, shape)
    if 0 in lengths:
        block = np.empty(lengths, dtype=dtype)
        block.flags.writeable = False
    else:
        raw = await be.read_subset(store, path, metadata, tuple(starts), tuple(lengths))
        block = np.frombuffer(raw, dtype=dtype).reshape(lengths)
    return cast("np.ndarray[Any, np.dtype[Any]]", block[post_index])
```

Note: `BackendArg` is a documentation alias only; use the literal
`CrudBackend | str | None` annotations as written above.

- [ ] **Step 5: Export the facade from `src/zarr/crud/__init__.py`**

Add to the imports and `__all__` (keep `__all__` sorted):

```python
from zarr.crud._api import (
    CrudOptions,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    delete_chunk,
    delete_node,
    list_children,
    read_chunk,
    read_encoded_chunk,
    read_metadata,
    read_region,
    write_chunk,
)
```

Final `__all__`:

```python
__all__ = [
    "CrudBackend",
    "CrudOptions",
    "NodeExistsError",
    "ReferenceBackend",
    "create_new_array",
    "create_new_group",
    "create_overwrite_array",
    "create_overwrite_group",
    "delete_chunk",
    "delete_node",
    "get_backend",
    "list_children",
    "read_chunk",
    "read_encoded_chunk",
    "read_metadata",
    "read_region",
    "register_backend",
    "write_chunk",
]
```

- [ ] **Step 6: Run the suite against the reference backend**

Run: `uv run pytest tests/crud/test_crud.py -q`
Expected: all PASS. The `backend` fixture's `zarrs` param is skipped (no `--group zarrs`), so every test runs once on `reference` × {memory, local}. If `test_read_chunk_differential[>u2-...]` fails, the byte-order coercion in `_reference._native_dtype` / `_chunk_dtype_and_shape` is wrong — both must end in `.newbyteorder("=")`; do not weaken the assertion.

- [ ] **Step 7: Commit**

```bash
git add src/zarr/crud/_api.py src/zarr/crud/__init__.py tests/crud/conftest.py tests/crud/test_crud.py
git commit -m "feat: zarr.crud shared facade + differential suite (reference backend)"
```

---

### Task 4: `ZarrsBackend` + shrink `zarr.zarrs` + migrate zarrs tests

**Files:**
- Create: `src/zarr/zarrs/_backend.py`
- Modify: `src/zarr/zarrs/__init__.py`
- Delete: `src/zarr/zarrs/_api.py`
- Delete: `tests/zarrs/test_node.py`, `tests/zarrs/test_chunk.py`, `tests/zarrs/test_api.py`
- Modify: `tests/zarrs/test_cache.py`

- [ ] **Step 1: Create `src/zarr/zarrs/_backend.py`**

```python
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import _zarrs_bindings as _zb

from zarr.crud import NodeExistsError
from zarr.errors import NodeNotFoundError
from zarr.zarrs._bridge import resolve_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


def _node_path(path: str) -> str:
    """Convert a zarr path (`""`, `"foo/bar"`) to a zarrs node path (`"/"`,
    `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


@contextmanager
def _translate_errors() -> Iterator[None]:
    try:
        yield
    except _zb.NodeNotFoundError as err:
        raise NodeNotFoundError(str(err)) from err
    except _zb.NodeExistsError as err:
        raise NodeExistsError(str(err)) from err


class ZarrsBackend:
    """CRUD backend backed by the Rust `zarrs` crate via `_zarrs_bindings`.

    Owns the zarrs-specific plumbing: JSON-serializing the metadata document,
    the `/`-prefixed node-path form, store resolution, offloading the blocking
    Rust calls to a worker thread, and translating binding exceptions to the
    canonical `zarr.crud` / `zarr.errors` types.
    """

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        with _translate_errors():
            await asyncio.to_thread(
                _zb.create_array,
                resolve_store(store),
                _node_path(path),
                json.dumps(metadata),
                overwrite,
            )

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        with _translate_errors():
            await asyncio.to_thread(
                _zb.create_group,
                resolve_store(store),
                _node_path(path),
                json.dumps(metadata),
                overwrite,
            )

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]:
        with _translate_errors():
            raw = await asyncio.to_thread(_zb.read_metadata, resolve_store(store), _node_path(path))
        return cast("dict[str, JSON]", json.loads(raw))

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes:
        return await asyncio.to_thread(
            _zb.retrieve_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
        )

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        return await asyncio.to_thread(
            _zb.retrieve_array_subset,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(start),
            list(shape),
        )

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        await asyncio.to_thread(
            _zb.store_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
            data,
        )

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        await asyncio.to_thread(
            _zb.erase_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
        )

    async def delete_node(self, store: Store, path: str) -> None:
        with _translate_errors():
            await asyncio.to_thread(_zb.delete_node, resolve_store(store), _node_path(path))

    async def list_children(
        self, store: Store, path: str
    ) -> list[tuple[str, dict[str, JSON]]]:
        with _translate_errors():
            raw: list[tuple[str, str]] = await asyncio.to_thread(
                _zb.list_children, resolve_store(store), _node_path(path)
            )
        return [
            (child_path.lstrip("/"), cast("dict[str, JSON]", json.loads(doc)))
            for child_path, doc in raw
        ]
```

- [ ] **Step 2: Rewrite `src/zarr/zarrs/__init__.py`**

```python
"""
The zarrs CRUD backend for `zarr.crud`, backed by the Rust
[`zarrs`](https://zarrs.dev) crate.

Importing this module registers the `"zarrs"` backend. Requires the
`zarrs-bindings` extension (in-repo Rust crate; `uv sync --group zarrs`). Select
it with `zarr.config.set({"crud.backend": "zarrs"})` or per call via
`backend="zarrs"`.
"""

try:
    import _zarrs_bindings
except ImportError as e:
    raise ImportError(
        "zarr.zarrs requires the `zarrs-bindings` package, which is not installed. "
        "It is built from the zarr-python repository: run `uv sync --group zarrs`."
    ) from e

from zarr.crud import register_backend
from zarr.zarrs._backend import ZarrsBackend

__version__: str = _zarrs_bindings.version()

register_backend("zarrs", ZarrsBackend())

__all__ = ["ZarrsBackend", "__version__"]
```

- [ ] **Step 3: Delete the moved module and obsolete tests**

```bash
git rm src/zarr/zarrs/_api.py tests/zarrs/test_node.py tests/zarrs/test_chunk.py tests/zarrs/test_api.py
```

- [ ] **Step 4: Update `tests/zarrs/test_cache.py`** — change imports from the old `zarr.zarrs` functions to the `zarr.crud` facade with the zarrs backend.

Replace the import block:

```python
from zarr.zarrs import decode_chunk, encode_chunk
```

with:

```python
from zarr.crud import read_chunk, write_chunk
```

Then in that file replace every `decode_chunk(` call with `read_chunk(` and every `encode_chunk(` call with `write_chunk(`, adding `backend="zarrs"` as the final keyword argument to each so they exercise the cached zarrs path. For example:

```python
    await read_chunk(meta, store, "a", (0, 0), backend="zarrs")
...
    await write_chunk(meta, store, "a", (0, 0), new, backend="zarrs")
```

The cache assertions (`zb.array_cache_len()` / `zb.clear_array_cache()`) and the `import _zarrs_bindings as zb` line are unchanged. The module-level `pytest.importorskip("_zarrs_bindings", ...)` stays.

- [ ] **Step 5: Add the zarrs param coverage — already wired**

`tests/crud/conftest.py` already parametrizes `backend` over `["reference", "zarrs"]` with the zarrs case skipped when the extension is missing. No change needed; running with `--group zarrs` now exercises it.

- [ ] **Step 6: Run everything with the zarrs extension**

Run: `uv run --group zarrs pytest tests/crud tests/zarrs -q`
Expected: all PASS. `tests/crud/test_crud.py` now runs each test on both `reference` and `zarrs` × {memory, local}; `tests/zarrs/test_cache.py` and `test_bridge.py` pass. If a differential test passes on `reference` but fails on `zarrs` (or vice versa), the two backends disagree — investigate the backend, never weaken the assertion.

- [ ] **Step 7: Run without the extension (reference-only path stays green)**

Run: `uv run pytest tests/crud -q`
Expected: all PASS, zarrs params skipped. (`tests/zarrs` is not collectable without the extension; that's fine — its module-level `importorskip` skips it.)

- [ ] **Step 8: Commit**

```bash
git add src/zarr/zarrs tests/zarrs
git commit -m "feat: ZarrsBackend conforms to CrudBackend; zarr.zarrs is now a backend"
```

---

### Task 5: changelog, CI, and final verification

**Files:**
- Modify: `changes/+zarrs-bindings.feature.md`
- Modify: `.github/workflows/zarrs.yml`

- [ ] **Step 1: Reword the changelog fragment** — overwrite `changes/+zarrs-bindings.feature.md`

```markdown
Added `zarr.crud`, an experimental backend-agnostic low-level functional API for
zarr hierarchy CRUD (`create_*`, `read_chunk`, `read_region`, `read_encoded_chunk`,
`write_chunk`, `delete_chunk`, `read_metadata`, `delete_node`, `list_children`).
Array routines take an explicit metadata document, enabling read-only views.
Operations delegate to a pluggable `CrudBackend`: a pure-Python reference backend
(the default) or the zarrs-accelerated backend in `zarr.zarrs`, backed by the Rust
[zarrs](https://zarrs.dev) crate via the in-repo `zarrs-bindings` PyO3 crate.
Select a backend with the `crud.backend` config key or a per-call `backend=`
argument. Build the zarrs backend for development with `uv sync --group zarrs`.
```

- [ ] **Step 2: Update the CI test command** — `.github/workflows/zarrs.yml`

Change the test step's `run:` from:

```yaml
        run: uv run --group zarrs pytest tests/zarrs -v
```

to:

```yaml
        run: uv run --group zarrs pytest tests/crud tests/zarrs -v
```

Validate: `uvx zizmor .github/workflows/zarrs.yml` → no findings.

- [ ] **Step 3: Lint and type-check the new code**

Run: `uv run --group dev ruff format src/zarr/crud src/zarr/zarrs tests/crud tests/zarrs`
Run: `uv run --group dev ruff check --fix src/zarr/crud src/zarr/zarrs tests/crud tests/zarrs`
Run: `uv run --group dev --group zarrs mypy src/zarr/crud src/zarr/zarrs tests/crud tests/zarrs`
Expected: all clean. (mypy is strict; the facade and backends are fully annotated.)

- [ ] **Step 4: Full suites, both with and without the extension**

Run: `uv run --group zarrs pytest tests/crud tests/zarrs -q`  → all pass
Run: `uv run pytest tests/crud -q`  → all pass (zarrs skipped)
Run (regression — the rest of zarr-python is untouched): `uv run pytest tests/test_array.py tests/test_group.py -q`  → pass

- [ ] **Step 5: Commit**

```bash
git add changes/+zarrs-bindings.feature.md .github/workflows/zarrs.yml
git commit -m "docs/ci: zarr.crud changelog and CI coverage"
```

---

## Out of scope (per spec)

- Wiring `zarr.crud` under zarr-python's `Array`/`Group` classes.
- Entrypoint-based backend discovery (registration is explicit/import-time).
- A write-side region operation (`write_region`).
- Renaming the Rust `_zarrs_bindings` pyfunctions (private; adapted by `ZarrsBackend`).
- `CrudOptions` fields (concurrency, checksums) — still a placeholder.
