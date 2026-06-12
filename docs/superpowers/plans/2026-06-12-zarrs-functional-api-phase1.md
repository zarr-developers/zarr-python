# zarrs functional API (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new in-repo PyO3 crate `zarrs-bindings` plus a `zarr.zarrs` subpackage exposing an async functional API (node lifecycle + whole-chunk I/O) that delegates to the Rust `zarrs` crate, working against any zarr-python `Store`.

**Architecture:** Two layers. The Rust crate (`zarrs-bindings/`, maturin/PyO3 abi3-py312, native module `_zarrs_bindings`) is a thin binding over `zarrs` ≈0.23: functions take metadata as JSON strings, a store object, and a node path. The Python subpackage `src/zarr/zarrs/` owns the public API: dict metadata documents, `Store` adaptation (native `LocalStore` fast path + a generic sync-shim callback bridge), numpy conversion, and error translation. Spec: `docs/superpowers/specs/2026-06-11-zarrs-functional-api-design.md`.

**Tech Stack:** Rust 1.91+ (1.96 installed), zarrs 0.23 (default features), pyo3 0.28 (abi3-py312), maturin build backend driven by uv (no maturin CLI needed), pytest with `asyncio_mode = "auto"`.

---

## Environment notes (read first)

- **Python/pytest/mypy always via `uv run`** (user preference).
- **Build/refresh the extension:** `uv sync --group zarrs --reinstall-package zarrs-bindings`. Plain `uv run --group zarrs ...` does NOT reliably rebuild after Rust edits — always re-sync with `--reinstall-package zarrs-bindings` after touching `zarrs-bindings/`.
- **Fast Rust feedback:** `cargo check --manifest-path zarrs-bindings/Cargo.toml` (compiles without packaging a wheel).
- Builds need network access (crates.io for cargo, PyPI for maturin). The Claude Code sandbox on this host fails at bwrap init, so run build commands with the sandbox disabled.
- Pre-commit hooks (ruff format/check, mypy, codespell) run on `git commit`. If a hook modifies files, `git add` the changes and commit again.
- The Rust snippets below were written against verified zarrs 0.23.13 / zarrs_storage 0.4.3 signatures. If `cargo check` reports a mismatch (most likely candidates: the exact signature of `zarrs::node::node_exists`, the re-export path of `store_set_partial_many`, or `TryInto<StorePrefix> for &NodePath`), check https://docs.rs/zarrs/latest — the primitives all exist; only spelling may need adjustment.
- Docstrings use **markdown** (mkdocs), single backticks — not RST.

## File structure

```
zarrs-bindings/                  # new Rust crate (own wheel: zarrs-bindings / _zarrs_bindings)
  Cargo.toml
  pyproject.toml                 # maturin backend
  src/lib.rs                     # pymodule, exceptions, shared error helpers
  src/store.rs                   # PyStore bridge + store resolution
  src/node.rs                    # group/array creation, read_metadata, delete_node, list_children
  src/chunk.rs                   # retrieve/store/erase chunk, retrieve_encoded_chunk
src/zarr/zarrs/                  # new Python subpackage (public API)
  __init__.py                    # import guard + re-exports
  _bridge.py                     # StoreShim (sync adapter over async Store), resolve_store
  _api.py                        # async functional API, numpy/JSON conversion, error translation
tests/zarrs/                     # new test directory (skips when bindings missing)
  __init__.py
  conftest.py                    # store fixtures, array_metadata helper
  test_bridge.py
  test_node.py
  test_chunk.py
.github/workflows/zarrs.yml      # new CI job
pyproject.toml                   # modified: zarrs dependency group, uv source, sdist exclude
.gitignore                       # modified: zarrs-bindings/target/
changes/+zarrs-bindings.feature.md
```

---

### Task 1: Rust crate scaffolding + uv wiring

**Files:**
- Create: `zarrs-bindings/Cargo.toml`
- Create: `zarrs-bindings/pyproject.toml`
- Create: `zarrs-bindings/src/lib.rs`
- Modify: `pyproject.toml` (root)
- Modify: `.gitignore`

- [ ] **Step 1: Create `zarrs-bindings/Cargo.toml`**

```toml
[package]
name = "zarrs-bindings"
version = "0.1.0"
edition = "2024"
rust-version = "1.91"
publish = false

[lib]
name = "_zarrs_bindings"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["abi3-py312"] }
serde_json = "1"
zarrs = "0.23"

[profile.release]
lto = "thin"
```

- [ ] **Step 2: Create `zarrs-bindings/pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "zarrs-bindings"
version = "0.1.0"
description = "PyO3 bindings to the zarrs Rust crate, consumed by zarr.zarrs"
requires-python = ">=3.12"
license = "MIT"

[tool.maturin]
module-name = "_zarrs_bindings"
strip = true
```

- [ ] **Step 3: Create `zarrs-bindings/src/lib.rs`** (exceptions + version only for now)

```rust
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    _zarrs_bindings,
    NodeExistsError,
    PyValueError,
    "A node already exists at the given path."
);
pyo3::create_exception!(
    _zarrs_bindings,
    NodeNotFoundError,
    PyValueError,
    "No node was found at the given path."
);

pub(crate) fn runtime_err(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

pub(crate) fn value_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _zarrs_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NodeExistsError", m.py().get_type::<NodeExistsError>())?;
    m.add("NodeNotFoundError", m.py().get_type::<NodeNotFoundError>())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Wire into the root `pyproject.toml`**

Add to the `[dependency-groups]` table (after the `dev` group):

```toml
zarrs = [
    {include-group = "test"},
    "zarrs-bindings",
]
```

Add a new section at the end of the file:

```toml
[tool.uv.sources]
zarrs-bindings = { path = "zarrs-bindings" }
```

Add `"/zarrs-bindings",` to the `exclude` list under `[tool.hatch.build.targets.sdist]`.

- [ ] **Step 5: Add `zarrs-bindings/target/` to `.gitignore`**

- [ ] **Step 6: Lock, build, smoke-test**

Run: `cargo check --manifest-path zarrs-bindings/Cargo.toml`
Expected: compiles clean (first run downloads ~zarrs dependency tree).

Run: `uv lock && uv sync --group zarrs`
Expected: lockfile updated; `zarrs-bindings` builds via maturin and installs.

Run: `uv run --group zarrs python -c "import _zarrs_bindings as z; print(z.version())"`
Expected: `0.1.0`

- [ ] **Step 7: Commit** (include `zarrs-bindings/Cargo.lock`, which the build created)

```bash
git add zarrs-bindings .gitignore pyproject.toml uv.lock
git commit -m "feat: scaffold zarrs-bindings PyO3 crate"
```

---

### Task 2: `zarr.zarrs` package skeleton + test scaffolding

**Files:**
- Create: `src/zarr/zarrs/__init__.py`
- Create: `tests/zarrs/__init__.py` (empty)
- Create: `tests/zarrs/conftest.py`
- Test: `tests/zarrs/test_api.py`

- [ ] **Step 1: Write the failing test** — `tests/zarrs/test_api.py`

```python
from __future__ import annotations


def test_import() -> None:
    import zarr.zarrs

    assert isinstance(zarr.zarrs.__version__, str)
```

- [ ] **Step 2: Create `tests/zarrs/__init__.py`** (empty file) **and `tests/zarrs/conftest.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("_zarrs_bindings", reason="zarrs-bindings is not installed")

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import Store


@pytest.fixture(params=["memory", "local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    """A writable store: MemoryStore exercises the generic Python-callback bridge,
    LocalStore exercises the native zarrs filesystem store."""
    if request.param == "memory":
        return await MemoryStore.open()
    return await LocalStore.open(root=tmp_path / "store")


def array_metadata(**kwargs: Any) -> dict[str, Any]:
    """Build an array metadata document using zarr-python itself, so the
    documents fed to zarrs always match what zarr-python would write."""
    params: dict[str, Any] = {
        "shape": (8, 8),
        "chunks": (4, 4),
        "dtype": "uint16",
        "zarr_format": 3,
    } | kwargs
    arr = zarr.create_array(store=MemoryStore(), **params)
    doc = dict(arr.metadata.to_dict())
    if params["zarr_format"] == 2:
        # v2 attributes live in .zattrs, not in the .zarray document
        doc.pop("attributes", None)
    return doc
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --group zarrs pytest tests/zarrs -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zarr.zarrs'`

- [ ] **Step 4: Create `src/zarr/zarrs/__init__.py`**

```python
"""
Low-level functional API for zarr hierarchies, backed by the Rust
[`zarrs`](https://zarrs.dev) crate.

This subpackage is experimental. It requires the `zarrs-bindings` package
(in-repo Rust crate; install for development with `uv sync --group zarrs`).

All array routines take an explicit metadata document (a `dict` matching the
`zarr.json` / `.zarray` document) rather than reading metadata from the store,
which makes read-only and virtual views possible.
"""

try:
    import _zarrs_bindings
except ImportError as e:
    raise ImportError(
        "zarr.zarrs requires the `zarrs-bindings` package, which is not installed. "
        "It is built from the zarr-python repository: run `uv sync --group zarrs`."
    ) from e

__version__: str = _zarrs_bindings.version()

__all__ = ["__version__"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run --group zarrs pytest tests/zarrs -v`
Expected: 1 passed. Also verify the skip path works in the default env: `uv run pytest tests/zarrs -v` → all skipped/deselected with "zarrs-bindings is not installed" (the default group lacks the bindings).

- [ ] **Step 6: Commit**

```bash
git add src/zarr/zarrs tests/zarrs
git commit -m "feat: add zarr.zarrs package skeleton and test scaffolding"
```

---

### Task 3: StoreShim — sync bridge over async stores (pure Python, TDD)

**Files:**
- Create: `src/zarr/zarrs/_bridge.py`
- Test: `tests/zarrs/test_bridge.py`

- [ ] **Step 1: Write the failing tests** — `tests/zarrs/test_bridge.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.storage import LocalStore, MemoryStore
from zarr.zarrs._bridge import StoreShim, resolve_store

if TYPE_CHECKING:
    from pathlib import Path


def test_shim_get_set_delete() -> None:
    shim = StoreShim(MemoryStore())
    assert shim.get("a/b") is None
    shim.set("a/b", b"xyz")
    assert shim.get("a/b") == b"xyz"
    assert shim.get_range("a/b", 1, 1) == b"y"
    assert shim.get_range("a/b", 1, None) == b"yz"
    assert shim.get_suffix("a/b", 2) == b"yz"
    assert shim.getsize("a/b") == 3
    assert shim.getsize("missing") is None
    shim.delete("a/b")
    assert shim.get("a/b") is None


def test_shim_listing() -> None:
    shim = StoreShim(MemoryStore())
    shim.set("zarr.json", b"{}")
    shim.set("a/zarr.json", b"{}")
    shim.set("a/c/0/0", b"\x00")
    assert shim.list() == ["a/c/0/0", "a/zarr.json", "zarr.json"]
    assert shim.list_prefix("a/") == ["a/c/0/0", "a/zarr.json"]
    assert shim.list_dir("a/") == (["a/zarr.json"], ["a/c/"])
    assert shim.list_dir("") == (["zarr.json"], ["a/"])
    assert shim.getsize_prefix("a/") == 3
    shim.delete_prefix("a/")
    assert shim.list() == ["zarr.json"]


def test_resolve_store(tmp_path: Path) -> None:
    local = LocalStore(tmp_path)
    assert resolve_store(local) == {"filesystem": str(tmp_path)}
    # read-only LocalStore must go through the shim so writes are rejected in Python
    assert isinstance(resolve_store(LocalStore(tmp_path, read_only=True)), StoreShim)
    assert isinstance(resolve_store(MemoryStore()), StoreShim)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group zarrs pytest tests/zarrs/test_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zarr.zarrs._bridge'`

- [ ] **Step 3: Create `src/zarr/zarrs/_bridge.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import _collect_aiterator, sync
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.abc.store import Store


class StoreShim:
    """
    Synchronous adapter over an async `Store`, called from Rust worker threads.

    Each method blocks the calling thread by submitting a coroutine to the zarr
    event-loop thread (`zarr.core.sync`). Methods must never be called from the
    zarr event-loop thread itself; the Rust bindings only call them from
    `asyncio.to_thread` worker threads.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._prototype = default_buffer_prototype()

    def get(self, key: str) -> bytes | None:
        buf = sync(self._store.get(key, prototype=self._prototype))
        return None if buf is None else buf.to_bytes()

    def get_range(self, key: str, offset: int, length: int | None) -> bytes | None:
        byte_range = (
            RangeByteRequest(offset, offset + length)
            if length is not None
            else OffsetByteRequest(offset)
        )
        buf = sync(self._store.get(key, prototype=self._prototype, byte_range=byte_range))
        return None if buf is None else buf.to_bytes()

    def get_suffix(self, key: str, suffix: int) -> bytes | None:
        buf = sync(
            self._store.get(key, prototype=self._prototype, byte_range=SuffixByteRequest(suffix))
        )
        return None if buf is None else buf.to_bytes()

    def set(self, key: str, value: bytes) -> None:
        sync(self._store.set(key, self._prototype.buffer.from_bytes(value)))

    def delete(self, key: str) -> None:
        sync(self._store.delete(key))

    def delete_prefix(self, prefix: str) -> None:
        sync(self._store.delete_dir(prefix.rstrip("/")))

    def getsize(self, key: str) -> int | None:
        try:
            return sync(self._store.getsize(key))
        except FileNotFoundError:
            return None

    def getsize_prefix(self, prefix: str) -> int:
        return sync(self._store.getsize_prefix(prefix.rstrip("/")))

    def list(self) -> list[str]:
        return sorted(sync(_collect_aiterator(self._store.list())))

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(sync(_collect_aiterator(self._store.list_prefix(prefix))))

    def list_dir(self, prefix: str) -> tuple[list[str], list[str]]:
        """Return `(keys, prefixes)` directly under `prefix`, as zarrs expects:
        full keys, and child prefixes ending in `/`."""
        stripped = prefix.rstrip("/")
        children = sorted(sync(_collect_aiterator(self._store.list_dir(stripped))))
        keys: list[str] = []
        prefixes: list[str] = []
        for child in children:
            full = f"{stripped}/{child}" if stripped else child
            if sync(self._store.exists(full)):
                keys.append(full)
            else:
                prefixes.append(full + "/")
        return keys, prefixes


def resolve_store(store: Store) -> StoreShim | dict[str, str]:
    """
    Convert a zarr `Store` into the representation `_zarrs_bindings` expects:
    a config dict for stores with a native Rust implementation, otherwise a
    `StoreShim` that Rust calls back into.
    """
    if isinstance(store, LocalStore) and not store.read_only:
        return {"filesystem": str(store.root)}
    return StoreShim(store)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group zarrs pytest tests/zarrs/test_bridge.py -v`
Expected: 3 passed. (If `list_dir`/`delete_dir`/`getsize_prefix` choke on the stripped prefix, check the `Store` ABC docstrings in `src/zarr/abc/store.py:348-501` — these methods take prefixes without trailing slashes.)

- [ ] **Step 5: Commit**

```bash
git add src/zarr/zarrs/_bridge.py tests/zarrs/test_bridge.py
git commit -m "feat: sync store bridge for zarrs bindings"
```

---

### Task 4: Rust store bridge + group creation, end to end

**Files:**
- Create: `zarrs-bindings/src/store.rs`
- Create: `zarrs-bindings/src/node.rs`
- Modify: `zarrs-bindings/src/lib.rs`
- Create: `src/zarr/zarrs/_api.py`
- Modify: `src/zarr/zarrs/__init__.py`
- Test: `tests/zarrs/test_node.py`

- [ ] **Step 1: Write the failing tests** — `tests/zarrs/test_node.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import zarr
from zarr.core.buffer.core import default_buffer_prototype
from zarr.zarrs import NodeExistsError, create_new_group, create_overwrite_group

if TYPE_CHECKING:
    from zarr.abc.store import Store

GROUP_META: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {"answer": 42},
}


async def test_create_new_group(store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo")
    group = zarr.open_group(store=store, path="foo", mode="r")
    assert dict(group.attrs) == {"answer": 42}


async def test_create_new_group_at_root(store: Store) -> None:
    await create_new_group(GROUP_META, store, "")
    group = zarr.open_group(store=store, mode="r")
    assert dict(group.attrs) == {"answer": 42}


async def test_create_new_group_existing_node(store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo")
    with pytest.raises(NodeExistsError):
        await create_new_group(GROUP_META, store, "foo")


async def test_create_overwrite_group(store: Store) -> None:
    # an array and its chunks previously occupied the path; overwrite removes both
    arr = zarr.create_array(store=store, name="foo", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    assert await store.exists("foo/c/0")
    await create_overwrite_group(GROUP_META, store, "foo")
    group = zarr.open_group(store=store, path="foo", mode="r")
    assert dict(group.attrs) == {"answer": 42}
    assert not await store.exists("foo/c/0")
    assert await store.get("foo/zarr.json", prototype=default_buffer_prototype()) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group zarrs pytest tests/zarrs/test_node.py -v`
Expected: FAIL with `ImportError: cannot import name 'NodeExistsError' from 'zarr.zarrs'`

- [ ] **Step 3: Create `zarrs-bindings/src/store.rs`**

```rust
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{
    Bytes, ByteRange, ByteRangeIterator, ListableStorageTraits, MaybeBytes, MaybeBytesIterator,
    OffsetBytesIterator, ReadableStorageTraits, ReadableWritableListableStorage, StorageError,
    StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

/// A zarrs store backed by a Python `zarr.zarrs._bridge.StoreShim`.
///
/// Every method attaches to the Python interpreter and calls the shim, which
/// blocks on the zarr event loop. Blocking waits in Python release the GIL, so
/// the loop thread can make progress while a Rust worker waits here.
pub(crate) struct PyStore(Py<PyAny>);

fn py_err(err: PyErr) -> StorageError {
    StorageError::Other(err.to_string())
}

fn invalid(err: impl std::fmt::Display) -> StorageError {
    StorageError::Other(err.to_string())
}

impl PyStore {
    fn get_with_range(
        &self,
        key: &StoreKey,
        range: Option<&ByteRange>,
    ) -> Result<MaybeBytes, StorageError> {
        Python::attach(|py| {
            let shim = self.0.bind(py);
            let result = match range {
                None => shim.call_method1("get", (key.as_str(),)),
                Some(ByteRange::FromStart(offset, length)) => {
                    shim.call_method1("get_range", (key.as_str(), *offset, *length))
                }
                Some(ByteRange::Suffix(suffix)) => {
                    shim.call_method1("get_suffix", (key.as_str(), *suffix))
                }
            }
            .map_err(py_err)?;
            if result.is_none() {
                Ok(None)
            } else {
                let bytes: Vec<u8> = result.extract().map_err(py_err)?;
                Ok(Some(Bytes::from(bytes)))
            }
        })
    }
}

impl ReadableStorageTraits for PyStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.get_with_range(key, None)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let mut out = Vec::new();
        for byte_range in byte_ranges {
            match self.get_with_range(key, Some(&byte_range))? {
                Some(bytes) => out.push(Ok(bytes)),
                None => return Ok(None),
            }
        }
        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("getsize", (key.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)
        })
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for PyStore {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        Python::attach(|py| {
            let data = PyBytes::new(py, &value);
            self.0
                .bind(py)
                .call_method1("set", (key.as_str(), data))
                .map_err(py_err)?;
            Ok(())
        })
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        // read-modify-write fallback provided by zarrs
        zarrs::storage::store_set_partial_many(self, key, offset_values)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("delete", (key.as_str(),))
                .map_err(py_err)?;
            Ok(())
        })
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("delete_prefix", (prefix.as_str(),))
                .map_err(py_err)?;
            Ok(())
        })
    }
}

impl ListableStorageTraits for PyStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Python::attach(|py| {
            let keys: Vec<String> = self
                .0
                .bind(py)
                .call_method0("list")
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            keys.into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect()
        })
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Python::attach(|py| {
            let keys: Vec<String> = self
                .0
                .bind(py)
                .call_method1("list_prefix", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            keys.into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect()
        })
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        Python::attach(|py| {
            let (keys, prefixes): (Vec<String>, Vec<String>) = self
                .0
                .bind(py)
                .call_method1("list_dir", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            let keys = keys
                .into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect::<Result<Vec<_>, StorageError>>()?;
            let prefixes = prefixes
                .into_iter()
                .map(|p| StorePrefix::new(p).map_err(invalid))
                .collect::<Result<Vec<_>, StorageError>>()?;
            Ok(StoreKeysPrefixes::new(keys, prefixes))
        })
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("getsize_prefix", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)
        })
    }
}

/// Convert the Python-side store representation (`zarr.zarrs._bridge.resolve_store`
/// output) into a zarrs storage handle.
pub(crate) fn resolve_store(obj: &Bound<'_, PyAny>) -> PyResult<ReadableWritableListableStorage> {
    if let Ok(config) = obj.downcast::<PyDict>() {
        if let Some(root) = config.get_item("filesystem")? {
            let root: String = root.extract()?;
            let store =
                FilesystemStore::new(root).map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(Arc::new(store));
        }
        return Err(PyValueError::new_err("unrecognized store configuration"));
    }
    Ok(Arc::new(PyStore(obj.clone().unbind())))
}
```

- [ ] **Step 4: Create `zarrs-bindings/src/node.rs`** (group functions only; later tasks extend this file)

```rust
use pyo3::prelude::*;
use zarrs::group::Group;
use zarrs::metadata::GroupMetadata;
use zarrs::node::{node_exists, NodePath};
use zarrs::storage::{ReadableWritableListableStorage, StorePrefix};

use crate::store::resolve_store;
use crate::{runtime_err, value_err, NodeExistsError};

pub(crate) fn parse_node_path(path: &str) -> PyResult<NodePath> {
    NodePath::new(path).map_err(value_err)
}

/// When a node exists at `node_path`: erase it (and everything under it) if
/// `overwrite`, otherwise raise `NodeExistsError`.
pub(crate) fn prepare_target(
    storage: &ReadableWritableListableStorage,
    node_path: &NodePath,
    overwrite: bool,
) -> PyResult<()> {
    if node_exists(storage, node_path).map_err(runtime_err)? {
        if !overwrite {
            return Err(NodeExistsError::new_err(format!(
                "a node already exists at path {}",
                node_path.as_str()
            )));
        }
        let prefix: StorePrefix = node_path.try_into().map_err(value_err)?;
        storage.erase_prefix(&prefix).map_err(runtime_err)?;
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn create_group(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    overwrite: bool,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    let metadata = GroupMetadata::try_from(metadata_json.as_str()).map_err(value_err)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        prepare_target(&storage, &node_path, overwrite)?;
        let group = Group::new_with_metadata(storage, &path, metadata).map_err(value_err)?;
        group.store_metadata().map_err(runtime_err)
    })
}
```

- [ ] **Step 5: Register in `zarrs-bindings/src/lib.rs`**

Add after the `use` lines:

```rust
mod node;
mod store;
```

Add to the `#[pymodule]` body before `Ok(())`:

```rust
    m.add_function(wrap_pyfunction!(node::create_group, m)?)?;
```

- [ ] **Step 6: Compile**

Run: `cargo check --manifest-path zarrs-bindings/Cargo.toml`
Expected: success. If `node_exists` or `try_into::<StorePrefix>()` signatures mismatch, fix per https://docs.rs/zarrs/latest/zarrs/node/ (the helpers exist; argument form may differ, e.g. `node_exists(&storage, &node_path)` vs a `&Arc` receiver).

- [ ] **Step 7: Create `src/zarr/zarrs/_api.py`**

```python
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import _zarrs_bindings as _zb

from zarr.errors import NodeNotFoundError
from zarr.zarrs._bridge import resolve_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr.abc.store import Store
    from zarr.core.common import JSON

NodeExistsError = _zb.NodeExistsError
"""Raised by `create_new_*` when a node already exists at the target path."""


@dataclass(frozen=True, slots=True)
class ZarrsOptions:
    """Options for zarrs-backed operations.

    Currently empty: fields (concurrency limits, checksum validation) arrive in
    a later phase. Accepting it now keeps signatures stable.
    """


def _node_path(path: str) -> str:
    """Convert a zarr-python node path (`""`, `"foo/bar"`) to a zarrs node path
    (`"/"`, `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


@contextmanager
def _translate_errors() -> Iterator[None]:
    try:
        yield
    except _zb.NodeNotFoundError as err:
        raise NodeNotFoundError(str(err)) from err


async def create_new_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create a group at `path` from a group metadata document.

    Raises `NodeExistsError` if any node already exists at `path`.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_group, resolve_store(store), _node_path(path), json.dumps(metadata), False
        )


async def create_overwrite_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create a group at `path`, deleting any existing node (and its children) first."""
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_group, resolve_store(store), _node_path(path), json.dumps(metadata), True
        )
```

- [ ] **Step 8: Re-export from `src/zarr/zarrs/__init__.py`**

Replace the `__version__`/`__all__` lines at the end with:

```python
__version__: str = _zarrs_bindings.version()

from zarr.zarrs._api import (
    NodeExistsError,
    ZarrsOptions,
    create_new_group,
    create_overwrite_group,
)

__all__ = [
    "NodeExistsError",
    "ZarrsOptions",
    "__version__",
    "create_new_group",
    "create_overwrite_group",
]
```

- [ ] **Step 9: Rebuild and run the tests**

Run: `uv sync --group zarrs --reinstall-package zarrs-bindings`
Run: `uv run --group zarrs pytest tests/zarrs/test_node.py -v`
Expected: 8 passed (4 tests × 2 store params). The MemoryStore param proves the full Rust→Python callback bridge; LocalStore proves the native path.

- [ ] **Step 10: Commit**

```bash
git add zarrs-bindings/src src/zarr/zarrs tests/zarrs/test_node.py
git commit -m "feat: zarrs store bridge and group creation"
```

---

### Task 5: Array creation + read_metadata

**Files:**
- Modify: `zarrs-bindings/src/node.rs`
- Modify: `zarrs-bindings/src/lib.rs`
- Modify: `src/zarr/zarrs/_api.py`, `src/zarr/zarrs/__init__.py`
- Test: `tests/zarrs/test_node.py`

- [ ] **Step 1: Add failing tests to `tests/zarrs/test_node.py`**

Extend the imports:

```python
import json

import numpy as np

from tests.zarrs.conftest import array_metadata
from zarr.errors import NodeNotFoundError
from zarr.zarrs import create_new_array, create_overwrite_array, read_metadata
```

(If `from tests.zarrs.conftest import ...` fails at collection, use a relative import `from .conftest import array_metadata` — `tests` is a package.)

Add tests:

```python
async def test_create_new_array(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    arr = zarr.open_array(store=store, path="arr", mode="r")
    assert arr.shape == (8, 8)
    assert arr.chunks == (4, 4)
    assert arr.dtype == np.dtype("uint16")


async def test_create_new_array_existing_node(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    with pytest.raises(NodeExistsError):
        await create_new_array(array_metadata(), store, "arr")


async def test_create_overwrite_array(store: Store) -> None:
    zarr.create_group(store=store, path="arr")
    await create_overwrite_array(array_metadata(), store, "arr")
    arr = zarr.open_array(store=store, path="arr", mode="r")
    assert arr.shape == (8, 8)


async def test_read_metadata_matches_stored_document(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    observed = await read_metadata(store, "arr")
    raw = await store.get("arr/zarr.json", prototype=default_buffer_prototype())
    assert raw is not None
    assert observed == json.loads(raw.to_bytes())


async def test_read_metadata_zarr_python_group(store: Store) -> None:
    zarr.create_group(store=store, path="g", attributes={"a": 1})
    observed = await read_metadata(store, "g")
    assert observed["node_type"] == "group"
    assert observed["attributes"] == {"a": 1}


async def test_read_metadata_missing(store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await read_metadata(store, "nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --group zarrs pytest tests/zarrs/test_node.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_new_array'`

- [ ] **Step 3: Add Rust functions to `zarrs-bindings/src/node.rs`**

Extend the `use` block:

```rust
use zarrs::array::Array;
use zarrs::metadata::ArrayMetadata;
use zarrs::node::Node;

use crate::NodeNotFoundError;
```

Append:

```rust
#[pyfunction]
pub(crate) fn create_array(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    overwrite: bool,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    let metadata = ArrayMetadata::try_from(metadata_json.as_str()).map_err(value_err)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        prepare_target(&storage, &node_path, overwrite)?;
        let array = Array::new_with_metadata(storage, &path, metadata).map_err(value_err)?;
        array.store_metadata().map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn read_metadata(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<String> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let node = Node::open(&storage, &path)
            .map_err(|e| NodeNotFoundError::new_err(e.to_string()))?;
        serde_json::to_string(node.metadata()).map_err(runtime_err)
    })
}
```

Register both in `lib.rs`:

```rust
    m.add_function(wrap_pyfunction!(node::create_array, m)?)?;
    m.add_function(wrap_pyfunction!(node::read_metadata, m)?)?;
```

- [ ] **Step 4: Add Python wrappers to `src/zarr/zarrs/_api.py`**

```python
async def create_new_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create an array at `path` from a v2 or v3 array metadata document.

    Raises `NodeExistsError` if any node already exists at `path`.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_array, resolve_store(store), _node_path(path), json.dumps(metadata), False
        )


async def create_overwrite_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create an array at `path`, deleting any existing node (and its children) first."""
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_array, resolve_store(store), _node_path(path), json.dumps(metadata), True
        )


async def read_metadata(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> dict[str, JSON]:
    """Read the metadata document of the array or group at `path`.

    Raises `zarr.errors.NodeNotFoundError` if no node exists there.
    """
    with _translate_errors():
        raw = await asyncio.to_thread(_zb.read_metadata, resolve_store(store), _node_path(path))
    result: dict[str, JSON] = json.loads(raw)
    return result
```

Add `create_new_array`, `create_overwrite_array`, `read_metadata` to the `__init__.py` import and `__all__`.

- [ ] **Step 5: Rebuild and test**

Run: `cargo check --manifest-path zarrs-bindings/Cargo.toml` → success
Run: `uv sync --group zarrs --reinstall-package zarrs-bindings`
Run: `uv run --group zarrs pytest tests/zarrs/test_node.py -v`
Expected: all pass (20 = 10 tests × 2 stores). Note: `test_read_metadata_matches_stored_document` asserts zarrs round-trips the document zarrs itself wrote; if zarrs normalizes a field zarr-python emits differently (e.g. drops a `null` `dimension_names`), adjust the *fixture* (`array_metadata`) to drop the field, not the assertion.

- [ ] **Step 6: Commit**

```bash
git add zarrs-bindings/src src/zarr/zarrs tests/zarrs
git commit -m "feat: zarrs-backed array creation and metadata reads"
```

---

### Task 6: delete_node + list_children

**Files:**
- Modify: `zarrs-bindings/src/node.rs`, `zarrs-bindings/src/lib.rs`
- Modify: `src/zarr/zarrs/_api.py`, `src/zarr/zarrs/__init__.py`
- Test: `tests/zarrs/test_node.py`

- [ ] **Step 1: Add failing tests to `tests/zarrs/test_node.py`**

```python
from zarr.zarrs import delete_node, list_children


async def test_delete_node(store: Store) -> None:
    arr = zarr.create_array(store=store, name="doomed", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    await delete_node(store, "doomed")
    assert not await store.exists("doomed/zarr.json")
    assert not await store.exists("doomed/c/0")


async def test_delete_node_missing(store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await delete_node(store, "nope")


async def test_list_children(store: Store) -> None:
    root = zarr.create_group(store=store)
    root.create_group("sub_group", attributes={"kind": "group"})
    root.create_array("sub_array", shape=(4,), chunks=(2,), dtype="uint8")
    children = await list_children(store, "")
    by_path = dict(children)
    assert set(by_path) == {"sub_group", "sub_array"}
    assert by_path["sub_group"]["node_type"] == "group"
    assert by_path["sub_array"]["node_type"] == "array"


async def test_list_children_missing(store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await list_children(store, "nope")
```

- [ ] **Step 2: Run to verify failure** — `uv run --group zarrs pytest tests/zarrs/test_node.py -v` → ImportError.

- [ ] **Step 3: Add Rust functions to `node.rs`**

```rust
#[pyfunction]
pub(crate) fn delete_node(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        if !node_exists(&storage, &node_path).map_err(runtime_err)? {
            return Err(NodeNotFoundError::new_err(format!(
                "no node found at path {}",
                node_path.as_str()
            )));
        }
        let prefix: StorePrefix = (&node_path).try_into().map_err(value_err)?;
        storage.erase_prefix(&prefix).map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn list_children(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<Vec<(String, String)>> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let group = Group::open(storage, &path)
            .map_err(|e| NodeNotFoundError::new_err(e.to_string()))?;
        let children = group.children(false).map_err(runtime_err)?;
        children
            .into_iter()
            .map(|node| {
                let metadata = serde_json::to_string(node.metadata()).map_err(runtime_err)?;
                Ok((node.path().as_str().to_string(), metadata))
            })
            .collect()
    })
}
```

Register both in `lib.rs` as before.

- [ ] **Step 4: Add Python wrappers to `_api.py`**

```python
async def delete_node(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Delete the node at `path`, including all keys and child nodes under it.

    Raises `zarr.errors.NodeNotFoundError` if no node exists there. Deleting the
    root node (`path=""`) clears the entire store.
    """
    with _translate_errors():
        await asyncio.to_thread(_zb.delete_node, resolve_store(store), _node_path(path))


async def list_children(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> list[tuple[str, dict[str, JSON]]]:
    """List the direct children of the group at `path` as
    `(path, metadata_document)` pairs. Paths are store-relative (no leading `/`).

    Raises `zarr.errors.NodeNotFoundError` if no group exists at `path`.
    """
    with _translate_errors():
        raw = await asyncio.to_thread(_zb.list_children, resolve_store(store), _node_path(path))
    return [(child_path.lstrip("/"), json.loads(doc)) for child_path, doc in raw]
```

Export both from `__init__.py`.

- [ ] **Step 5: Rebuild and test**

Run: `uv sync --group zarrs --reinstall-package zarrs-bindings && uv run --group zarrs pytest tests/zarrs/test_node.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add zarrs-bindings/src src/zarr/zarrs tests/zarrs
git commit -m "feat: zarrs-backed node deletion and child listing"
```

---

### Task 7: Whole-chunk I/O (decode/encode/raw/erase)

**Files:**
- Create: `zarrs-bindings/src/chunk.rs`
- Modify: `zarrs-bindings/src/lib.rs`
- Modify: `src/zarr/zarrs/_api.py`, `src/zarr/zarrs/__init__.py`
- Test: `tests/zarrs/test_chunk.py`

- [ ] **Step 1: Write the failing tests** — `tests/zarrs/test_chunk.py`

```python
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from tests.zarrs.conftest import array_metadata
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.zarrs import (
    create_new_array,
    decode_chunk,
    encode_chunk,
    erase_chunk,
    read_encoded_chunk,
)

if TYPE_CHECKING:
    from zarr.abc.store import Store


def _filled(
    store: Store, **kwargs: Any
) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
    """Create an 8x8 array named 'a' via zarr-python, fill it with a ramp, and
    return (data, metadata_document)."""
    params: dict[str, Any] = {"shape": (8, 8), "chunks": (4, 4), "dtype": "uint16"} | kwargs
    arr = zarr.create_array(store=store, name="a", **params)
    data = np.arange(64, dtype=params["dtype"]).reshape(8, 8)
    arr[:, :] = data
    doc = dict(arr.metadata.to_dict())
    if params.get("zarr_format") == 2:
        # v2 attributes live in .zattrs, not in the .zarray document
        doc.pop("attributes", None)
    return data, doc


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float64"])
async def test_decode_chunk_differential(store: Store, dtype: str) -> None:
    data, meta = _filled(store, dtype=dtype)
    observed = await decode_chunk(meta, store, "a", (1, 0))
    np.testing.assert_array_equal(observed, data[4:8, 0:4])


@pytest.mark.parametrize(
    "compressors", [None, (GzipCodec(),), (ZstdCodec(),), (BloscCodec(cname="lz4"),)]
)
async def test_decode_chunk_codecs(store: Store, compressors: Any) -> None:
    data, meta = _filled(store, compressors=compressors)
    observed = await decode_chunk(meta, store, "a", (0, 1))
    np.testing.assert_array_equal(observed, data[0:4, 4:8])


async def test_decode_chunk_v2(store: Store) -> None:
    data, meta = _filled(store, zarr_format=2)
    observed = await decode_chunk(meta, store, "a", (1, 1))
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_decode_chunk_sharding(store: Store) -> None:
    # with sharding, the metadata chunk grid is the shard grid
    data, meta = _filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await decode_chunk(meta, store, "a", (1, 1))
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_decode_chunk_missing_returns_fill_value(store: Store) -> None:
    arr = zarr.create_array(
        store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", fill_value=7
    )
    meta = dict(arr.metadata.to_dict())
    observed = await decode_chunk(meta, store, "a", (0, 0))
    np.testing.assert_array_equal(observed, np.full((4, 4), 7, dtype="uint16"))


async def test_decode_chunk_selection_not_implemented(store: Store) -> None:
    _, meta = _filled(store)
    with pytest.raises(NotImplementedError):
        await decode_chunk(meta, store, "a", (0, 0), selection=(slice(0, 2), slice(0, 2)))


async def test_decode_chunk_metadata_view(store: Store) -> None:
    # the read-only-view case: decode with a metadata document the store never saw
    data, meta = _filled(store, dtype="uint16", compressors=None)
    view = copy.deepcopy(meta)
    view["data_type"] = "uint8"
    view["shape"] = [8, 16]
    view["chunk_grid"]["configuration"]["chunk_shape"] = [4, 8]
    observed = await decode_chunk(view, store, "a", (1, 0))
    np.testing.assert_array_equal(observed, data[4:8, 0:4].view("uint8"))


async def test_encode_chunk_differential(store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a")
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await encode_chunk(meta, store, "a", (0, 1), value)
    arr = zarr.open_array(store=store, path="a", mode="r")
    np.testing.assert_array_equal(arr[0:4, 4:8], value)


async def test_encode_chunk_shape_mismatch(store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a")
    with pytest.raises(ValueError, match="chunk shape"):
        await encode_chunk(meta, store, "a", (0, 0), np.zeros((2, 2), dtype="uint16"))


async def test_read_encoded_chunk_matches_store(store: Store) -> None:
    _, meta = _filled(store)
    raw = await read_encoded_chunk(meta, store, "a", (0, 0))
    expected = await store.get("a/c/0/0", prototype=default_buffer_prototype())
    assert expected is not None
    assert raw == expected.to_bytes()


async def test_read_encoded_chunk_missing_returns_none(store: Store) -> None:
    arr = zarr.create_array(store=store, name="empty", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    meta = dict(arr.metadata.to_dict())
    assert await read_encoded_chunk(meta, store, "empty", (0, 0)) is None


async def test_erase_chunk(store: Store) -> None:
    data, meta = _filled(store)
    assert await store.exists("a/c/0/0")
    await erase_chunk(meta, store, "a", (0, 0))
    assert not await store.exists("a/c/0/0")
    arr = zarr.open_array(store=store, path="a", mode="r")
    np.testing.assert_array_equal(arr[0:4, 0:4], np.zeros((4, 4), dtype="uint16"))
```

- [ ] **Step 2: Run to verify failure** — `uv run --group zarrs pytest tests/zarrs/test_chunk.py -v` → ImportError.

- [ ] **Step 3: Create `zarrs-bindings/src/chunk.rs`**

```rust
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zarrs::array::{Array, ArrayBytes};
use zarrs::metadata::ArrayMetadata;
use zarrs::storage::ReadableWritableListableStorage;

use crate::store::resolve_store;
use crate::{runtime_err, value_err};

type DynArray = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Construct an Array view from an explicit metadata document, without
/// consulting the store for metadata.
fn array_view(
    storage: ReadableWritableListableStorage,
    path: &str,
    metadata_json: &str,
) -> PyResult<DynArray> {
    let metadata = ArrayMetadata::try_from(metadata_json).map_err(value_err)?;
    Array::new_with_metadata(storage, path, metadata).map_err(value_err)
}

#[pyfunction]
pub(crate) fn retrieve_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<Py<PyBytes>> {
    let storage = resolve_store(store)?;
    let data = py.detach(move || -> PyResult<Vec<u8>> {
        let array = array_view(storage, &path, &metadata_json)?;
        let bytes: ArrayBytes<'static> =
            array.retrieve_chunk(&chunk_coords).map_err(runtime_err)?;
        let fixed = bytes.into_fixed().map_err(|_| {
            PyNotImplementedError::new_err("variable-length data types are not supported")
        })?;
        Ok(fixed.into_owned())
    })?;
    Ok(PyBytes::new(py, &data).unbind())
}

#[pyfunction]
pub(crate) fn retrieve_encoded_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<Option<Py<PyBytes>>> {
    let storage = resolve_store(store)?;
    let data = py.detach(move || -> PyResult<Option<Vec<u8>>> {
        let array = array_view(storage, &path, &metadata_json)?;
        array
            .retrieve_encoded_chunk(&chunk_coords)
            .map_err(runtime_err)
    })?;
    Ok(data.map(|d| PyBytes::new(py, &d).unbind()))
}

#[pyfunction]
pub(crate) fn store_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
    data: Vec<u8>,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let array = array_view(storage, &path, &metadata_json)?;
        array
            .store_chunk(&chunk_coords, ArrayBytes::new_flen(data))
            .map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn erase_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let array = array_view(storage, &path, &metadata_json)?;
        array.erase_chunk(&chunk_coords).map_err(runtime_err)
    })
}
```

Register in `lib.rs`: add `mod chunk;` and

```rust
    m.add_function(wrap_pyfunction!(chunk::retrieve_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(chunk::retrieve_encoded_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(chunk::store_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(chunk::erase_chunk, m)?)?;
```

- [ ] **Step 4: Add Python wrappers to `_api.py`**

Extend imports:

```python
from typing import Any

import numpy as np
import numpy.typing as npt
```

Add:

```python
def _chunk_dtype_and_shape(
    metadata: Mapping[str, JSON],
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Resolve the numpy dtype and chunk shape from a metadata document, using
    zarr-python's own metadata parsing."""
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

    if metadata.get("zarr_format") == 3:
        meta3 = ArrayV3Metadata.from_dict(dict(metadata))
        grid = meta3.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        return meta3.data_type.to_native_dtype(), grid.chunk_shape
    meta2 = ArrayV2Metadata.from_dict(dict(metadata))
    return meta2.dtype.to_native_dtype(), meta2.chunks


async def decode_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    selection: tuple[slice | int, ...] | None = None,
    options: ZarrsOptions | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode the chunk at `chunk_coords` of the array described by
    `metadata`, located at `path` in `store`.

    The metadata document is authoritative: it is not read from the store.
    Missing chunks decode to the fill value. `selection` (a chunk-relative
    subset) is not implemented yet.
    """
    if selection is not None:
        raise NotImplementedError("chunk subset selection is not implemented yet")
    raw = await asyncio.to_thread(
        _zb.retrieve_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)


async def read_encoded_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: ZarrsOptions | None = None,
) -> bytes | None:
    """Read the raw, still-encoded bytes of the chunk at `chunk_coords`, or
    `None` if the chunk does not exist. No codecs are applied."""
    result: bytes | None = await asyncio.to_thread(
        _zb.retrieve_encoded_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
    return result


async def encode_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    value: npt.ArrayLike,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Encode `value` with the codecs in `metadata` and store it as the chunk
    at `chunk_coords`. `value` must match the chunk shape exactly."""
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.shape != chunk_shape:
        raise ValueError(f"value shape {arr.shape} does not match chunk shape {chunk_shape}")
    await asyncio.to_thread(
        _zb.store_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
        arr.tobytes(),
    )


async def erase_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Delete the chunk at `chunk_coords`. Deleting a missing chunk is a no-op."""
    await asyncio.to_thread(
        _zb.erase_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
```

Export `decode_chunk`, `read_encoded_chunk`, `encode_chunk`, `erase_chunk` from `__init__.py`.

- [ ] **Step 5: Rebuild and test**

Run: `cargo check --manifest-path zarrs-bindings/Cargo.toml` → success
Run: `uv sync --group zarrs --reinstall-package zarrs-bindings`
Run: `uv run --group zarrs pytest tests/zarrs/test_chunk.py -v`
Expected: all pass. Likely first-run issues and their fixes:
  - v2 differential test fails on dtype byte order → constrain the v2 test to `dtype="<u2"` explicitly (still differential; big-endian support is a later phase).
  - `test_decode_chunk_metadata_view`: if `data_type`/`chunk_grid` key paths differ from the document zarr-python emits, print `meta` and adjust the three edited keys — the point of the test (decode with a never-stored document) must stay.

- [ ] **Step 6: Run the whole suite**

Run: `uv run --group zarrs pytest tests/zarrs -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add zarrs-bindings/src src/zarr/zarrs tests/zarrs
git commit -m "feat: zarrs-backed whole-chunk decode/encode/raw-read/erase"
```

---

### Task 8: Lint, types, changelog

**Files:**
- Create: `changes/+zarrs-bindings.feature.md`
- Modify: anything the linters flag

- [ ] **Step 1: Create `changes/+zarrs-bindings.feature.md`** (orphan towncrier fragment — no PR number needed)

```markdown
Added `zarr.zarrs`, an experimental low-level functional API for zarr hierarchy
CRUD backed by the Rust [zarrs](https://zarrs.dev) crate via the new in-repo
`zarrs-bindings` PyO3 crate. Array routines take an explicit metadata document,
enabling read-only views such as decoding chunks with externally supplied
metadata or reading raw encoded chunk bytes. Build for development with
`uv sync --group zarrs`.
```

- [ ] **Step 2: Run linters and fix findings**

Run: `uv run --group dev ruff format src/zarr/zarrs tests/zarrs && uv run --group dev ruff check --fix src/zarr/zarrs tests/zarrs`
Run: `uv run --group dev --group zarrs mypy src/zarr/zarrs tests/zarrs`
Run: `cargo fmt --manifest-path zarrs-bindings/Cargo.toml && cargo clippy --manifest-path zarrs-bindings/Cargo.toml -- -D warnings`
Expected: clean (fix anything flagged; mypy is strict — tests need `-> None` annotations, which the code above has).

- [ ] **Step 3: Re-run the full zarrs suite** — `uv run --group zarrs pytest tests/zarrs -v` → all pass.

- [ ] **Step 4: Verify the rest of the test suite is unaffected**

Run: `uv run pytest tests/test_array.py tests/test_group.py -x -q`
Expected: pass (no production code outside `src/zarr/zarrs/` changed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: lint fixes and changelog for zarr.zarrs"
```

---

### Task 9: CI workflow

**Files:**
- Create: `.github/workflows/zarrs.yml`

- [ ] **Step 1: Create `.github/workflows/zarrs.yml`** (action SHAs copied from `.github/workflows/test.yml` — keep them identical so dependabot groups them)

```yaml
name: Zarrs bindings

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6.0.3
        with:
          fetch-depth: 0  # hatch-vcs needs tags to compute zarr's version
          persist-credentials: false
      - name: Install uv
        uses: astral-sh/setup-uv@fac544c07dec837d0ccb6301d7b5580bf5edae39 # v8.2.0
        with:
          python-version: '3.12'
      - name: Run zarrs bindings tests
        # the ubuntu runner image ships a Rust toolchain; the maturin build
        # backend is fetched by uv on demand
        run: uv run --group zarrs pytest tests/zarrs -v
```

- [ ] **Step 2: Validate the workflow**

Run: `uvx zizmor .github/workflows/zarrs.yml`
Expected: no findings (matches the repo's zizmor policy).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/zarrs.yml
git commit -m "ci: test job for zarrs bindings"
```

---

## Out of scope for this plan (later phases, per spec)

- `decode_region` / `encode_region` and chunk-subset `selection` (Phase 2: zarrs `retrieve_array_subset` / `partial_decoder`).
- `ZarrsOptions` fields (concurrency, checksum validation, direct IO), obstore native path, benchmarks (Phase 3).
- Variable-length data types, non-regular chunk grids, fancy indexing.
- Publishing the `zarrs-bindings` wheel / a `zarr[zarrs]` extra on PyPI.
