# Minimal Chunk-Key-Encoding API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the unreleased `zarr-chunk-key-encoding` package to its essential encoding, decoding, JSON, and prepared grid-bounds behavior.

**Architecture:** Keep the two spec-defined encodings and strict JSON dispatch as the core. Preserve `BoundedChunkKeyEncoding` only as a frozen prepared view that validates `grid_shape` once and delegates `encode`/`decode`; remove its collection, persistence, and extra-constructor surface, along with other unused public conveniences.

**Tech Stack:** Python 3.11+, dataclasses, `typing_extensions`, pytest, ruff, pyright, MkDocs, towncrier, uv/just.

**Spec:** `docs/superpowers/specs/2026-09-06-chunk-key-encoding-minimal-api-design.md`

## Global Constraints

- `grid_shape` is normalized and validated once when the bounded view is constructed.
- Per-call coordinates and decoded keys still receive rank and bounds validation.
- `BoundedChunkKeyEncoding` exposes only its data attributes plus `encode` and `decode`; it is not a collection and has no JSON form or alternate constructor.
- The public API is exactly the names listed in the design spec.
- The package remains limited to the two Zarr v3 core encodings and gains no registration mechanism.
- Existing strict, canonical decoding and JSON validation behavior must remain unchanged.
- The package has not been released, so removed PR-only names receive no deprecation aliases.
- Do not integrate this package into zarr-python's existing array classes in this change.

---

### Task 1: Replace the bounded collection with a prepared bounds checker

**Files:**

- Modify: `packages/zarr-chunk-key-encoding/tests/test_bounded.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_bounded.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_abc.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_errors.py`

**Interfaces:**

- Consumes: `normalize_chunk_coords(Sequence[int]) -> tuple[int, ...]`, `ChunkKeyEncoding.encode(Sequence[int]) -> str`, and `ChunkKeyEncoding.decode(str) -> tuple[int, ...]`.
- Produces: `ChunkKeyEncoding.to_bounded(grid_shape: Sequence[int]) -> BoundedChunkKeyEncoding`; `BoundedChunkKeyEncoding(encoding: ChunkKeyEncoding, grid_shape: Sequence[int])`; bounded `encode(Sequence[int]) -> str`; bounded `decode(str) -> tuple[int, ...]`.

- [ ] **Step 1: Rewrite the bounded success tests around the prepared API**

  Replace collection/JSON/construction-path tests with one parametrized normal-behavior test and a focused one-time-normalization regression:

  ```python
  @pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
  @pytest.mark.parametrize(
      ("grid_shape", "chunk_coords"),
      [((), ()), ((1,), (0,)), ((5,), (4,)), ((2, 3), (1, 2)), ((4, 1, 7), (3, 0, 6))],
  )
  def test_encode_decode(
      encoding: ChunkKeyEncoding,
      grid_shape: tuple[int, ...],
      chunk_coords: tuple[int, ...],
  ) -> None:
      bounded = encoding.to_bounded(grid_shape)
      key = bounded.encode(chunk_coords)
      assert bounded.encoding is encoding
      assert bounded.grid_shape == grid_shape
      assert key == encoding.encode(chunk_coords)
      assert bounded.decode(key) == chunk_coords

  class _OneShotShape(Sequence[int]):
      def __init__(self) -> None:
          self._read = False

      def __len__(self) -> int:
          return 2

      def __getitem__(self, index: int) -> int:
          if self._read:
              raise AssertionError("grid shape was read again")
          if index == 0:
              return 2
          if index == 1:
              self._read = True
              return 3
          raise IndexError

  def test_grid_shape_is_normalized_once() -> None:
      bounded = DefaultChunkKeyEncoding().to_bounded(_OneShotShape())
      assert bounded.grid_shape == (2, 3)
      assert bounded.encode((1, 2)) == "c/1/2"
      assert bounded.decode("c/1/2") == (1, 2)
  ```

  Retain `_NoDecode`, changing its `encode` return type and implementation to plain `str`. Assert that ordinary bounded decode propagates `NotImplementedError` while rank-zero decode recognizes the single encoded key directly.

- [ ] **Step 2: Write one bounded test for each error category and absence of removed protocols**

  Use separate tests for invalid grid shape, invalid coordinate value, wrong coordinate rank, out-of-bounds coordinate, malformed key, wrong decoded rank, out-of-bounds decoded coordinate, and rank-zero out-of-domain key. Expect only the parent exceptions:

  ```python
  def test_encode_wrong_rank() -> None:
      bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
      with pytest.raises(InvalidChunkCoordsError, match="shape"):
          bounded.encode((1,))

  def test_decode_out_of_bounds() -> None:
      bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
      with pytest.raises(ChunkKeyDecodeError, match="outside"):
          bounded.decode("c/2/0")

  def test_removed_bounded_protocols() -> None:
      bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
      for name in ("from_json", "from_unbounded", "to_json", "__contains__", "__iter__", "__len__"):
          assert not hasattr(bounded, name)
  ```

- [ ] **Step 3: Run the rewritten bounded tests and confirm the old implementation fails**

  Run:

  ```bash
  uv run --group test --with-editable ../zarr-metadata python -m pytest tests/test_bounded.py -q
  ```

  Expected: failures from the removed-protocol assertion and old bounded-specific exception expectations/imports until production code is slimmed.

- [ ] **Step 4: Implement the minimal frozen bounded dataclass**

  Replace `_bounded.py` with a focused implementation shaped like:

  ```python
  from collections.abc import Sequence
  from dataclasses import dataclass

  from zarr_chunk_key_encoding._abc import ChunkKeyEncoding
  from zarr_chunk_key_encoding._errors import (
      ChunkKeyConfigurationError,
      ChunkKeyDecodeError,
      InvalidChunkCoordsError,
  )
  from zarr_chunk_key_encoding._parsing import normalize_chunk_coords

  def _normalize_grid_shape(grid_shape: Sequence[int]) -> tuple[int, ...]:
      try:
          return normalize_chunk_coords(grid_shape)
      except InvalidChunkCoordsError as exc:
          raise ChunkKeyConfigurationError(
              f"Invalid chunk grid shape {grid_shape!r}: entries must be non-negative integers."
          ) from exc

  @dataclass(frozen=True)
  class BoundedChunkKeyEncoding:
      encoding: ChunkKeyEncoding
      grid_shape: tuple[int, ...]

      def __init__(self, encoding: ChunkKeyEncoding, grid_shape: Sequence[int]) -> None:
          object.__setattr__(self, "encoding", encoding)
          object.__setattr__(self, "grid_shape", _normalize_grid_shape(grid_shape))

      def _in_grid(self, coords: tuple[int, ...]) -> bool:
          return len(coords) == len(self.grid_shape) and all(
              coordinate < extent
              for coordinate, extent in zip(coords, self.grid_shape, strict=True)
          )

      def encode(self, chunk_coords: Sequence[int]) -> str:
          coords = normalize_chunk_coords(chunk_coords)
          if not self._in_grid(coords):
              raise InvalidChunkCoordsError(
                  f"Chunk coordinates {coords!r} do not name a cell of the chunk grid "
                  f"with shape {self.grid_shape!r}."
              )
          return self.encoding.encode(coords)

      def decode(self, chunk_key: str) -> tuple[int, ...]:
          if self.grid_shape == () and chunk_key == self.encoding.encode(()):
              return ()
          coords = self.encoding.decode(chunk_key)
          if not self._in_grid(coords):
              raise ChunkKeyDecodeError(
                  f"Chunk key {chunk_key!r} names a chunk outside the chunk grid "
                  f"with shape {self.grid_shape!r}."
              )
          return coords
  ```

  Update `ChunkKeyEncoding.to_bounded` to call `BoundedChunkKeyEncoding(self, grid_shape)` directly and describe only the prepared bounds behavior. Remove the two bounded-specific exception classes from `_errors.py`.

- [ ] **Step 5: Run bounded tests and commit**

  Run the command from Step 3. Expected: all bounded tests pass.

  ```bash
  git add packages/zarr-chunk-key-encoding/tests/test_bounded.py \
    packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_bounded.py \
    packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_abc.py \
    packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_errors.py
  git commit -m "refactor: slim bounded chunk key encoding" \
    -m "Assisted-by: Codex:gpt-5"
  ```

---

### Task 2: Reduce and harden the core public surface

**Files:**

- Modify: `packages/zarr-chunk-key-encoding/tests/test_public_api.py`
- Modify: `packages/zarr-chunk-key-encoding/tests/test_from_json.py`
- Modify: `packages/zarr-chunk-key-encoding/tests/test_default.py`
- Modify: `packages/zarr-chunk-key-encoding/tests/test_v2.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/__init__.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_abc.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_default.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_v2.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_from_json.py`
- Modify: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_parsing.py`
- Delete: `packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding/_separator.py`

**Interfaces:**

- Consumes: the prepared bounded API from Task 1 and the existing `zarr_metadata` JSON types.
- Produces: the exact top-level API set from the design spec; concrete `encode` methods returning plain `str`; private `_get_chunk_key_encoding_class`; unexported `parse_separator` colocated in the private parsing module.

- [ ] **Step 1: Pin the reduced API and retained JSON behavior in tests**

  Add an exact set assertion to `test_public_api.py`:

  ```python
  EXPECTED_PUBLIC_API = {
      "BoundedChunkKeyEncoding",
      "ChunkKeyConfigurationError",
      "ChunkKeyDecodeError",
      "ChunkKeyEncoding",
      "ChunkKeyEncodingError",
      "ChunkKeyEncodingJSON",
      "DefaultChunkKeyEncoding",
      "InvalidChunkCoordsError",
      "Separator",
      "UnknownChunkKeyEncodingError",
      "V2ChunkKeyEncoding",
      "__version__",
      "chunk_key_encoding_from_json",
  }

  def test_public_api_is_minimal() -> None:
      assert set(zarr_chunk_key_encoding.__all__) == EXPECTED_PUBLIC_API
  ```

  Remove tests for public dispatch tables, lookup helpers, and loose parsing from `test_from_json.py`; retain one parametrized normal dispatch test plus separate missing-name, unknown-name, and invalid-input error tests. Add assertions in the concrete encoder tests that encoded keys are instances of `str`.

- [ ] **Step 2: Run the focused tests and confirm the oversized API fails**

  Run:

  ```bash
  uv run --group test --with-editable ../zarr-metadata python -m pytest \
    tests/test_public_api.py tests/test_from_json.py tests/test_default.py tests/test_v2.py -q
  ```

  Expected: `test_public_api_is_minimal` fails because removed names are still exported.

- [ ] **Step 3: Remove `ChunkKey` and return plain strings**

  Delete the `NewType` and its documentation from `_abc.py`; change the abstract `encode` return type and both concrete overrides to `str`; return the joined string directly in `_default.py` and `_v2.py`. Remove every `ChunkKey` import.

- [ ] **Step 4: Privatize closed dispatch and remove loose parsing**

  In `_from_json.py`, replace the public mapping proxy and lookup helper with a private immutable mapping and helper:

  ```python
  _CHUNK_KEY_ENCODINGS: Final = {
      DefaultChunkKeyEncoding.name: DefaultChunkKeyEncoding,
      V2ChunkKeyEncoding.name: V2ChunkKeyEncoding,
  }

  def _get_chunk_key_encoding_class(name: str) -> type[ChunkKeyEncoding]:
      try:
          return _CHUNK_KEY_ENCODINGS[name]
      except KeyError:
          raise UnknownChunkKeyEncodingError(name, tuple(_CHUNK_KEY_ENCODINGS)) from None
  ```

  Keep `chunk_key_encoding_from_json` and point it at the private helper. Delete `ChunkKeyEncodingParams`, `ChunkKeyEncodingLike`, `parse_chunk_key_encoding`, related imports, and explanatory text that promises the loose input form.

- [ ] **Step 5: Fold separator validation into `_parsing.py`**

  Move `Separator = Literal[".", "/"]` and separator validation into the private `_parsing.py` module as `parse_separator(data: object) -> Separator`. Keep the permitted tuple private and do not re-export the validator. Update `_default.py` and `_v2.py` imports, then delete `_separator.py`.

- [ ] **Step 6: Make only the top-level module declare public API**

  Remove `__all__` from `_abc.py`, `_bounded.py`, `_default.py`, `_errors.py`, `_from_json.py`, `_parsing.py`, and `_v2.py`. Rewrite `__init__.py` imports and module docstring to expose exactly `EXPECTED_PUBLIC_API` and describe the prepared bounded view without collection or loose-parser claims.

- [ ] **Step 7: Run all package tests and commit**

  Run:

  ```bash
  uv run --group test --with-editable ../zarr-metadata python -m pytest tests -q
  ```

  Expected: all package tests pass (the parity module may skip when repo-root zarr is absent).

  ```bash
  git add packages/zarr-chunk-key-encoding/src packages/zarr-chunk-key-encoding/tests
  git commit -m "refactor: reduce chunk key encoding API" \
    -m "Assisted-by: Codex:gpt-5"
  ```

---

### Task 3: Trim duplicated documentation and release notes

**Files:**

- Modify: `packages/zarr-chunk-key-encoding/README.md`
- Modify: `packages/zarr-chunk-key-encoding/docs/index.md`
- Modify: `packages/zarr-chunk-key-encoding/changes/299.feature.md`
- Delete: `packages/zarr-chunk-key-encoding/changes/299.feature.1.md`
- Delete: `packages/zarr-chunk-key-encoding/changes/299.feature.2.md`

**Interfaces:**

- Consumes: the final API from Tasks 1 and 2.
- Produces: concise user-facing examples for core JSON construction and reusable bounded operation, plus a single accurate initial-release feature fragment.

- [ ] **Step 1: Rewrite README and docs examples against the reduced API**

  Keep installation, one JSON construction/encode/decode example, one bounded example, a short explanation of strict decoding, a short closed-core-set statement, and development commands. The bounded example must use only:

  ```python
  bounded = encoding.to_bounded((2, 3))
  bounded.encode((1, 2))
  bounded.decode("c/1/2")
  ```

  Remove all references to `ChunkKey`, collection membership/iteration/length, bounded JSON, public dispatch mappings, loose parsing, and total-inverse guarantees. Condense `docs/index.md` to the same facts without reproducing the README's full design essay.

- [ ] **Step 2: Consolidate the unreleased feature fragments**

  Rewrite `changes/299.feature.md` to name the retained classes, `chunk_key_encoding_from_json`, strict decoding, and the thin prepared bounded view. Delete the two fragments dedicated to the removed bounded extras and `ChunkKey`. Preserve `changes/299.bugfix.md` and the sibling `zarr-metadata/changes/299.misc.md` unchanged.

- [ ] **Step 3: Verify docs and changelog, then commit**

  Run:

  ```bash
  env DISABLE_MKDOCS_2_WARNING=true uv run --group docs \
    --with-editable ../zarr-metadata mkdocs build --strict
  uvx towncrier build --draft --version Unreleased
  ```

  Expected: strict docs build succeeds and the draft contains one accurate feature entry plus the retained bugfix entry.

  ```bash
  git add packages/zarr-chunk-key-encoding/README.md \
    packages/zarr-chunk-key-encoding/docs/index.md \
    packages/zarr-chunk-key-encoding/changes
  git commit -m "docs: focus chunk key encoding package" \
    -m "Assisted-by: Codex:gpt-5"
  ```

---

### Task 4: Survey for residual fat and verify the package end to end

**Files:**

- Modify only if the searches expose a stale reference: files under `packages/zarr-chunk-key-encoding/`
- Do not modify: release workflows, CI workflows, lockfiles, package metadata, parity coverage, or `packages/zarr-metadata/` unless a concrete failure proves a change is required.

**Interfaces:**

- Consumes: the complete reduced package.
- Produces: evidence that removed concepts are absent, retained infrastructure works, and the package remains compatible with zarr-python's current encoders.

- [ ] **Step 1: Search the entire package for removed concepts and accidental public surface**

  Run:

  ```bash
  rg -n "BoundedChunkKeyEncodingJSON|ChunkCoordsOutOfBoundsError|ChunkKeyOutOfBoundsError|ChunkKeyEncodingLike|ChunkKeyEncodingParams|CHUNK_KEY_ENCODINGS|SEPARATORS|get_chunk_key_encoding_class|parse_chunk_key_encoding|parse_separator|Collection|from_unbounded|to_json\(\).*bounded|total inverse|membership|finite key set|ChunkKey\(" packages/zarr-chunk-key-encoding
  rg -n "^__all__" packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding
  find packages/zarr-chunk-key-encoding/src/zarr_chunk_key_encoding -maxdepth 1 -type f -print | sort
  ```

  Expected: no removed public names or claims remain; only top-level `__init__.py` defines `__all__`; all non-`__init__` Python modules remain underscore-prefixed.

- [ ] **Step 2: Fix only concrete residuals found by the survey**

  For each hit, classify it before editing: remove stale API/docs/tests; retain private `_get_chunk_key_encoding_class` and the unexported separator validator in `_parsing.py`; retain release/CI/lockfile material. Use `apply_patch` for any textual correction, then rerun Step 1 until only intended internal-name hits remain.

- [ ] **Step 3: Run the full verification matrix**

  From `packages/zarr-chunk-key-encoding`, run:

  ```bash
  uvx ruff check .
  uv run --python 3.11 --group test --with pyright \
    --with-editable ../zarr-metadata pyright src
  uv run --group test --with-editable ../zarr-metadata python -m pytest tests -q
  uv run --project ../.. --group test --with-editable . \
    python -m pytest tests/test_zarr_parity.py -q
  env DISABLE_MKDOCS_2_WARNING=true uv run --group docs \
    --with-editable ../zarr-metadata mkdocs build --strict
  uvx towncrier build --draft --version Unreleased
  git diff --check
  ```

  Expected: ruff and pyright report no errors; all package and parity tests pass (or parity skips only when its documented environment is absent); strict docs and changelog draft succeed; `git diff --check` is silent.

- [ ] **Step 4: Review the final diff and commit any survey cleanup**

  Run:

  ```bash
  git diff --stat origin/claude/zarr-chunk-key-encoding-pkg-ef7f13...HEAD
  git diff --check
  git status --short
  ```

  If Step 2 changed files after Task 3, commit only those concrete cleanups:

  ```bash
  git add packages/zarr-chunk-key-encoding
  git commit -m "refactor: remove residual chunk key encoding surface" \
    -m "Assisted-by: Codex:gpt-5"
  ```

  Expected: the worktree is clean, the diff is confined to the design/plan and `packages/zarr-chunk-key-encoding`, and no sibling package or release infrastructure was changed accidentally.
