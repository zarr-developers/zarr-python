# NOTE on dependencies / how to run
# ----------------------------------
# Unlike the other examples in this directory, this one does NOT carry a PEP 723
# `# /// script` inline-dependency header.
#
# The feature it demonstrates -- selecting a per-array execution "engine" -- is
# unreleased: it currently lives only on the `zarrs-bindings` development branch
# of zarr-python and is not present in any published release or in `main`. The
# `zarrista` engine additionally requires the `zarrista` package, which this repo
# pulls in through the `zarrista` dependency *group* (something a single inline
# `dependencies = [...]` pin cannot express cleanly).
#
# A PEP 723 header pinning `zarr @ git+...@main` (as the other examples do) would
# therefore silently install a zarr WITHOUT this feature and fail at runtime, so
# we deliberately omit it rather than ship a header that lies.
#
# To run this example, run it from a checkout of the `zarrs-bindings` branch with
# the zarrista group synced:
#
#     uv sync --group zarrista
#     uv run examples/open_with_backend/open_with_backend.py
#
# The zarrista portion is guarded: if `zarr.zarrista` cannot be imported, the
# example still runs and demonstrates the native vs. reference engines.

"""
Open the same array data with a different backend ("engine").

This branch lets you drive zarr-python's ordinary top-level API
(``create_array`` / ``open_array`` / ``array[...]``) through a selectable
*engine*. The engine is purely an *execution* setting: it chooses which compute
backend reads and writes the chunks. The bytes on disk are identical regardless
of the engine -- the same Zarr array, just a different machine doing the work.

Engines demonstrated here:

- ``"zarr"`` -- the native Python path (the default).
- ``"reference"`` -- a pure-Python backend that routes through ``zarr.crud``;
  works on any store.
- ``"zarrista"`` -- a Rust-backed backend (via the ``zarrista`` package) that
  ingests a ``LocalStore`` or an obstore-backed ``ObjectStore``. Guarded:
  skipped if the package is absent.

We write one array once, then open and read it back through each engine and
assert the results are byte-for-byte identical. We also show the two ways to
select an engine: globally via ``zarr.config`` and per call via ``engine=``.
"""

import tempfile
from pathlib import Path

import numpy as np

import zarr
from zarr.storage import LocalStore


def zarrista_available() -> bool:
    """Report whether the optional Rust-backed ``zarrista`` engine can be used."""
    try:
        import zarr.zarrista  # noqa: F401  (import registers the "zarrista" engine)
    except ImportError:
        return False
    return True


def main() -> None:
    # Discover which engines exist. `zarr.list_engines()` lists native ("zarr")
    # first, then the crud-backed engines -- including lazy ones (zarrista,
    # zarrs) that have not been imported yet.
    print("available engines:", zarr.list_engines())

    # zarrista ingests a LocalStore (used here, on a temp directory) or an
    # obstore-backed ObjectStore. The
    # "reference" engine works on any store, but we keep a single store here so
    # every engine reads the exact same bytes off the same disk.
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalStore(Path(tmp) / "store")

        # The data we will write once and read back through several engines.
        data = np.arange(8 * 8, dtype="uint16").reshape(8, 8)

        # --- Write the array once, with the native engine. ------------------
        # No engine= here, so this uses the default "zarr" (native) engine.
        source = zarr.create_array(
            store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16"
        )
        source[:] = data

        # --- 1. Native engine (the baseline). -------------------------------
        # Either omit engine= or pass engine="zarr"; both mean "native".
        native = zarr.open_array(store=store, path="a", engine="zarr")
        # The selected engine is a public property on the array.
        assert native.engine == "zarr"
        np.testing.assert_array_equal(native[:], data)
        # Basic indexing (ints and slices) is what engines route; check a slice.
        np.testing.assert_array_equal(native[2:6, 1:5], data[2:6, 1:5])
        print("native  (engine='zarr')      : read back OK")

        # --- 2. reference engine (pure Python), via per-call engine= kwarg. --
        reference = zarr.open_array(store=store, path="a", engine="reference")
        assert reference.engine == "reference"
        np.testing.assert_array_equal(reference[:], data)
        np.testing.assert_array_equal(reference[2:6, 1:5], data[2:6, 1:5])
        # Same bytes on disk, different compute backend: identical to native.
        np.testing.assert_array_equal(reference[:], native[:])
        print("reference (engine='reference'): read back OK, equals native")

        # --- 2b. The other way to select an engine: the global config key. --
        # Inside this context manager every array defaults to the "reference"
        # engine without passing engine= at all. We pin the writer that filled
        # `data` already, so here we only read.
        with zarr.config.set({"array.engine": "reference"}):
            via_config = zarr.open_array(store=store, path="a")
            assert via_config.engine == "reference"  # public property; also via .config.engine
            np.testing.assert_array_equal(via_config[:], data)
        print("reference (via array.engine)  : read back OK")

        # --- 3. zarrista engine (Rust), guarded behind availability. --------
        if zarrista_available():
            zst = zarr.open_array(store=store, path="a", engine="zarrista")
            assert zst.engine == "zarrista"
            np.testing.assert_array_equal(zst[:], data)
            np.testing.assert_array_equal(zst[2:6, 1:5], data[2:6, 1:5])
            np.testing.assert_array_equal(zst[:], native[:])
            print("zarrista (engine='zarrista')  : read back OK, equals native")

            # Writes also route through the engine. Create + write + read back
            # entirely through zarrista, then confirm a *native* reader agrees.
            written = zarr.create_array(
                store=store,
                name="b",
                shape=(8, 8),
                chunks=(4, 4),
                dtype="uint16",
                engine="zarrista",
            )
            written[:] = data
            np.testing.assert_array_equal(zarr.open_array(store=store, path="b")[:], data)
            print("zarrista (write path)         : round-trip OK, native reader agrees")
        else:
            print("zarrista                      : SKIPPED (package not installed)")

        # --- Strict policy: engines do basic indexing only. -----------------
        # Orthogonal/coordinate indexing under a non-native engine raises rather
        # than silently falling back to the native path.
        try:
            reference.oindex[[0, 2], :]
        except NotImplementedError:
            print("reference (oindex)            : NotImplementedError as expected (strict policy)")
        else:  # pragma: no cover
            raise AssertionError(
                "expected NotImplementedError from oindex under a non-native engine"
            )

    print("\nAll engines returned identical data. Same bytes on disk, different backend.")


if __name__ == "__main__":
    main()
