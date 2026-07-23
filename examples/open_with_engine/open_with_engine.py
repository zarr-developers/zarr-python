# NOTE on dependencies / how to run
# ----------------------------------
# Unlike some other examples in this directory, this one does NOT carry a PEP 723
# `# /// script` inline-dependency header.
#
# The feature it demonstrates -- selecting a per-array execution "engine" -- is
# unreleased: it is not present in any published release. The `"zarrista"` engine
# additionally requires the `zarrista` package, which this repo pulls in through
# the `zarrista` dependency *group* (something a single inline `dependencies`
# pin cannot express cleanly). A PEP 723 header pinning a published zarr would
# therefore silently install a zarr WITHOUT this feature and fail at runtime, so
# we deliberately omit it.
#
# To run this example from a checkout of this branch:
#
#     uv run python examples/open_with_engine/open_with_engine.py
#
# and, with the Rust-backed engine available:
#
#     uv run --group zarrista python examples/open_with_engine/open_with_engine.py
#
# The zarrista portion is guarded: if the `zarrista` engine cannot be imported,
# the example still runs and demonstrates the default engine.

"""
Open the same array data with a different backend ("engine").

zarr-python routes an array's data I/O through a selectable *engine*. The engine
is purely an *execution* setting: it chooses which compute backend reads and
writes the array's chunks. The bytes on disk are identical regardless of the
engine -- the same Zarr array, just a different machine doing the work.

Engines demonstrated here:

- `"default"` -- the built-in engine (used when `engine=` is omitted). Works on
  every store and format.
- `"zarrista"` -- a Rust-backed engine (via the `zarrista` package) that serves
  Zarr v3 arrays on a `LocalStore` or an obstore-backed `ObjectStore`. Guarded:
  skipped if the package is not installed.

We write one array once, then open and read it back through each engine and
assert the results are byte-for-byte identical.
"""

import tempfile
from pathlib import Path

import numpy as np

import zarr
from zarr.storage import LocalStore


def zarrista_available() -> bool:
    """Report whether the optional Rust-backed `"zarrista"` engine can be used.

    The `"zarrista"` engine imports the `zarrista` package lazily, so we probe
    that package directly: importing `zarr.zarrista` alone succeeds even when the
    package is absent (the actual `ImportError` would only surface on first I/O).
    """
    try:
        import zarrista  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    # Discover which engines exist. `zarr.list_engines()` returns the built-in
    # engine names; `"zarrista"` additionally requires the `zarrista` package.
    print("available engines:", zarr.list_engines())

    # zarrista ingests a LocalStore (used here, on a temp directory) or an
    # obstore-backed ObjectStore. We keep a single store so every engine reads
    # the exact same bytes off the same disk.
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalStore(Path(tmp) / "store")

        # The data we will write once and read back through several engines.
        data = np.arange(8 * 8, dtype="uint16").reshape(8, 8)

        # --- Write the array once, with the default engine. -----------------
        # No engine= here, so this uses the "default" engine.
        source = zarr.create_array(
            store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16"
        )
        source[:] = data

        # --- 1. Default engine (the baseline). ------------------------------
        # Either omit engine= or pass engine="default"; both mean the built-in.
        default = zarr.open_array(store=store, path="a", engine="default")
        # `array.engine` is the resolved engine instance backing this array.
        print("default  engine:", type(default.engine).__name__)
        np.testing.assert_array_equal(default[:], data)
        # Basic indexing (ints and slices) is what engines route; check a slice.
        np.testing.assert_array_equal(default[2:6, 1:5], data[2:6, 1:5])
        print("default  (engine='default')   : read back OK")

        # --- 2. zarrista engine (Rust), guarded behind availability. --------
        if zarrista_available():
            zst = zarr.open_array(store=store, path="a", engine="zarrista")
            print("zarrista engine:", type(zst.engine).__name__)
            np.testing.assert_array_equal(zst[:], data)
            np.testing.assert_array_equal(zst[2:6, 1:5], data[2:6, 1:5])
            # Same bytes on disk, different compute backend: identical to default.
            np.testing.assert_array_equal(zst[:], default[:])
            print("zarrista (engine='zarrista')  : read back OK, equals default")

            # Writes also route through the engine. Create + write + read back
            # entirely through zarrista, then confirm a *default* reader agrees.
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
            print("zarrista (write path)         : round-trip OK, default reader agrees")
        else:
            print("zarrista                      : SKIPPED (package not installed)")

    print("\nAll engines returned identical data. Same bytes on disk, different backend.")


if __name__ == "__main__":
    main()
