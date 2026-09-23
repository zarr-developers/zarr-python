# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr @ git+https://github.com/d-v-b/zarr-python.git@feat/nczarr-char-dtype",
#   "netCDF4>=1.7",
# ]
# ///
"""
Reproducer: zarr-python and NCZarr (netCDF-C) agree on the netCDF `NC_CHAR` data type.

NCZarr writes a netCDF `NC_CHAR` variable to Zarr V2 with the data type `">S1"`, and an
`NC_STRING` variable as `"|S{N}"`. The Zarr V2 specification says the byte order of `"S"` is not
relevant, so to zarr-python both are fixed-length bytes; the `NCZarrChar` data type keeps the
`">S1"` spelling so the netCDF type survives a round trip through zarr-python.

This script checks, against the netCDF-C library bundled with the `netCDF4` wheel:

1. zarr-python reads an `NC_CHAR` variable written by NCZarr, as `NCZarrChar`.
2. After zarr-python rewrites the array metadata (by updating attributes), netCDF still reads the
   variable as `NC_CHAR`.
3. The spelling is what carries the type: the same variable with its data type rewritten to
   `"|S1"` reads as `NC_STRING`. This is what zarr-python did before `NCZarrChar`, whenever it
   wrote the metadata of an array it had read.
4. In an NCZarr group, netCDF reads an array created by zarr-python with `NCZarrChar` as
   `NC_CHAR`, and one created with `NullTerminatedBytes(length=1)` as `NC_STRING`.

netCDF-C gives `">S1"` this meaning only for arrays with NCZarr metadata (`_nczarr_array`). It
reads a pure Zarr array (`#mode=zarr`) with `">S1"` as `NC_STRING`.

It is not part of the test suite. Run it from the repository root against the local checkout with

    uv run --with netCDF4 python tests/interop/nczarr_char.py

or on its own (which installs zarr from the branch named in the header) with

    uv run tests/interop/nczarr_char.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

import zarr
from zarr.dtype import NCZarrChar, NullTerminatedBytes

if TYPE_CHECKING:
    from zarr.types import JSON

VALUES = np.array([b"a", b"b", b"c"])


def nczarr_url(path: Path) -> str:
    """A netCDF-C URL for an NCZarr store on the local file system."""
    return f"file://{path}#mode=nczarr,file"


def netcdf_types(path: Path) -> dict[str, str]:
    """The netCDF type of each variable, as netCDF-C reads it: `NC_CHAR` or `NC_STRING`."""
    with netCDF4.Dataset(nczarr_url(path), "r") as ds:
        # netCDF4-python reports NC_STRING variables with the Python `str` type
        return {
            name: "NC_STRING" if var.dtype is str else "NC_CHAR"
            for name, var in ds.variables.items()
        }


def stored_dtype(path: Path, variable: str) -> object:
    return json.loads((path / variable / ".zarray").read_text())["dtype"]


def write_netcdf(path: Path) -> None:
    """Write an NC_CHAR and an NC_STRING variable with netCDF-C."""
    with netCDF4.Dataset(nczarr_url(path), "w") as ds:
        ds.createDimension("n", len(VALUES))
        ds.createVariable("char", "S1", ("n",))[:] = VALUES
        ds.createVariable("string", str, ("n",))[:] = np.array(["x", "yy", "zzz"], dtype=object)


def check_read(root: Path) -> None:
    """zarr-python reads NC_CHAR written by NCZarr, and keeps its spelling on a metadata write."""
    path = root / "read.zarr"
    write_netcdf(path)
    assert stored_dtype(path, "char") == ">S1"
    assert netcdf_types(path) == {"char": "NC_CHAR", "string": "NC_STRING"}

    array = zarr.open_array(path / "char", mode="r+")
    assert array.metadata.dtype == NCZarrChar()
    np.testing.assert_array_equal(array[:], VALUES)
    print(f"1. zarr-python reads NCZarr NC_CHAR ('>S1') as {array.metadata.dtype!r}")

    array.update_attributes({"written_by": "zarr-python"})
    assert stored_dtype(path, "char") == ">S1"
    assert netcdf_types(path) == {"char": "NC_CHAR", "string": "NC_STRING"}
    with netCDF4.Dataset(nczarr_url(path), "r") as ds:
        var = ds.variables["char"]
        assert var.getncattr("written_by") == "zarr-python"
        np.testing.assert_array_equal(var[:], VALUES)
    print("2. after zarr-python rewrites the metadata, netCDF still reads NC_CHAR")


def check_canonical_spelling(root: Path) -> None:
    """Rewriting '>S1' as '|S1' turns NC_CHAR into NC_STRING."""
    path = root / "canonical.zarr"
    write_netcdf(path)
    zarray_path = path / "char" / ".zarray"
    zarray = json.loads(zarray_path.read_text())
    zarray_path.write_text(json.dumps({**zarray, "dtype": "|S1"}))
    assert netcdf_types(path) == {"char": "NC_STRING", "string": "NC_STRING"}
    print("3. the same variable written back as '|S1' reads as NC_STRING")


def check_write(root: Path) -> None:
    """netCDF reads NCZarrChar as NC_CHAR, and NullTerminatedBytes(length=1) as NC_STRING."""
    path = root / "write.zarr"
    with netCDF4.Dataset(nczarr_url(path), "w") as ds:
        ds.createDimension("n", len(VALUES))

    group = zarr.open_group(path, mode="r+")
    # the NCZarr metadata netCDF-C writes for a one-dimensional variable
    nczarr_attributes: dict[str, JSON] = {
        "_ARRAY_DIMENSIONS": ["n"],
        "_nczarr_array": {"dimension_references": ["/n"], "storage": "chunked"},
    }
    for name, dtype in (("char", NCZarrChar()), ("bytes", NullTerminatedBytes(length=1))):
        array = group.create_array(
            name, shape=VALUES.shape, dtype=dtype, attributes=nczarr_attributes, compressors=None
        )
        array[:] = VALUES
    # netCDF-C lists the variables of a group in the group's NCZarr metadata
    nczarr_group = group.attrs["_nczarr_group"]
    assert isinstance(nczarr_group, dict)
    group.attrs["_nczarr_group"] = {**nczarr_group, "arrays": ["char", "bytes"]}

    assert stored_dtype(path, "char") == ">S1"
    assert stored_dtype(path, "bytes") == "|S1"
    assert netcdf_types(path) == {"char": "NC_CHAR", "bytes": "NC_STRING"}
    with netCDF4.Dataset(nczarr_url(path), "r") as ds:
        np.testing.assert_array_equal(ds.variables["char"][:], VALUES)
    print(
        "4. netCDF reads zarr-python's NCZarrChar ('>S1') as NC_CHAR, "
        "and NullTerminatedBytes(length=1) ('|S1') as NC_STRING"
    )


def main() -> int:
    print(
        f"zarr {zarr.__version__}, netCDF4 {netCDF4.__version__}, "
        f"netCDF-C {netCDF4.__netcdf4libversion__}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        check_read(root)
        check_canonical_spelling(root)
        check_write(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
