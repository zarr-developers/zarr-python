import os
import textwrap
from typing import Any

import pytest

import zarr

pytest.importorskip("rich")


@pytest.mark.parametrize("root_name", [None, "root"])
def test_tree(root_name: Any) -> None:
    os.environ["OVERRIDE_COLOR_SYSTEM"] = "truecolor"

    g = zarr.group(path=root_name)
    A = g.create_group("A")
    B = g.create_group("B")
    C = B.create_group("C")
    D = C.create_group("C")

    A.create_array(name="x", shape=(2), dtype="float64")
    A.create_array(name="y", shape=(0,), dtype="int8")
    B.create_array(name="x", shape=(0,), dtype="float64")
    C.create_array(name="x", shape=(0,), dtype="float64")
    D.create_array(name="x", shape=(0,), dtype="float64")

    result = repr(g.tree())
    root = root_name or ""

    BOPEN = "\x1b[1m"
    BCLOSE = "\x1b[0m"

    expected = textwrap.dedent(f"""\
        {BOPEN}/{root}{BCLOSE}
        ├── {BOPEN}A{BCLOSE}
        │   ├── {BOPEN}x{BCLOSE} (2,) float64
        │   └── {BOPEN}y{BCLOSE} (0,) int8
        └── {BOPEN}B{BCLOSE}
            ├── {BOPEN}C{BCLOSE}
            │   ├── {BOPEN}C{BCLOSE}
            │   │   └── {BOPEN}x{BCLOSE} (0,) float64
            │   └── {BOPEN}x{BCLOSE} (0,) float64
            └── {BOPEN}x{BCLOSE} (0,) float64
        """)

    assert result == expected

    result = repr(g.tree(level=0))
    expected = textwrap.dedent(f"""\
        {BOPEN}/{root}{BCLOSE}
        ├── {BOPEN}A{BCLOSE}
        └── {BOPEN}B{BCLOSE}
        """)

    assert result == expected


def test_expand_not_implemented() -> None:
    g = zarr.group()
    with pytest.raises(NotImplementedError):
        g.tree(expand=True)
