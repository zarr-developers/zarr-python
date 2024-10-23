import textwrap
from typing import Any

import pytest

import zarr


@pytest.mark.parametrize("root_name", [None, "root"])
def test_tree(root_name: Any) -> None:
    g = zarr.group(path=root_name)
    A = g.create_group("A")
    B = g.create_group("B")
    C = B.create_group("C")
    D = C.create_group("C")

    A.create_array(name="x", shape=(2), dtype="float64")
    A.create_array(name="y", shape=(0,), dtype="int8")
    B.create_array(name="x", shape=(0,))
    C.create_array(name="x", shape=(0,))
    D.create_array(name="x", shape=(0,))

    result = repr(g.tree())
    root = root_name or ""

    expected = textwrap.dedent(f"""\
        /{root}
        ├── A
        │   ├── x (2,) float64
        │   └── y (0,) int8
        └── B
            ├── C
            │   ├── C
            │   │   └── x (0,) float64
            │   └── x (0,) float64
            └── x (0,) float64
        """)

    assert result == expected

    result = repr(g.tree(level=0))
    expected = textwrap.dedent(f"""\
        /{root}
        ├── A
        └── B
        """)

    assert result == expected


def test_expand_not_implemented() -> None:
    g = zarr.group()
    with pytest.raises(NotImplementedError):
        g.tree(expand=True)
