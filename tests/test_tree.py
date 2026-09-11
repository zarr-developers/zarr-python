import textwrap
from typing import Any

import pytest

import zarr


@pytest.mark.parametrize("root_name", [None, "root"])
@pytest.mark.parametrize("atty", [True, False])
@pytest.mark.parametrize("plain", [True, False])
def test_tree(root_name: Any, atty: bool, plain: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: atty)

    if atty and not plain:
        BOPEN = "\x1b[1m"
        BCLOSE = "\x1b[0m"
    else:
        BOPEN = ""
        BCLOSE = ""

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

    result = repr(g.tree(plain=plain))
    root = root_name or ""

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

    result = repr(g.tree(level=0, plain=plain))
    expected = textwrap.dedent(f"""\
        {BOPEN}/{root}{BCLOSE}
        ├── {BOPEN}A{BCLOSE}
        └── {BOPEN}B{BCLOSE}
        """)
    assert result == expected

    if not plain:
        tree = g.tree(plain=False)
        bundle = tree._repr_mimebundle_()
        assert "text/plain" in bundle
        assert "text/html" in bundle
        assert "<b>A</b>" in bundle["text/html"]
        assert "<b>x</b>" in bundle["text/html"]
        assert "<pre" in bundle["text/html"]


def test_tree_truncation() -> None:
    g = zarr.group()
    g.create_group("a")
    g.create_group("b")
    g.create_group("c")
    g.create_group("d")
    g.create_group("e")

    result = repr(g.tree(max_nodes=3, plain=True))
    assert "Truncated at max_nodes=3" in result
    # Should show exactly 3 nodes (lines with ── connectors).
    lines = result.strip().split("\n")
    node_lines = [line for line in lines if "──" in line]
    assert len(node_lines) == 3

    # Full tree should not show truncation message.
    full = repr(g.tree(max_nodes=500, plain=True))
    assert "truncated" not in full


def test_tree_html_escaping() -> None:
    g = zarr.group()
    g.create_group("<img onerror=alert(1) src=x>")

    tree = g.tree()
    bundle = tree._repr_mimebundle_()
    assert "&lt;img" in bundle["text/html"]
    assert "<img" not in bundle["text/html"]
    assert "<img onerror=alert(1) src=x>" in bundle["text/plain"]


def test_expand_not_implemented() -> None:
    g = zarr.group()
    with pytest.raises(NotImplementedError):
        g.tree(expand=True)
