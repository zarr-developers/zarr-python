from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict
from xml.etree import ElementTree

import pytest

import docs.diagrams.render as diagram_render
from docs.diagrams.render import STYLESHEET_NAME, render_all, render_figure
from docs.diagrams.specifications import FigureSpec, validate_figure

SemanticRole = Literal["request", "source", "selected", "chunk-local", "cell-domain", "text"]


class GridSpec(TypedDict):
    id: str
    kind: Literal["grid"]
    role: SemanticRole
    rows: int
    columns: int
    cell_size: int
    origin: tuple[int, int]
    selected: tuple[tuple[int, int], ...]
    chunk_shape: tuple[int, int] | None


class AxisSpec(TypedDict):
    id: str
    kind: Literal["axis"]
    role: SemanticRole
    x: int
    y: int
    length: int
    orientation: Literal["horizontal", "vertical"]
    start: int
    stop: int


class NodeSpec(TypedDict):
    id: str
    kind: Literal["node"]
    role: SemanticRole
    x: int
    y: int
    width: int
    height: int
    label: str


class ArrowSpec(TypedDict):
    id: str
    kind: Literal["arrow"]
    role: SemanticRole
    source: str
    target: str
    label: str


class TextSpec(TypedDict):
    id: str
    kind: Literal["text"]
    role: SemanticRole
    x: int
    y: int
    label: str


type ElementSpec = GridSpec | AxisSpec | NodeSpec | ArrowSpec | TextSpec


def figure(figure_id: str, *, elements: tuple[ElementSpec, ...]) -> FigureSpec:
    return {
        "id": figure_id,
        "title": f"Figure {figure_id}",
        "description": "A deterministic renderer test figure.",
        "width": 240,
        "height": 120,
        "elements": elements,
    }


GRID: GridSpec = {
    "id": "grid",
    "kind": "grid",
    "role": "selected",
    "rows": 2,
    "columns": 3,
    "cell_size": 20,
    "origin": (10, 10),
    "selected": ((1, 2),),
    "chunk_shape": (1, 2),
}
AXIS: AxisSpec = {
    "id": "axis",
    "kind": "axis",
    "role": "text",
    "x": 10,
    "y": 90,
    "length": 100,
    "orientation": "horizontal",
    "start": 0,
    "stop": 5,
}
LEFT_NODE: NodeSpec = {
    "id": "left",
    "kind": "node",
    "role": "source",
    "x": 10,
    "y": 10,
    "width": 40,
    "height": 24,
    "label": "source",
}
RIGHT_NODE: NodeSpec = {
    "id": "right",
    "kind": "node",
    "role": "request",
    "x": 100,
    "y": 10,
    "width": 40,
    "height": 24,
    "label": "request",
}
ARROW: ArrowSpec = {
    "id": "arrow",
    "kind": "arrow",
    "role": "text",
    "source": "left",
    "target": "right",
    "label": "maps to",
}
TEXT: TextSpec = {
    "id": "text",
    "kind": "text",
    "role": "cell-domain",
    "x": 10,
    "y": 110,
    "label": "cell domain",
}
VALID_ELEMENT_COMBINATIONS: tuple[tuple[ElementSpec, ...], ...] = (
    (GRID,),
    (AXIS,),
    (LEFT_NODE,),
    (LEFT_NODE, RIGHT_NODE, ARROW),
    (TEXT,),
)


@pytest.mark.parametrize("elements", VALID_ELEMENT_COMBINATIONS)
def test_render_figure_accepts_reasonable_element_combinations(
    elements: tuple[ElementSpec, ...],
) -> None:
    spec = figure("valid", elements=elements)
    first = render_figure(spec)
    second = render_figure(spec)
    assert first == second
    root = ElementTree.fromstring(first)
    assert root.tag.endswith("svg")
    assert root.attrib["role"] == "img"


def test_grid_size_must_be_positive() -> None:
    invalid_grid = {**GRID, "rows": 0}
    with pytest.raises(ValueError, match="invalid-grid.*rows"):
        validate_figure(figure("invalid-grid", elements=(invalid_grid,)))


def test_selected_coordinate_must_be_inside_grid() -> None:
    invalid_grid = {**GRID, "selected": ((2, 0),)}
    with pytest.raises(ValueError, match="outside-grid.*selected"):
        validate_figure(figure("outside-grid", elements=(invalid_grid,)))


def test_element_ids_must_be_unique() -> None:
    duplicate = {**TEXT, "id": "grid"}
    with pytest.raises(ValueError, match="duplicate-id.*grid"):
        validate_figure(figure("duplicate-id", elements=(GRID, duplicate)))


def test_title_must_not_be_empty() -> None:
    spec = figure("empty-title", elements=(TEXT,))
    spec["title"] = ""
    with pytest.raises(ValueError, match="empty-title.*title"):
        validate_figure(spec)


def test_description_must_not_be_empty() -> None:
    spec = figure("empty-description", elements=(TEXT,))
    spec["description"] = ""
    with pytest.raises(ValueError, match="empty-description.*description"):
        validate_figure(spec)


def test_semantic_role_must_be_known() -> None:
    invalid_text = {**TEXT, "role": "unknown"}
    with pytest.raises(ValueError, match="unknown-role.*role"):
        validate_figure(figure("unknown-role", elements=(invalid_text,)))


def test_arrow_target_must_resolve() -> None:
    broken_arrow: ArrowSpec = {
        "id": "a",
        "kind": "arrow",
        "role": "text",
        "source": "left",
        "target": "missing",
        "label": "broken",
    }
    spec = figure("broken-arrow", elements=(LEFT_NODE, broken_arrow))
    with pytest.raises(ValueError, match="broken-arrow.*missing"):
        validate_figure(spec)


def test_grid_chunk_shape_renders_chunk_boundaries() -> None:
    grid = {
        **GRID,
        "rows": 6,
        "columns": 8,
        "cell_size": 10,
        "origin": (5, 7),
        "chunk_shape": (3, 4),
    }
    root = ElementTree.fromstring(render_figure(figure("chunked", elements=(grid,))))
    boundaries = [
        line.attrib
        for line in root.iter()
        if line.tag.endswith("line") and line.attrib.get("class") == "zi-chunk-boundary"
    ]
    assert boundaries == [
        {"class": "zi-chunk-boundary", "x1": "45", "x2": "45", "y1": "7", "y2": "67"},
        {"class": "zi-chunk-boundary", "x1": "5", "x2": "85", "y1": "37", "y2": "37"},
    ]
    unchunked_root = ElementTree.fromstring(
        render_figure(figure("unchunked", elements=({**grid, "chunk_shape": None},)))
    )
    assert not any(
        line.tag.endswith("line") and line.attrib.get("class") == "zi-chunk-boundary"
        for line in unchunked_root.iter()
    )


def test_arrow_label_must_be_a_string() -> None:
    arrow_without_label = {
        "id": "arrow",
        "kind": "arrow",
        "role": "text",
        "source": "left",
        "target": "right",
    }
    spec = figure("missing-arrow-label", elements=(LEFT_NODE, RIGHT_NODE, arrow_without_label))
    with pytest.raises(ValueError, match="missing-arrow-label.*arrow.label"):
        validate_figure(spec)


def test_grid_geometry_must_not_be_boolean() -> None:
    grid_with_boolean_rows = {**GRID, "rows": True}
    with pytest.raises(ValueError, match="boolean-grid.*grid.rows"):
        validate_figure(figure("boolean-grid", elements=(grid_with_boolean_rows,)))


def test_figure_geometry_must_not_be_boolean() -> None:
    spec = figure("boolean-figure", elements=(TEXT,))
    spec["width"] = True
    with pytest.raises(ValueError, match="boolean-figure.*width"):
        validate_figure(spec)


@pytest.fixture
def generated_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(diagram_render, "FIGURES", (figure("generated", elements=(TEXT,)),))
    render_all(tmp_path, check=False)
    return tmp_path


def _snapshot(output_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in output_dir.iterdir()}


def test_render_all_update_then_check_writes_deterministic_utf8_lf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram_render, "FIGURES", (figure("generated", elements=(TEXT,)),))
    original_write_text = Path.write_text
    write_options: list[dict[str, object]] = []

    def write_text_with_options(self: Path, data: str, **kwargs: object) -> int:
        write_options.append(kwargs)
        return original_write_text(self, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text_with_options)
    render_all(tmp_path, check=False)
    render_all(tmp_path, check=True)
    assert set(_snapshot(tmp_path)) == {"generated.svg", STYLESHEET_NAME}
    assert write_options == [{"encoding": "utf-8", "newline": "\n"}] * 2


@pytest.mark.parametrize(
    ("filename", "replacement"),
    [
        ("generated.svg", b"changed"),
        (STYLESHEET_NAME, b"changed stylesheet"),
        ("missing.svg", None),
        ("orphan.svg", b"orphan"),
    ],
)
def test_render_all_check_reports_the_exact_stale_path(
    generated_assets: Path, filename: str, replacement: bytes | None
) -> None:
    path = generated_assets / ("generated.svg" if filename == "missing.svg" else filename)
    if replacement is None:
        path.unlink()
    else:
        path.write_bytes(replacement)
    with pytest.raises(RuntimeError) as error:
        render_all(generated_assets, check=True)
    assert str(error.value) == f"stale generated diagram: {path}"


def test_render_all_check_includes_stylesheet_and_does_not_mutate(generated_assets: Path) -> None:
    stylesheet = generated_assets / STYLESHEET_NAME
    assert stylesheet.exists()
    before = _snapshot(generated_assets)
    render_all(generated_assets, check=True)
    assert _snapshot(generated_assets) == before
