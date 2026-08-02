from __future__ import annotations

from typing import Literal, TypedDict
from xml.etree import ElementTree

import pytest

from docs.diagrams.render import render_figure
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
