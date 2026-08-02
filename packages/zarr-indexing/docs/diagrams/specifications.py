"""Typed, data-only specifications for documentation diagrams."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


type SemanticRole = Literal["request", "source", "selected", "chunk-local", "cell-domain", "text"]


class GridSpec(TypedDict):
    """A rectangular grid of equally sized cells."""

    id: str
    kind: Literal["grid"]
    role: SemanticRole
    rows: int
    columns: int
    cell_size: int
    origin: tuple[int, int]
    selected: tuple[tuple[int, int], ...]
    chunk_shape: NotRequired[tuple[int, int] | None]


class AxisSpec(TypedDict):
    """A labelled horizontal or vertical axis."""

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
    """A labelled rectangular node."""

    id: str
    kind: Literal["node"]
    role: SemanticRole
    x: int
    y: int
    width: int
    height: int
    label: str


class ArrowSpec(TypedDict):
    """A labelled edge between two node IDs."""

    id: str
    kind: Literal["arrow"]
    role: SemanticRole
    source: str
    target: str
    label: str


class TextSpec(TypedDict):
    """A free-standing text label."""

    id: str
    kind: Literal["text"]
    role: SemanticRole
    x: int
    y: int
    label: str


type ElementSpec = GridSpec | AxisSpec | NodeSpec | ArrowSpec | TextSpec


class FigureSpec(TypedDict):
    """A complete, accessible diagram."""

    id: str
    title: str
    description: str
    width: int
    height: int
    elements: tuple[ElementSpec, ...]


FIGURES: tuple[FigureSpec, ...] = ()

_SEMANTIC_ROLES: frozenset[str] = frozenset(
    {"request", "source", "selected", "chunk-local", "cell-domain", "text"}
)
_ELEMENT_KINDS: frozenset[str] = frozenset({"grid", "axis", "node", "arrow", "text"})


def _invalid(figure_id: str, field: str, detail: str) -> None:
    raise ValueError(f"{figure_id}: {field} {detail}")


def _positive(figure_id: str, element_id: str, element: dict[str, object], field: str) -> None:
    value = element.get(field)
    if not isinstance(value, int) or value <= 0:
        _invalid(figure_id, f"{element_id}.{field}", "must be a positive integer")


def validate_figure(spec: FigureSpec) -> None:
    """Validate a figure before rendering it.

    Raises
    ------
    ValueError
        If the specification cannot be rendered unambiguously.
    """
    figure_id = spec.get("id", "<missing-id>")
    if not isinstance(figure_id, str) or not figure_id:
        _invalid("<missing-id>", "id", "must be a non-empty string")
    for field in ("title", "description"):
        if not isinstance(spec.get(field), str) or not spec[field].strip():
            _invalid(figure_id, field, "must be a non-empty string")
    for field in ("width", "height"):
        value = spec.get(field)
        if not isinstance(value, int) or value <= 0:
            _invalid(figure_id, field, "must be a positive integer")

    element_ids: set[str] = set()
    node_ids: set[str] = set()
    arrows: list[ArrowSpec] = []
    for element in spec["elements"]:
        element_id = element.get("id")
        if not isinstance(element_id, str) or not element_id:
            _invalid(figure_id, "elements.id", "must be a non-empty string")
        if element_id in element_ids:
            _invalid(figure_id, element_id, "is a duplicate element ID")
        element_ids.add(element_id)

        role = element.get("role")
        if role not in _SEMANTIC_ROLES:
            _invalid(figure_id, f"{element_id}.role", "must be a known semantic role")
        kind = element.get("kind")
        if kind not in _ELEMENT_KINDS:
            _invalid(figure_id, f"{element_id}.kind", "must be a known element kind")

        if kind == "grid":
            _validate_grid(figure_id, element_id, element)
        elif kind == "axis":
            _validate_axis(figure_id, element_id, element)
        elif kind == "node":
            _validate_node(figure_id, element_id, element)
            node_ids.add(element_id)
        elif kind == "arrow":
            arrows.append(element)  # type: ignore[arg-type]
        elif kind == "text":
            _validate_text(figure_id, element_id, element)

    for arrow in arrows:
        arrow_id = arrow["id"]
        for field in ("source", "target"):
            node_id = arrow.get(field)
            if not isinstance(node_id, str) or node_id not in node_ids:
                _invalid(figure_id, f"{arrow_id}.{field}", f"references missing node {node_id!r}")


def _validate_grid(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    for field in ("rows", "columns", "cell_size"):
        _positive(figure_id, element_id, element, field)
    origin = element.get("origin")
    if not _pair_of_ints(origin):
        _invalid(figure_id, f"{element_id}.origin", "must be a pair of integers")
    selected = element.get("selected")
    if not isinstance(selected, tuple) or not all(_pair_of_ints(cell) for cell in selected):
        _invalid(figure_id, f"{element_id}.selected", "must contain coordinate pairs")
    rows = element["rows"]
    columns = element["columns"]
    assert isinstance(rows, int)
    assert isinstance(columns, int)
    for row, column in selected:
        if not 0 <= row < rows or not 0 <= column < columns:
            _invalid(figure_id, f"{element_id}.selected", "contains a coordinate outside the grid")
    chunk_shape = element.get("chunk_shape")
    if chunk_shape is not None:
        if not _pair_of_ints(chunk_shape) or any(size <= 0 for size in chunk_shape):
            _invalid(figure_id, f"{element_id}.chunk_shape", "must be positive dimensions")


def _validate_axis(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    _positive(figure_id, element_id, element, "length")
    for field in ("x", "y", "start", "stop"):
        if not isinstance(element.get(field), int):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    if element.get("orientation") not in {"horizontal", "vertical"}:
        _invalid(figure_id, f"{element_id}.orientation", "must be horizontal or vertical")
    if element["stop"] < element["start"]:
        _invalid(figure_id, f"{element_id}.stop", "must not precede start")


def _validate_node(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    for field in ("x", "y"):
        if not isinstance(element.get(field), int):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    for field in ("width", "height"):
        _positive(figure_id, element_id, element, field)
    _validate_label(figure_id, element_id, element)


def _validate_text(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    for field in ("x", "y"):
        if not isinstance(element.get(field), int):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    _validate_label(figure_id, element_id, element)


def _validate_label(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    if not isinstance(element.get("label"), str):
        _invalid(figure_id, f"{element_id}.label", "must be a string")


def _pair_of_ints(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, int) for item in value)
