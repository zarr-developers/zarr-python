"""Typed, data-only specifications for documentation diagrams."""

from __future__ import annotations

import re
from typing import Literal, Never, NotRequired, TypeGuard, TypedDict


SemanticRole = Literal["request", "source", "selected", "chunk-local", "cell-domain", "text"]


class RoleStyle(TypedDict):
    """Explicit colours for one semantic role."""

    fill: str
    stroke: str
    text: str


PALETTES: dict[str, dict[SemanticRole, RoleStyle]] = {
    "default": {
        "request": {"fill": "#dbeafe", "stroke": "#0969da", "text": "#0b376d"},
        "source": {"fill": "#f6f8fa", "stroke": "#57606a", "text": "#24292f"},
        "selected": {"fill": "#fff1a8", "stroke": "#9a6700", "text": "#3d2b00"},
        "chunk-local": {"fill": "#dafbe1", "stroke": "#1a7f37", "text": "#123d1c"},
        "cell-domain": {"fill": "#fbefff", "stroke": "#8250df", "text": "#3b1764"},
        "text": {"fill": "#ffffff", "stroke": "#57606a", "text": "#24292f"},
    },
    "slate": {
        "request": {"fill": "#153a5b", "stroke": "#58a6ff", "text": "#e6f2ff"},
        "source": {"fill": "#24282f", "stroke": "#9da7b3", "text": "#f4f7fb"},
        "selected": {"fill": "#4a3b00", "stroke": "#e3b341", "text": "#fff4bd"},
        "chunk-local": {"fill": "#153d29", "stroke": "#56d364", "text": "#dff7e7"},
        "cell-domain": {"fill": "#392557", "stroke": "#a371f7", "text": "#f3e8ff"},
        "text": {"fill": "#1f2329", "stroke": "#9da7b3", "text": "#f4f7fb"},
    },
}

ROLE_CUES: dict[SemanticRole, str] = {
    "request": "request label + double border",
    "source": "source label + neutral single border",
    "selected": "coordinate label + heavy outline",
    "chunk-local": "chunk-local label + dashed border",
    "cell-domain": "cell-domain label + rounded border",
    "text": "text label",
}


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
    label_offset: NotRequired[tuple[int, int]]


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


FIGURES: tuple[FigureSpec, ...] = (
    {
        "id": "indexing-selection",
        "title": "Basic indexing selects a one-dimensional request",
        "description": "image[1:5, 2] maps four source cells to a length-four request.",
        "width": 900,
        "height": 390,
        "elements": (
            {
                "id": "source-grid",
                "kind": "grid",
                "role": "source",
                "rows": 6,
                "columns": 8,
                "cell_size": 42,
                "origin": (420, 34),
                "selected": ((1, 2), (2, 2), (3, 2), (4, 2)),
            },
            {
                "id": "request-vector",
                "kind": "node",
                "role": "request",
                "x": 42,
                "y": 114,
                "width": 180,
                "height": 58,
                "label": "request: length 4",
            },
            {
                "id": "selection-map",
                "kind": "arrow",
                "role": "text",
                "source": "request-vector",
                "target": "source-grid-label",
                "label": "i ↦ (i + 1, 2)",
                "label_offset": (0, -115),
            },
            {
                "id": "source-grid-label",
                "kind": "node",
                "role": "selected",
                "x": 790,
                "y": 130,
                "width": 92,
                "height": 46,
                "label": "4 cells",
            },
            {
                "id": "expression",
                "kind": "text",
                "role": "text",
                "x": 110,
                "y": 82,
                "label": "image[1:5, 2]",
            },
        ),
    },
    {
        "id": "half-open-intervals",
        "title": "Adjacent half-open intervals concatenate exactly",
        "description": (
            "adjacent [0, 1) and [1, 3) meet without a gap or overlap to form [0, 3)."
        ),
        "width": 760,
        "height": 330,
        "elements": (
            {
                "id": "left-piece",
                "kind": "node",
                "role": "request",
                "x": 54,
                "y": 48,
                "width": 250,
                "height": 64,
                "label": "[0, 1) contains 0",
            },
            {
                "id": "right-piece",
                "kind": "node",
                "role": "request",
                "x": 456,
                "y": 48,
                "width": 250,
                "height": 64,
                "label": "[1, 3) contains 1, 2",
            },
            {
                "id": "combined",
                "kind": "node",
                "role": "selected",
                "x": 230,
                "y": 190,
                "width": 300,
                "height": 68,
                "label": "[0, 3) contains 0, 1, 2",
            },
            {
                "id": "left-concatenates",
                "kind": "arrow",
                "role": "text",
                "source": "left-piece",
                "target": "combined",
                "label": "concatenate",
            },
            {
                "id": "right-concatenates",
                "kind": "arrow",
                "role": "text",
                "source": "right-piece",
                "target": "combined",
                "label": "concatenate",
            },
            {
                "id": "shared-boundary",
                "kind": "text",
                "role": "text",
                "x": 380,
                "y": 170,
                "label": "1 is excluded on the left and included on the right",
            },
        ),
    },
    {
        "id": "coordinate-addresses",
        "title": "Negative and non-negative coordinates are peer addresses",
        "description": "domain [-2, 3) gives equal status to -2, -1, 0, 1, 2.",
        "width": 720,
        "height": 250,
        "elements": tuple(
            NodeSpec(
                id=f"coordinate-{coordinate}",
                kind="node",
                role="cell-domain",
                x=80 + index * 112,
                y=62,
                width=72,
                height=62,
                label=str(coordinate),
            )
            for index, coordinate in enumerate(range(-2, 3))
        )
        + (
            TextSpec(
                id="domain-label",
                kind="text",
                role="text",
                x=120,
                y=42,
                label="cell domain [-2, 3)",
            ),
        ),
    },
    {
        "id": "prepend-chunk",
        "title": "Prepending extends the domain without moving old coordinates",
        "description": "[0, 6) becomes [-3, 6) while old coordinates stay fixed.",
        "width": 760,
        "height": 350,
        "elements": tuple(
            NodeSpec(
                id=f"before-{coordinate}",
                kind="node",
                role="source",
                x=260 + coordinate * 52,
                y=48,
                width=44,
                height=44,
                label=str(coordinate),
            )
            for coordinate in range(6)
        )
        + tuple(
            NodeSpec(
                id=f"after-{coordinate}",
                kind="node",
                role="source",
                x=104 + (coordinate + 3) * 52,
                y=156,
                width=44,
                height=44,
                label=str(coordinate),
            )
            for coordinate in range(-3, 6)
        )
        + (
            TextSpec(
                id="prepended-label",
                kind="text",
                role="selected",
                x=178,
                y=132,
                label="prepended cells",
            ),
            TextSpec(
                id="before-label",
                kind="text",
                role="text",
                x=80,
                y=75,
                label="before [0, 6)",
            ),
            TextSpec(
                id="after-label",
                kind="text",
                role="text",
                x=270,
                y=224,
                label="after [-3, 6); coordinates 0–5 stay aligned",
            ),
        ),
    },
    {
        "id": "transform-mapping",
        "title": "A request coordinate maps into source coordinates",
        "description": "request i maps to source (i + 1, 2).",
        "width": 680,
        "height": 250,
        "elements": (
            {
                "id": "request-i",
                "kind": "node",
                "role": "request",
                "x": 72,
                "y": 62,
                "width": 170,
                "height": 70,
                "label": "request i",
            },
            {
                "id": "source-coordinate",
                "kind": "node",
                "role": "source",
                "x": 438,
                "y": 62,
                "width": 190,
                "height": 70,
                "label": "source (i + 1, 2)",
            },
            {
                "id": "request-to-source",
                "kind": "arrow",
                "role": "text",
                "source": "request-i",
                "target": "source-coordinate",
                "label": "transform",
            },
        ),
    },
    {
        "id": "compose-views",
        "title": "Composed views reduce to one direct request-to-source map",
        "description": "two view maps collapse into one direct map.",
        "width": 760,
        "height": 340,
        "elements": (
            {
                "id": "request-chain",
                "kind": "node",
                "role": "request",
                "x": 34,
                "y": 42,
                "width": 128,
                "height": 52,
                "label": "request",
            },
            {
                "id": "outer-view",
                "kind": "node",
                "role": "cell-domain",
                "x": 222,
                "y": 42,
                "width": 128,
                "height": 52,
                "label": "outer view",
            },
            {
                "id": "inner-view",
                "kind": "node",
                "role": "cell-domain",
                "x": 410,
                "y": 42,
                "width": 128,
                "height": 52,
                "label": "inner view",
            },
            {
                "id": "source-chain",
                "kind": "node",
                "role": "source",
                "x": 598,
                "y": 42,
                "width": 128,
                "height": 52,
                "label": "source",
            },
            {
                "id": "map-one",
                "kind": "arrow",
                "role": "text",
                "source": "request-chain",
                "target": "outer-view",
                "label": "map 2",
            },
            {
                "id": "map-two",
                "kind": "arrow",
                "role": "text",
                "source": "outer-view",
                "target": "inner-view",
                "label": "compose",
            },
            {
                "id": "map-three",
                "kind": "arrow",
                "role": "text",
                "source": "inner-view",
                "target": "source-chain",
                "label": "map 1",
            },
            {
                "id": "request-direct",
                "kind": "node",
                "role": "request",
                "x": 128,
                "y": 164,
                "width": 150,
                "height": 56,
                "label": "request",
            },
            {
                "id": "source-direct",
                "kind": "node",
                "role": "source",
                "x": 482,
                "y": 164,
                "width": 150,
                "height": 56,
                "label": "source",
            },
            {
                "id": "direct-map",
                "kind": "arrow",
                "role": "selected",
                "source": "request-direct",
                "target": "source-direct",
                "label": "one direct map",
            },
        ),
    },
    {
        "id": "chunk-overlay",
        "title": "The selection crosses one chunk-row boundary",
        "description": (
            "touched global chunk domains map selected cells to zero-origin "
            "chunk-local coordinates."
        ),
        "width": 440,
        "height": 610,
        "elements": (
            {
                "id": "chunked-source",
                "kind": "grid",
                "role": "source",
                "rows": 6,
                "columns": 8,
                "cell_size": 42,
                "origin": (52, 140),
                "selected": ((1, 2), (2, 2), (3, 2), (4, 2)),
                "chunk_shape": (3, 4),
            },
            {
                "id": "chunk-zero-coords",
                "kind": "text",
                "role": "source",
                "x": 220,
                "y": 35,
                "label": "global chunk_coords (0, 0)",
            },
            {
                "id": "chunk-zero-domain",
                "kind": "text",
                "role": "source",
                "x": 220,
                "y": 75,
                "label": "global chunk_domain [0, 3) × [0, 4)",
            },
            {
                "id": "chunk-zero-local",
                "kind": "text",
                "role": "chunk-local",
                "x": 220,
                "y": 115,
                "label": "chunk-local selected (1, 2), (2, 2)",
            },
            {
                "id": "chunk-one-coords",
                "kind": "text",
                "role": "source",
                "x": 220,
                "y": 425,
                "label": "global chunk_coords (1, 0)",
            },
            {
                "id": "chunk-one-domain",
                "kind": "text",
                "role": "source",
                "x": 220,
                "y": 465,
                "label": "global chunk_domain [3, 6) × [0, 4)",
            },
            {
                "id": "chunk-one-local",
                "kind": "text",
                "role": "chunk-local",
                "x": 220,
                "y": 505,
                "label": "chunk-local selected (0, 2), (1, 2)",
            },
            {
                "id": "selection-label",
                "kind": "text",
                "role": "selected",
                "x": 220,
                "y": 545,
                "label": "selected global cells",
            },
        ),
    },
    {
        "id": "projection-pair",
        "title": "A cell domain projects into two coordinate spaces",
        "description": "one cell domain points to request and chunk-local spaces.",
        "width": 720,
        "height": 330,
        "elements": (
            {
                "id": "domain",
                "kind": "node",
                "role": "cell-domain",
                "x": 70,
                "y": 104,
                "width": 180,
                "height": 68,
                "label": "cell domain",
            },
            {
                "id": "request-space",
                "kind": "node",
                "role": "request",
                "x": 462,
                "y": 42,
                "width": 188,
                "height": 60,
                "label": "request space",
            },
            {
                "id": "chunk-space",
                "kind": "node",
                "role": "chunk-local",
                "x": 462,
                "y": 174,
                "width": 188,
                "height": 60,
                "label": "chunk-local space",
            },
            {
                "id": "request-projection",
                "kind": "arrow",
                "role": "text",
                "source": "domain",
                "target": "request-space",
                "label": "request projection",
            },
            {
                "id": "chunk-projection",
                "kind": "arrow",
                "role": "text",
                "source": "domain",
                "target": "chunk-space",
                "label": "chunk projection",
            },
        ),
    },
    {
        "id": "orthogonal-contrast",
        "title": "Orthogonal row indexing preserves order and duplicates",
        "description": "rows [4, 1, 1] retain order and duplication.",
        "width": 740,
        "height": 380,
        "elements": (
            {
                "id": "row-request",
                "kind": "node",
                "role": "request",
                "x": 52,
                "y": 94,
                "width": 196,
                "height": 64,
                "label": "rows [4, 1, 1]",
            },
            {
                "id": "result-four",
                "kind": "node",
                "role": "selected",
                "x": 470,
                "y": 34,
                "width": 160,
                "height": 48,
                "label": "result[0] = row 4",
            },
            {
                "id": "result-one-a",
                "kind": "node",
                "role": "selected",
                "x": 470,
                "y": 104,
                "width": 160,
                "height": 48,
                "label": "result[1] = row 1",
            },
            {
                "id": "result-one-b",
                "kind": "node",
                "role": "selected",
                "x": 470,
                "y": 174,
                "width": 160,
                "height": 48,
                "label": "result[2] = row 1",
            },
            {
                "id": "select-four",
                "kind": "arrow",
                "role": "text",
                "source": "row-request",
                "target": "result-four",
                "label": "first",
            },
            {
                "id": "select-one-a",
                "kind": "arrow",
                "role": "text",
                "source": "row-request",
                "target": "result-one-a",
                "label": "second",
            },
            {
                "id": "select-one-b",
                "kind": "arrow",
                "role": "text",
                "source": "row-request",
                "target": "result-one-b",
                "label": "third; duplicate retained",
            },
        ),
    },
)

_SEMANTIC_ROLES: frozenset[str] = frozenset(
    {"request", "source", "selected", "chunk-local", "cell-domain", "text"}
)
_ELEMENT_KINDS: frozenset[str] = frozenset({"grid", "axis", "node", "arrow", "text"})
_SAFE_FRAGMENT_ID_PATTERN = r"[A-Za-z][A-Za-z0-9_-]*"


def _invalid(figure_id: str, field: str, detail: str) -> Never:
    raise ValueError(f"{figure_id}: {field} {detail}")


def _positive(figure_id: str, element_id: str, element: dict[str, object], field: str) -> None:
    value = element.get(field)
    if not _is_integer(value) or value <= 0:
        _invalid(figure_id, f"{element_id}.{field}", "must be a positive integer")


def validate_figure(spec: FigureSpec) -> None:
    """Validate a figure before rendering it.

    Figure and element IDs must match ``[A-Za-z][A-Za-z0-9_-]*`` so every
    generated XML ID is one whitespace-free fragment token.

    Raises
    ------
    ValueError
        If the specification cannot be rendered unambiguously.
    """
    raw_spec = dict(spec)
    figure_id = raw_spec.get("id", "<missing-id>")
    if not isinstance(figure_id, str) or not figure_id:
        _invalid("<missing-id>", "id", "must be a non-empty string")
    if re.fullmatch(_SAFE_FRAGMENT_ID_PATTERN, figure_id) is None:
        _invalid(figure_id, "id", f"must match {_SAFE_FRAGMENT_ID_PATTERN}")
    for field in ("title", "description"):
        value = raw_spec.get(field)
        if not isinstance(value, str) or not value.strip():
            _invalid(figure_id, field, "must be a non-empty string")
    for field in ("width", "height"):
        value = raw_spec.get(field)
        if not _is_integer(value) or value <= 0:
            _invalid(figure_id, field, "must be a positive integer")

    element_ids: set[str] = set()
    node_ids: set[str] = set()
    arrows: list[dict[str, object]] = []
    elements = raw_spec.get("elements")
    if not isinstance(elements, tuple):
        _invalid(figure_id, "elements", "must be a tuple")
    for raw_element in elements:
        if not isinstance(raw_element, dict):
            _invalid(figure_id, "elements", "must contain element mappings")
        element: dict[str, object] = raw_element
        element_id = element.get("id")
        if not isinstance(element_id, str) or not element_id:
            _invalid(figure_id, "elements.id", "must be a non-empty string")
        if re.fullmatch(_SAFE_FRAGMENT_ID_PATTERN, element_id) is None:
            _invalid(figure_id, "elements.id", f"must match {_SAFE_FRAGMENT_ID_PATTERN}")
        if element_id in element_ids:
            _invalid(figure_id, element_id, "is a duplicate element ID")
        element_ids.add(element_id)

        role = element.get("role")
        if not isinstance(role, str) or role not in _SEMANTIC_ROLES:
            _invalid(figure_id, f"{element_id}.role", "must be a known semantic role")
        kind = element.get("kind")
        if not isinstance(kind, str) or kind not in _ELEMENT_KINDS:
            _invalid(figure_id, f"{element_id}.kind", "must be a known element kind")

        if kind == "grid":
            _validate_grid(figure_id, element_id, element)
        elif kind == "axis":
            _validate_axis(figure_id, element_id, element)
        elif kind == "node":
            _validate_node(figure_id, element_id, element)
            node_ids.add(element_id)
        elif kind == "arrow":
            _validate_label(figure_id, element_id, element)
            if "label_offset" in element and not _pair_of_ints(element["label_offset"]):
                _invalid(
                    figure_id,
                    f"{element_id}.label_offset",
                    "must be a pair of integers",
                )
            arrows.append(element)
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
    assert _is_integer(rows)
    assert _is_integer(columns)
    assert isinstance(selected, tuple)
    for coordinate in selected:
        assert _pair_of_ints(coordinate)
        row, column = coordinate
        assert _is_integer(row)
        assert _is_integer(column)
        if not 0 <= row < rows or not 0 <= column < columns:
            _invalid(figure_id, f"{element_id}.selected", "contains a coordinate outside the grid")
    chunk_shape = element.get("chunk_shape")
    if chunk_shape is not None:
        if not _pair_of_ints(chunk_shape):
            _invalid(figure_id, f"{element_id}.chunk_shape", "must be positive dimensions")
        first, second = chunk_shape
        assert _is_integer(first)
        assert _is_integer(second)
        if first <= 0 or second <= 0:
            _invalid(figure_id, f"{element_id}.chunk_shape", "must be positive dimensions")


def _validate_axis(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    _positive(figure_id, element_id, element, "length")
    for field in ("x", "y", "start", "stop"):
        if not _is_integer(element.get(field)):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    if element.get("orientation") not in {"horizontal", "vertical"}:
        _invalid(figure_id, f"{element_id}.orientation", "must be horizontal or vertical")
    start = element.get("start")
    stop = element.get("stop")
    assert _is_integer(start)
    assert _is_integer(stop)
    if stop < start:
        _invalid(figure_id, f"{element_id}.stop", "must not precede start")


def _validate_node(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    for field in ("x", "y"):
        if not _is_integer(element.get(field)):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    for field in ("width", "height"):
        _positive(figure_id, element_id, element, field)
    _validate_label(figure_id, element_id, element)


def _validate_text(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    for field in ("x", "y"):
        if not _is_integer(element.get(field)):
            _invalid(figure_id, f"{element_id}.{field}", "must be an integer")
    _validate_label(figure_id, element_id, element)


def _validate_label(figure_id: str, element_id: str, element: dict[str, object]) -> None:
    if not isinstance(element.get("label"), str):
        _invalid(figure_id, f"{element_id}.label", "must be a string")


def _pair_of_ints(value: object) -> TypeGuard[tuple[int, int]]:
    return isinstance(value, tuple) and len(value) == 2 and all(_is_integer(item) for item in value)


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
