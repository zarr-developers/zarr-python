"""JSON serialization for index transforms.

Defines TypedDict types matching TensorStore's JSON representation of
IndexTransform and IndexDomain, plus conversion functions.

The JSON format follows TensorStore's conventions for interoperability::

    {
      "input_inclusive_min": [0, 0],
      "input_exclusive_max": [100, 200],
      "input_labels": ["x", "y"],
      "output": [
        {"offset": 5},
        {"offset": 10, "stride": 2, "input_dimension": 1},
        {"offset": 0, "stride": 1, "index_array": [[1, 2, 0]]}
      ]
    }
"""

from __future__ import annotations

from typing import Required, TypedDict

import numpy as np

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr.core.transforms.transform import IndexTransform

# ---------------------------------------------------------------------------
# TypedDict definitions (JSON shapes)
# ---------------------------------------------------------------------------


class IndexDomainJSON(TypedDict, total=False):
    """JSON representation of an IndexDomain."""

    input_inclusive_min: Required[list[int]]
    input_exclusive_max: Required[list[int]]
    input_labels: list[str]


class OutputIndexMapJSON(TypedDict, total=False):
    """JSON representation of a single output index map.

    Exactly one of three forms:
    - ``{"offset": 5}`` — constant
    - ``{"offset": 0, "stride": 1, "input_dimension": 0}`` — dimension
    - ``{"offset": 0, "stride": 1, "index_array": [...]}`` — array
    """

    offset: int
    stride: int
    input_dimension: int
    index_array: list[int] | list[list[int]]


class IndexTransformJSON(TypedDict, total=False):
    """JSON representation of an IndexTransform."""

    input_inclusive_min: Required[list[int]]
    input_exclusive_max: Required[list[int]]
    input_labels: list[str]
    output: Required[list[OutputIndexMapJSON]]


# ---------------------------------------------------------------------------
# IndexDomain serialization
# ---------------------------------------------------------------------------


def index_domain_to_json(domain: IndexDomain) -> IndexDomainJSON:
    """Convert an IndexDomain to its JSON representation."""
    result: IndexDomainJSON = {
        "input_inclusive_min": list(domain.inclusive_min),
        "input_exclusive_max": list(domain.exclusive_max),
    }
    if domain.labels is not None:
        result["input_labels"] = list(domain.labels)
    return result


def index_domain_from_json(data: IndexDomainJSON) -> IndexDomain:
    """Construct an IndexDomain from its JSON representation."""
    return IndexDomain(
        inclusive_min=tuple(data["input_inclusive_min"]),
        exclusive_max=tuple(data["input_exclusive_max"]),
        labels=tuple(data["input_labels"]) if "input_labels" in data else None,
    )


# ---------------------------------------------------------------------------
# OutputIndexMap serialization
# ---------------------------------------------------------------------------


def output_index_map_to_json(m: OutputIndexMap) -> OutputIndexMapJSON:
    """Convert an output index map to its JSON representation."""
    if isinstance(m, ConstantMap):
        result: OutputIndexMapJSON = {"offset": m.offset}
        return result

    if isinstance(m, DimensionMap):
        result = {"offset": m.offset, "input_dimension": m.input_dimension}
        if m.stride != 1:
            result["stride"] = m.stride
        return result

    if isinstance(m, ArrayMap):
        result = {"offset": m.offset, "index_array": m.index_array.tolist()}
        if m.stride != 1:
            result["stride"] = m.stride
        return result

    raise TypeError(f"Unknown output map type: {type(m)}")


def output_index_map_from_json(data: OutputIndexMapJSON) -> OutputIndexMap:
    """Construct an output index map from its JSON representation."""
    if "index_array" in data:
        return ArrayMap(
            index_array=np.asarray(data["index_array"], dtype=np.intp),
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )

    if "input_dimension" in data:
        return DimensionMap(
            input_dimension=data["input_dimension"],
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )

    # Constant map: only offset present
    return ConstantMap(offset=data.get("offset", 0))


# ---------------------------------------------------------------------------
# IndexTransform serialization
# ---------------------------------------------------------------------------


def index_transform_to_json(transform: IndexTransform) -> IndexTransformJSON:
    """Convert an IndexTransform to its JSON representation."""
    result: IndexTransformJSON = {
        "input_inclusive_min": list(transform.domain.inclusive_min),
        "input_exclusive_max": list(transform.domain.exclusive_max),
        "output": [output_index_map_to_json(m) for m in transform.output],
    }
    if transform.domain.labels is not None:
        result["input_labels"] = list(transform.domain.labels)
    return result


def index_transform_from_json(data: IndexTransformJSON) -> IndexTransform:
    """Construct an IndexTransform from its JSON representation."""
    domain = IndexDomain(
        inclusive_min=tuple(data["input_inclusive_min"]),
        exclusive_max=tuple(data["input_exclusive_max"]),
        labels=tuple(data["input_labels"]) if "input_labels" in data else None,
    )
    output = tuple(output_index_map_from_json(m) for m in data["output"])
    return IndexTransform(domain=domain, output=output)
