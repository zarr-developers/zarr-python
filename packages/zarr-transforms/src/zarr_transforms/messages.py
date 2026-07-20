"""The ndsel message layer — pure JSON in, canonical JSON out.

This module implements the [ndsel](https://github.com/d-v-b/ndsel) draft wire
format: a JSON-serializable representation of NumPy-style n-dimensional
selections that adapts TensorStore's `IndexTransform` model. It is a **pure
JSON→JSON** layer: it depends on nothing but the standard library, imposes no
engine (numpy/array) constraints, and never rounds, clamps, or drops
information. Engine constraints (finite bounds, in-memory `IndexTransform`
construction) live one layer up, in `json.py`.

Two entry points:

- `parse_ndsel(obj)` — structurally validate an ndsel message of any of the
  five kinds (`point`/`box`/`slice`/`points`/`transform`), returning it
  unchanged. Raises `NdselError` (carrying a spec reason code) on any defect.
- `normalize_ndsel(obj)` — desugar and canonicalize a message to the single
  deterministic **canonical transform body** of the spec (section 4.3): a bare
  `IndexTransform` JSON body, without the `kind` discriminator. `normalize` is
  idempotent when its output is re-tagged with `kind: "transform"`.

The canonical body is, field-for-field, a TensorStore `IndexTransform` (minus
`kind`), so a normalized `transform` loads directly into TensorStore once
`kind` is stripped.

Value rules enforced here: every integer is a 64-bit signed value; JSON
booleans are **not** integers (Python's `isinstance(True, int)` is guarded
against explicitly); the `"-inf"`/`"+inf"` sentinels are legal only in bound
positions; an implicit bound is the one-element `[n]`-bracket form, and its
implicit/explicit flag is preserved through normalization.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NdselError",
    "normalize_ndsel",
    "parse_ndsel",
]

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

#: The complete set of ndsel reason codes (spec section 6).
REASON_CODES = frozenset(
    {
        "invalid_json",
        "unknown_kind",
        "unknown_field",
        "multiple_upper_bounds",
        "bounds_out_of_order",
        "output_map_conflict",
        "rank_mismatch",
        "step_zero",
        "negative_step_unsupported",
    }
)


class NdselError(ValueError):
    """An ndsel message failed validation.

    Carries the spec `reason` code (one of `REASON_CODES`) so callers and the
    conformance harness can assert on it directly, plus a human-readable
    `detail`.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# 64-bit signed integer range (spec section 3.5)
# ---------------------------------------------------------------------------

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

_KNOWN_KINDS = frozenset({"point", "box", "slice", "points", "transform"})

# The two upper-bound spellings, keyed by message prefix. Only one of the three
# per group may appear (spec section 4.1 / 5.2).
_BOX_UPPER = ("exclusive_max", "inclusive_max", "shape")
_TRANSFORM_UPPER = ("input_exclusive_max", "input_inclusive_max", "input_shape")

_OUTPUT_MAP_FIELDS = frozenset(
    {"offset", "stride", "input_dimension", "index_array", "index_array_bounds"}
)


# ---------------------------------------------------------------------------
# Leaf value validators
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    """True iff `value` is a JSON integer — an `int` that is not a `bool`.

    JSON has no boolean-as-integer: `True`/`False` are rejected even though
    Python makes `bool` a subclass of `int` (spec section 3.6).
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int(value: Any, where: str) -> int:
    """Validate a plain-integer position: an in-range i64, never a sentinel."""
    if not _is_int(value):
        raise NdselError("invalid_json", f"{where} must be an integer, got {value!r}")
    if value < _I64_MIN or value > _I64_MAX:
        raise NdselError("invalid_json", f"{where} is outside the 64-bit signed range: {value}")
    return int(value)


def _is_sentinel(value: Any) -> bool:
    return value in ("-inf", "+inf")


def _check_index_value(value: Any, where: str) -> int | str:
    """Validate an `index-value`: an in-range i64 or a `"-inf"`/`"+inf"` sentinel."""
    if _is_sentinel(value):
        return str(value)
    return _check_int(value, where)


def _check_bound(value: Any, where: str) -> int | str | list[int | str]:
    """Validate a `bound`: an explicit `index-value`, or a one-element implicit `[index-value]`."""
    if isinstance(value, list):
        if len(value) != 1:
            raise NdselError(
                "invalid_json",
                f"{where} implicit bound must be a one-element array, got {value!r}",
            )
        return [_check_index_value(value[0], where)]
    return _check_index_value(value, where)


def _check_int_list(value: Any, where: str) -> list[int]:
    if not isinstance(value, list):
        raise NdselError("invalid_json", f"{where} must be an array, got {value!r}")
    return [_check_int(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _check_bound_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise NdselError("invalid_json", f"{where} must be an array, got {value!r}")
    return [_check_bound(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _check_label_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list):
        raise NdselError("invalid_json", f"{where} must be an array, got {value!r}")
    for i, v in enumerate(value):
        if not isinstance(v, str):
            raise NdselError("invalid_json", f"{where}[{i}] must be a string, got {v!r}")
    return list(value)


# ---------------------------------------------------------------------------
# Extended-integer order for bounds (spec section 4.1)
# ---------------------------------------------------------------------------


def _bound_value(bound: int | str | list[int | str]) -> int | str:
    """The underlying `index-value` of a bound, dropping the implicit bracket."""
    return bound[0] if isinstance(bound, list) else bound


def _bound_is_implicit(bound: int | str | list[int | str]) -> bool:
    return isinstance(bound, list)


def _ext_key(value: int | str) -> tuple[int, int]:
    """A sort key giving the extended-integer order `-inf < n < +inf` exactly.

    Uses an integer tier plus the value, so no float rounding of near-`2**63`
    integers can misorder the `inclusive_min <= exclusive_max` check.
    """
    if value == "-inf":
        return (0, 0)
    if value == "+inf":
        return (2, 0)
    assert isinstance(value, int)
    return (1, value)


def _rewrap(value: int | str, *, implicit: bool) -> int | str | list[int | str]:
    return [value] if implicit else value


# ---------------------------------------------------------------------------
# Message-level helpers
# ---------------------------------------------------------------------------


def _require_object(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise NdselError("invalid_json", f"message must be a JSON object, got {type(obj).__name__}")
    return obj


def _message_kind(obj: dict[str, Any]) -> str:
    kind = obj.get("kind")
    if not isinstance(kind, str):
        raise NdselError("invalid_json", "message must have a string 'kind' field")
    if kind not in _KNOWN_KINDS:
        raise NdselError("unknown_kind", f"unknown kind {kind!r}")
    return kind


def _check_membership(obj: dict[str, Any], allowed: frozenset[str], what: str) -> None:
    """Strict membership (spec section 3.7): reject any undefined member."""
    for key in obj:
        if key not in allowed:
            raise NdselError("unknown_field", f"{what} has undefined member {key!r}")


def _single_upper_bound(obj: dict[str, Any], fields: tuple[str, str, str]) -> str | None:
    present = [f for f in fields if f in obj]
    if len(present) > 1:
        raise NdselError(
            "multiple_upper_bounds",
            f"at most one of {fields} may be present; got {present}",
        )
    return present[0] if present else None


def _resolve_upper_bound(
    upper_field: str | None,
    upper_raw: list[Any] | None,
    inclusive_min: list[Any],
    rank: int,
    *,
    kind_of: str,
) -> list[int | str | list[int | str]]:
    """Produce `exclusive_max` from whichever upper-bound spelling was supplied.

    - `exclusive_max`/`input_exclusive_max` → used directly.
    - `inclusive_max`/`input_inclusive_max` → each element `+1`.
    - `shape`/`input_shape` → `inclusive_min + shape` per element.
    - none → an **implicit `+inf`** in every dimension.

    The implicit/explicit bracket travels with the extent-bearing field (the
    upper bound, or `shape`), matching the spec's `[n]`-bracket convention.
    """
    if upper_field is None:
        return [["+inf"] for _ in range(rank)]

    assert upper_raw is not None
    if kind_of == "exclusive":
        return list(upper_raw)

    result: list[int | str | list[int | str]] = []
    for k in range(rank):
        raw = upper_raw[k]
        implicit = _bound_is_implicit(raw)
        value = _bound_value(raw)
        if kind_of == "inclusive":
            new = _inclusive_to_exclusive(value)
        else:  # shape
            new = _shape_to_exclusive(_bound_value(inclusive_min[k]), value)
        result.append(_rewrap(new, implicit=implicit))
    return result


def _inclusive_to_exclusive(value: int | str) -> int | str:
    if value == "+inf" or value == "-inf":
        return value
    assert isinstance(value, int)
    return value + 1


def _shape_to_exclusive(min_value: int | str, shape_value: int | str) -> int | str:
    if shape_value == "+inf" or min_value == "+inf":
        return "+inf"
    if min_value == "-inf":
        return "-inf"
    assert isinstance(min_value, int)
    assert isinstance(shape_value, int)
    return min_value + shape_value


def _validate_domain(inclusive_min: list[Any], exclusive_max: list[Any], *, prefix: str) -> None:
    """Every dimension must satisfy `inclusive_min <= exclusive_max` (empty is valid)."""
    for k, (lo, hi) in enumerate(zip(inclusive_min, exclusive_max, strict=True)):
        if _ext_key(_bound_value(lo)) > _ext_key(_bound_value(hi)):
            raise NdselError(
                "bounds_out_of_order",
                f"{prefix}[{k}]: inclusive_min {_bound_value(lo)!r} > "
                f"exclusive_max {_bound_value(hi)!r}",
            )


def _identity_output(rank: int) -> list[dict[str, Any]]:
    return [{"offset": 0, "stride": 1, "input_dimension": k} for k in range(rank)]


# ---------------------------------------------------------------------------
# Per-kind desugaring
# ---------------------------------------------------------------------------


def _normalize_point(obj: dict[str, Any]) -> dict[str, Any]:
    _check_membership(obj, frozenset({"kind", "coords"}), "point")
    if "coords" not in obj:
        raise NdselError("invalid_json", "point requires 'coords'")
    coords = _check_int_list(obj["coords"], "coords")
    return {
        "input_rank": 0,
        "input_inclusive_min": [],
        "input_exclusive_max": [],
        "input_labels": [],
        "output": [{"offset": c} for c in coords],
    }


def _infer_rank(
    obj: dict[str, Any],
    named_lengths: list[tuple[str, int]],
    *,
    declared: int | None,
) -> int:
    """Reconcile a declared rank (if any) with every present array's length."""
    rank = declared
    for name, length in named_lengths:
        if rank is None:
            rank = length
        elif rank != length:
            raise NdselError(
                "rank_mismatch",
                f"{name} has length {length}, inconsistent with rank {rank}",
            )
    return rank if rank is not None else 0


def _normalize_box(obj: dict[str, Any]) -> dict[str, Any]:
    allowed = frozenset(
        {"kind", "inclusive_min", "exclusive_max", "inclusive_max", "shape", "labels"}
    )
    _check_membership(obj, allowed, "box")

    inclusive_min_raw = (
        _check_bound_list(obj["inclusive_min"], "inclusive_min") if "inclusive_min" in obj else None
    )
    upper_field = _single_upper_bound(obj, _BOX_UPPER)
    upper_raw = _check_bound_list(obj[upper_field], upper_field) if upper_field else None
    labels_raw = _check_label_list(obj["labels"], "labels") if "labels" in obj else None

    named_lengths: list[tuple[str, int]] = []
    if inclusive_min_raw is not None:
        named_lengths.append(("inclusive_min", len(inclusive_min_raw)))
    if upper_raw is not None:
        named_lengths.append((upper_field or "", len(upper_raw)))
    if labels_raw is not None:
        named_lengths.append(("labels", len(labels_raw)))
    rank = _infer_rank(obj, named_lengths, declared=None)

    inclusive_min = inclusive_min_raw if inclusive_min_raw is not None else [0] * rank
    exclusive_max = _resolve_upper_bound(
        upper_field, upper_raw, inclusive_min, rank, kind_of=_upper_kind(upper_field, _BOX_UPPER)
    )
    labels = labels_raw if labels_raw is not None else [""] * rank
    _validate_domain(inclusive_min, exclusive_max, prefix="box")

    return {
        "input_rank": rank,
        "input_inclusive_min": inclusive_min,
        "input_exclusive_max": exclusive_max,
        "input_labels": labels,
        "output": _identity_output(rank),
    }


def _upper_kind(upper_field: str | None, fields: tuple[str, str, str]) -> str:
    if upper_field is None or upper_field == fields[0]:
        return "exclusive"
    if upper_field == fields[1]:
        return "inclusive"
    return "shape"


def _normalize_slice(obj: dict[str, Any]) -> dict[str, Any]:
    allowed = frozenset({"kind", "start", "stop", "step", "labels"})
    _check_membership(obj, allowed, "slice")
    if "start" not in obj:
        raise NdselError("invalid_json", "slice requires 'start'")
    if "stop" not in obj:
        raise NdselError("invalid_json", "slice requires 'stop'")
    start = _check_int_list(obj["start"], "start")
    stop = _check_int_list(obj["stop"], "stop")
    step = _check_int_list(obj["step"], "step") if "step" in obj else [1] * len(start)
    labels_raw = _check_label_list(obj["labels"], "labels") if "labels" in obj else None

    n = len(start)
    for name, arr in (("stop", stop), ("step", step)):
        if len(arr) != n:
            raise NdselError(
                "rank_mismatch", f"{name} has length {len(arr)}, expected {n} (from start)"
            )
    if labels_raw is not None and len(labels_raw) != n:
        raise NdselError(
            "rank_mismatch", f"labels has length {len(labels_raw)}, expected {n} (from start)"
        )

    for k, s in enumerate(step):
        if s == 0:
            raise NdselError("step_zero", f"step[{k}] is zero")
        if s < 0:
            raise NdselError("negative_step_unsupported", f"step[{k}] is negative ({s})")

    inclusive_min: list[Any] = []
    exclusive_max: list[Any] = []
    output: list[dict[str, Any]] = []
    for k in range(n):
        a, b, s = start[k], stop[k], step[k]
        m = max(0, -(-(b - a) // s))  # ceil((b - a) / s)
        o = _trunc_div(a, s)  # trunc(a / s), toward zero
        offset = a - s * o  # lattice phase, in (-s, s)
        inclusive_min.append(o)
        exclusive_max.append(o + m)
        output.append({"offset": offset, "stride": s, "input_dimension": k})

    labels = labels_raw if labels_raw is not None else [""] * n
    return {
        "input_rank": n,
        "input_inclusive_min": inclusive_min,
        "input_exclusive_max": exclusive_max,
        "input_labels": labels,
        "output": output,
    }


def _normalize_points(obj: dict[str, Any]) -> dict[str, Any]:
    _check_membership(obj, frozenset({"kind", "coords"}), "points")
    if "coords" not in obj:
        raise NdselError("invalid_json", "points requires 'coords'")
    coords = obj["coords"]
    if not isinstance(coords, list):
        raise NdselError("invalid_json", f"points coords must be an array, got {coords!r}")

    rows: list[list[int]] = []
    n: int | None = None
    for i, row in enumerate(coords):
        if not isinstance(row, list):
            raise NdselError("invalid_json", f"points coords[{i}] must be an array, got {row!r}")
        row_ints = [_check_int(v, f"coords[{i}][{j}]") for j, v in enumerate(row)]
        if n is None:
            n = len(row_ints)
        elif len(row_ints) != n:
            raise NdselError(
                "rank_mismatch",
                f"points coords[{i}] has length {len(row_ints)}, expected {n} (ragged)",
            )
        rows.append(row_ints)

    m = len(rows)
    n = n if n is not None else 0
    output = [
        {
            "offset": 0,
            "stride": 1,
            "index_array": [rows[i][k] for i in range(m)],
            "index_array_bounds": ["-inf", "+inf"],
        }
        for k in range(n)
    ]
    return {
        "input_rank": 1,
        "input_inclusive_min": [0],
        "input_exclusive_max": [m],
        "input_labels": [""],
        "output": output,
    }


def _normalize_output_map(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise NdselError("invalid_json", f"{where} must be a JSON object, got {raw!r}")
    _check_membership(raw, _OUTPUT_MAP_FIELDS, where)

    has_index_array = "index_array" in raw
    has_input_dim = "input_dimension" in raw
    if has_index_array and has_input_dim:
        raise NdselError(
            "output_map_conflict",
            f"{where} carries both 'input_dimension' and 'index_array'",
        )

    offset = _check_int(raw["offset"], f"{where}.offset") if "offset" in raw else 0

    if has_index_array:
        stride = _check_int(raw["stride"], f"{where}.stride") if "stride" in raw else 1
        bounds = (
            _check_index_array_bounds(raw["index_array_bounds"], where)
            if "index_array_bounds" in raw
            else ["-inf", "+inf"]
        )
        # index_array is carried verbatim (spec section 7 defers shape validation).
        return {
            "offset": offset,
            "stride": stride,
            "index_array": raw["index_array"],
            "index_array_bounds": bounds,
        }

    if has_input_dim:
        input_dim = _check_int(raw["input_dimension"], f"{where}.input_dimension")
        if input_dim < 0:
            raise NdselError(
                "invalid_json", f"{where}.input_dimension must be >= 0, got {input_dim}"
            )
        stride = _check_int(raw["stride"], f"{where}.stride") if "stride" in raw else 1
        return {"offset": offset, "stride": stride, "input_dimension": input_dim}

    # Constant map: only offset survives. A stray `stride`/`index_array_bounds`
    # is schema-valid (the output-map schema permits those members on any map),
    # so it is silently dropped rather than rejected — a constant carries only
    # `offset` in canonical form (spec section 4.3).
    return {"offset": offset}


def _check_index_array_bounds(value: Any, where: str) -> list[int | str]:
    if not isinstance(value, list) or len(value) != 2:
        raise NdselError(
            "invalid_json",
            f"{where}.index_array_bounds must be a two-element array, got {value!r}",
        )
    return [
        _check_index_value(value[0], f"{where}.index_array_bounds[0]"),
        _check_index_value(value[1], f"{where}.index_array_bounds[1]"),
    ]


def _normalize_transform(obj: dict[str, Any]) -> dict[str, Any]:
    allowed = frozenset(
        {
            "kind",
            "input_rank",
            "input_inclusive_min",
            "input_exclusive_max",
            "input_inclusive_max",
            "input_shape",
            "input_labels",
            "output",
        }
    )
    _check_membership(obj, allowed, "transform")

    declared_rank: int | None = None
    if "input_rank" in obj:
        declared_rank = _check_int(obj["input_rank"], "input_rank")
        if declared_rank < 0:
            raise NdselError("invalid_json", f"input_rank must be >= 0, got {declared_rank}")

    inclusive_min_raw = (
        _check_bound_list(obj["input_inclusive_min"], "input_inclusive_min")
        if "input_inclusive_min" in obj
        else None
    )
    upper_field = _single_upper_bound(obj, _TRANSFORM_UPPER)
    upper_raw = _check_bound_list(obj[upper_field], upper_field) if upper_field else None
    labels_raw = (
        _check_label_list(obj["input_labels"], "input_labels") if "input_labels" in obj else None
    )

    named_lengths: list[tuple[str, int]] = []
    if inclusive_min_raw is not None:
        named_lengths.append(("input_inclusive_min", len(inclusive_min_raw)))
    if upper_raw is not None:
        named_lengths.append((upper_field or "", len(upper_raw)))
    if labels_raw is not None:
        named_lengths.append(("input_labels", len(labels_raw)))
    rank = _infer_rank(obj, named_lengths, declared=declared_rank)

    inclusive_min = inclusive_min_raw if inclusive_min_raw is not None else [0] * rank
    exclusive_max = _resolve_upper_bound(
        upper_field,
        upper_raw,
        inclusive_min,
        rank,
        kind_of=_upper_kind(upper_field, _TRANSFORM_UPPER),
    )
    labels = labels_raw if labels_raw is not None else [""] * rank
    _validate_domain(inclusive_min, exclusive_max, prefix="input")

    if "output" in obj:
        if not isinstance(obj["output"], list):
            raise NdselError("invalid_json", f"output must be an array, got {obj['output']!r}")
        output = [_normalize_output_map(m, f"output[{i}]") for i, m in enumerate(obj["output"])]
    else:
        output = _identity_output(rank)

    return {
        "input_rank": rank,
        "input_inclusive_min": inclusive_min,
        "input_exclusive_max": exclusive_max,
        "input_labels": labels,
        "output": output,
    }


_NORMALIZERS = {
    "point": _normalize_point,
    "box": _normalize_box,
    "slice": _normalize_slice,
    "points": _normalize_points,
    "transform": _normalize_transform,
}


# ---------------------------------------------------------------------------
# trunc division (spec section 5.3 correction, matches _trunc_div in transform.py)
# ---------------------------------------------------------------------------


def _trunc_div(a: int, b: int) -> int:
    """Integer division rounded toward zero (C semantics)."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def normalize_ndsel(obj: Any) -> dict[str, Any]:
    """Desugar and canonicalize an ndsel message to its canonical transform body.

    Accepts any of the five message kinds and returns the bare canonical
    `IndexTransform` body of spec section 4.3 — no `kind` field. Raises
    `NdselError` (carrying a reason code) for any invalid input.
    """
    message = _require_object(obj)
    kind = _message_kind(message)
    return _NORMALIZERS[kind](message)


def parse_ndsel(obj: Any) -> dict[str, Any]:
    """Structurally validate an ndsel message, returning it unchanged.

    A lighter gate than `normalize_ndsel`: it confirms the message is a
    well-formed ndsel message of a recognized kind (correct field membership,
    JSON types, upper-bound exclusivity, domain ordering, step signs) and
    raises `NdselError` otherwise, but does not desugar it. Useful for
    validating a message you intend to keep in its compact shorthand form.
    """
    message = _require_object(obj)
    _message_kind(message)
    # Validation and desugaring share one pass; run it and discard the body.
    normalize_ndsel(message)
    return message
