"""Properties every reader's problems hold, over valid documents changed at random.

Each test starts from valid documents of every kind a reader reads, and
changes them in a few places: a value replaced with any JSON value or with
a name a document might write, a member removed, a member added, a key
renamed. Whatever the reader makes of the result, its problems must be
located in the document it was given, and none may be reported twice. A
tool places a diagnostic at a problem's `loc`, so a location the document
does not have is a diagnostic nobody can place.

The other tests pin one rule each, where an earlier release broke it: a
typestr is one token, so each of its problems is a wrong value of the
dtype, never an unknown or missing key; a float for an integer `Literal`
is the wrong type wherever the `Literal` sits; a member nothing claims
may hold any number in both formats; and both formats refuse the same
node paths.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from zarr_metadata import UNSET
from zarr_metadata._json import ValidationProblem, value_at
from zarr_metadata.definition import RefusedField
from zarr_metadata.model import (
    read_array_metadata_v2,
    read_array_metadata_v3,
    read_consolidated_metadata_v2,
    read_consolidated_metadata_v3,
    read_group_metadata_v2,
    read_group_metadata_v3,
    read_node_metadata_v2,
)

_PROPERTY = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])

_V2_ARRAY: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [4, 4],
    "chunks": [2, 2],
    "dtype": "<f4",
    "compressor": {"id": "zlib", "level": 1},
    "fill_value": 0,
    "order": "C",
    "filters": [{"id": "delta", "dtype": "<f4"}],
    "dimension_separator": "/",
}
_V2_STRUCT: dict[str, Any] = {
    **_V2_ARRAY,
    "dtype": [["a", "<i4"], ["b", [["c", "|u1"], ["d", "<f8", [2]]]], ["e", "<M8[ns]"]],
    "fill_value": None,
    "filters": None,
}
_V3_ARRAY: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4, 4],
    "data_type": "int16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "fill_value": 0,
    "codecs": [
        {"name": "transpose", "configuration": {"order": [1, 0]}},
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 1}},
    ],
    "attributes": {"a": 1},
}
_V3_SHARDED: dict[str, Any] = {
    **_V3_ARRAY,
    "data_type": {
        "name": "struct",
        "configuration": {"fields": [{"name": "a", "data_type": "int16"}]},
    },
    "fill_value": {"a": 0},
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [1, 1],
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    "crc32c",
                ],
            },
        }
    ],
}
_V3_GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {"a": [1]}}
_INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}
_V3_CONSOLIDATED: dict[str, Any] = {
    **_INLINE,
    "metadata": {
        "g": {**_V3_GROUP, "consolidated_metadata": {**_INLINE, "metadata": {"x": _V3_ARRAY}}},
        "g/x": _V3_ARRAY,
        "h": _V3_GROUP,
    },
}
_V2_CONSOLIDATED: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {
        ".zgroup": {"zarr_format": 2},
        "a/.zgroup": {"zarr_format": 2},
        "a/.zattrs": {"x": 1},
        "a/b/.zarray": _V2_STRUCT,
    },
}

Reader = Callable[[object], Any]

_READERS: dict[str, tuple[Reader, dict[str, Any]]] = {
    "v2-array": (read_array_metadata_v2, _V2_ARRAY),
    "v2-struct": (read_array_metadata_v2, _V2_STRUCT),
    "v2-group": (read_group_metadata_v2, {"zarr_format": 2, "attributes": {"a": 1}}),
    "v2-consolidated": (read_consolidated_metadata_v2, _V2_CONSOLIDATED),
    "v3-array": (read_array_metadata_v3, _V3_ARRAY),
    "v3-sharded": (read_array_metadata_v3, _V3_SHARDED),
    "v3-group": (read_group_metadata_v3, _V3_GROUP),
    "v3-consolidated": (read_consolidated_metadata_v3, _V3_CONSOLIDATED),
}

# Names a document might write where a name goes, each a mistake some
# reader once made: an internal filing name, a typestr part a family does
# not take, a short-hand codec, a node path that is not normalized.
_NAMES = st.sampled_from(
    [
        "struct",
        "<f4[ns]",
        "|O4",
        "<M8",
        "<b2",
        "<U2",
        "bytes",
        "gzip",
        "crc32c",
        "transpose",
        "sharding_indexed",
        "int16",
        "float32",
        "string",
        "",
        "/",
        "a//b",
        "/a",
        "a/./b",
        "attributes",
        "configuration",
        "fields",
    ]
)
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**64), 2**64),
    st.floats(),
    st.sampled_from([0.0, 1.0, 2.0, 3.0]),
    st.text("ab/.<>|f4[]", max_size=6),
    _NAMES,
)
_JSON = st.recursive(
    _SCALARS,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3),
        st.dictionaries(st.one_of(_NAMES, st.text("ab", max_size=2)), inner, max_size=3),
    ),
    max_leaves=6,
)


def _places(value: object, at: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """Every location inside `value`, `value` itself first."""
    found = [at]
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_places(item, (*at, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_places(item, (*at, index)))
    return found


def _container(document: object, at: tuple[str | int, ...]) -> dict[str, Any] | list[Any]:
    """The object or array at `at` inside `document`, which a change edits in place."""
    found = cast("Any", document)
    for part in at:
        found = found[part]
    return cast("dict[str, Any] | list[Any]", found)


_KEYS = st.one_of(_NAMES, st.text("ab/.", max_size=4))


@st.composite
def _changed(draw: st.DrawFn, document: dict[str, Any]) -> dict[str, Any]:
    """`document`, deep-copied and changed in one to three places."""
    changed = copy.deepcopy(document)
    for _ in range(draw(st.integers(1, 3))):
        places = [place for place in _places(changed) if len(place) != 0]
        if len(places) == 0:
            break
        place = draw(st.sampled_from(places))
        parent = _container(changed, place[:-1])
        how = draw(st.sampled_from(["replace", "name", "remove", "add", "rename"]))
        event(f"change: {how}")
        value = draw(_NAMES if how == "name" else _JSON)
        if isinstance(parent, dict):
            key = cast("str", place[-1])
            if how in ("replace", "name"):
                parent[key] = value
            elif how == "remove":
                del parent[key]
            elif how == "add":
                parent[draw(_KEYS)] = value
            else:
                renamed = draw(_KEYS)
                if renamed not in parent:
                    parent[renamed] = parent.pop(key)
        else:
            index = cast("int", place[-1])
            if how in ("replace", "name"):
                parent[index] = value
            elif how == "remove":
                del parent[index]
            else:
                parent.append(value)
    return changed


def _misplaced(
    document: object, problems: tuple[ValidationProblem, ...]
) -> list[ValidationProblem]:
    """The problems whose location is not in `document`. A missing key is located where it would be, so its parent must be there."""
    misplaced: list[ValidationProblem] = []
    for found in problems:
        at = found.loc[:-1] if found.kind == "missing_key" and len(found.loc) != 0 else found.loc
        if value_at(document, at) is UNSET:
            misplaced.append(found)
    return misplaced


def _misquoted(
    document: object, problems: tuple[ValidationProblem, ...]
) -> list[ValidationProblem]:
    """The problems whose `input` is not the value at their location in `document`. A missing key has no value there to quote."""
    misquoted: list[ValidationProblem] = []
    for found in problems:
        if found.kind == "missing_key":
            continue
        there = value_at(document, found.loc)
        if there is UNSET or not _same(found.input, there):
            misquoted.append(found)
    return misquoted


def _same(one: object, other: object) -> bool:
    """Whether two JSON values are equal as JSON text: an array is the same as a tuple, and `NaN` as `NaN`."""
    return json.dumps(one, sort_keys=True) == json.dumps(other, sort_keys=True)


def _repeated(problems: tuple[ValidationProblem, ...]) -> list[tuple[object, ...]]:
    seen: set[tuple[object, ...]] = set()
    repeated: list[tuple[object, ...]] = []
    for found in problems:
        key = (found.loc, found.kind, found.message)
        if key in seen:
            repeated.append(key)
        seen.add(key)
    return repeated


@pytest.mark.parametrize("which", list(_READERS))
@_PROPERTY
@given(data=st.data())
def test_every_problem_is_located_in_the_document_and_reported_once(
    which: str, data: st.DataObject
) -> None:
    """Whatever a reader makes of a changed document, each problem is located in that document, a missing key where it would go, and carries the value found there as its `input`. No problem is reported twice. An unknown key is located at the key itself."""
    read, valid = _READERS[which]
    assert read(valid).problems == ()
    document = data.draw(_changed(valid))
    problems = read(document).problems
    event("refused" if len(problems) != 0 else "read")
    assert _misplaced(document, problems) == []
    assert _misquoted(document, problems) == []
    assert _repeated(problems) == []
    for found in problems:
        if found.kind == "unknown_key":
            assert isinstance(found.loc[-1], str)
            assert isinstance(value_at(document, found.loc[:-1]), Mapping)


@st.composite
def _node_files(draw: st.DrawFn) -> dict[str, object]:
    """The files of a v2 array node, its `.zarray` changed and its `.zattrs` any JSON."""
    zarray = draw(_changed(_V2_ARRAY))
    return {".zarray": zarray, ".zattrs": draw(st.one_of(_JSON, st.just({"a": 1})))}


@_PROPERTY
@given(files=_node_files())
def test_a_node_s_problems_are_located_in_the_files_it_reads(files: dict[str, object]) -> None:
    """`read_node_metadata_v2` reads a mapping of files, so each problem is located in one of them, and carries the value found there."""
    problems = read_node_metadata_v2(files).problems
    assert _misplaced(files, problems) == []
    assert _misquoted(files, problems) == []


_TYPESTRS = st.builds(
    lambda order, code, size, unit: f"{order}{code}{size}{unit}",
    st.sampled_from("<>|"),
    st.sampled_from("bBiufcmMOSUVxe"),
    st.sampled_from(["", "0", "1", "2", "3", "4", "8", "16"]),
    st.sampled_from(["", "[ns]", "[10ms]", "[D]", "[x]", "[]", "[2]"]),
)


@_PROPERTY
@given(typestr=_TYPESTRS, nested=st.booleans())
def test_a_typestr_s_problems_are_wrong_values_of_the_dtype(typestr: str, nested: bool) -> None:
    """A typestr is one token. It reads, or each of its wrong parts is a wrong value of the dtype, at the dtype: no part is a key a tolerant reader could leave out, and none is missing."""
    dtype: object = [["a", typestr]] if nested else typestr
    at: tuple[str | int, ...] = ("dtype", 0, 1) if nested else ("dtype",)
    document = {**_V2_ARRAY, "dtype": dtype, "fill_value": None, "filters": None}
    reading = read_array_metadata_v2(document)
    found = {(problem.loc, problem.kind) for problem in reading.problems}
    event("refused" if len(found) != 0 else "read")
    assert found <= {(at, "invalid_value")}
    # The typestr is refused exactly when it has a problem. A type code the
    # spec does not list is unclaimed, which is no problem.
    assert isinstance(dict(reading.fields())[at], RefusedField) == (len(found) != 0)


_INTEGER_LITERALS: dict[str, tuple[Reader, dict[str, Any], tuple[str, ...]]] = {
    "v2-zarr_format": (read_array_metadata_v2, _V2_ARRAY, ("zarr_format",)),
    "v3-zarr_format": (read_array_metadata_v3, _V3_ARRAY, ("zarr_format",)),
    "v2-group-zarr_format": (read_group_metadata_v2, {"zarr_format": 2}, ("zarr_format",)),
    "v2-consolidated-format": (
        read_consolidated_metadata_v2,
        _V2_CONSOLIDATED,
        ("zarr_consolidated_format",),
    ),
    "v2-blosc-shuffle": (
        read_array_metadata_v2,
        {
            **_V2_ARRAY,
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
            },
        },
        ("compressor", "shuffle"),
    ),
}


@pytest.mark.parametrize("which", list(_INTEGER_LITERALS))
@_PROPERTY
@given(value=st.floats(allow_nan=False, allow_infinity=False))
def test_a_float_for_an_integer_literal_is_the_wrong_type(which: str, value: float) -> None:
    """A float where a document requires one of a set of integers is the wrong type, whether or not it equals one of them, and wherever the set is declared."""
    read, valid, at = _INTEGER_LITERALS[which]
    document = copy.deepcopy(valid)
    cast("dict[str, Any]", _container(document, at[:-1]))[at[-1]] = value
    event("integral" if value.is_integer() else "fractional")
    assert [(p.loc, p.kind) for p in read(document).problems] == [(at, "invalid_type")]


_ANY_NUMBER = st.recursive(
    st.one_of(
        st.sampled_from([math.nan, math.inf, -math.inf]),
        st.floats(),
        st.integers(),
        st.text(max_size=3),
        st.none(),
    ),
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=3), inner, max_size=3)
    ),
    max_leaves=5,
)


@_PROPERTY
@given(member=_ANY_NUMBER)
def test_a_member_nothing_claims_may_hold_any_number_in_either_format(member: object) -> None:
    """A member the spec does not define is not read, so like attributes it may hold `NaN` and the infinities, in a v2 `.zarray`, in a `.zmetadata` entry, and in a v3 document alike."""
    v2 = {**_V2_ARRAY, "unexpected": member}
    v3 = {**_V3_ARRAY, "unexpected": {"must_understand": False, "value": member}}
    event(
        "non-finite"
        if any(isinstance(x, float) and not math.isfinite(x) for x in _leaves(member))
        else "finite"
    )
    assert read_array_metadata_v2(v2).problems == ()
    assert (
        read_consolidated_metadata_v2(
            {"zarr_consolidated_format": 1, "metadata": {".zarray": v2}}
        ).problems
        == ()
    )
    assert read_array_metadata_v3(v3).problems == ()


def _leaves(value: object) -> list[object]:
    """Each value inside `value` that is not an object or an array."""
    inside = (value_at(value, place) for place in _places(value))
    return [leaf for leaf in inside if not isinstance(leaf, (Mapping, list))]


_PATHS = st.builds(
    lambda slash, names: ("/" if slash else "") + "/".join(names),
    st.booleans(),
    st.lists(st.sampled_from(["a", "b", "", ".", ".."]), min_size=1, max_size=3),
)


@_PROPERTY
@given(path=_PATHS)
def test_both_formats_refuse_the_same_node_paths(path: str) -> None:
    """A node path is names joined by single `/` characters, with no leading `/`, no empty name and no `.` or `..`, in a `.zmetadata` key as in a v3 listing. Each listed node has its parent groups listed too, so only the path decides."""
    names = path.split("/")
    parents = ["/".join(names[:end]) for end in range(1, len(names))]
    v2 = {
        "zarr_consolidated_format": 1,
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            **{f"{parent}/.zgroup": {"zarr_format": 2} for parent in parents},
            f"{path}/.zgroup": {"zarr_format": 2},
        },
    }
    v3 = {**_INLINE, "metadata": {**dict.fromkeys(parents, _V3_GROUP), path: _V3_GROUP}}
    refused_v2 = ("metadata", f"{path}/.zgroup") in {
        p.loc for p in read_consolidated_metadata_v2(v2).problems
    }
    refused_v3 = ("metadata", path) in {p.loc for p in read_consolidated_metadata_v3(v3).problems}
    event("refused" if refused_v3 else "read")
    assert refused_v2 == refused_v3
