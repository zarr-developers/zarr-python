from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr_indexing.domain import IndexDomain
from zarr_indexing.messages import NdselError
from zarr_indexing.output_map import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    output_index_map_from_json,
)
from zarr_indexing.transform import IndexTransform

if TYPE_CHECKING:
    from zarr_indexing.json import IndexTransformJSON


def _maps_equal(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ConstantMap):
        assert isinstance(b, ConstantMap)
        return a.offset == b.offset
    if isinstance(a, DimensionMap):
        assert isinstance(b, DimensionMap)
        return (a.input_dimension, a.offset, a.stride) == (b.input_dimension, b.offset, b.stride)
    assert isinstance(a, ArrayMap)
    assert isinstance(b, ArrayMap)
    return (
        a.offset == b.offset
        and a.stride == b.stride
        and np.array_equal(a.index_array, b.index_array)
    )


def _transforms_equal(a: IndexTransform, b: IndexTransform) -> bool:
    """Structural equality that compares `ArrayMap` index arrays element-wise
    (`IndexTransform`'s dataclass `__eq__` cannot, as numpy `==` is ambiguous)."""
    return (
        a.domain == b.domain
        and len(a.output) == len(b.output)
        and all(_maps_equal(x, y) for x, y in zip(a.output, b.output, strict=True))
    )


class TestIndexDomainJSON:
    def test_roundtrip(self) -> None:
        domain = IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
        json = domain.to_json()
        assert json == {
            "input_inclusive_min": [2, 5],
            "input_exclusive_max": [10, 20],
            "input_labels": ["", ""],
        }
        restored = IndexDomain.from_json(json)
        assert restored == domain

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        json = domain.to_json()
        assert json["input_labels"] == ["x", "y"]
        restored = IndexDomain.from_json(json)
        assert restored.labels == ("x", "y")

    def test_without_labels_emits_empty_and_round_trips_to_none(self) -> None:
        domain = IndexDomain.from_shape((5,))
        json = domain.to_json()
        # Canonical form always writes labels; an unlabeled domain gets [""]*rank.
        assert json["input_labels"] == [""]
        restored = IndexDomain.from_json(json)
        assert restored.labels is None

    def test_zero_origin(self) -> None:
        domain = IndexDomain.from_shape((10, 20, 30))
        json = domain.to_json()
        assert json == {
            "input_inclusive_min": [0, 0, 0],
            "input_exclusive_max": [10, 20, 30],
            "input_labels": ["", "", ""],
        }
        assert IndexDomain.from_json(json) == domain


class TestOutputIndexMapJSON:
    def test_constant(self) -> None:
        m = ConstantMap(offset=42)
        json = m.to_json()
        assert json == {"offset": 42}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 42

    def test_constant_zero(self) -> None:
        m = ConstantMap(offset=0)
        json = m.to_json()
        assert json == {"offset": 0}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 0

    def test_dimension(self) -> None:
        m = DimensionMap(input_dimension=1, offset=10, stride=3)
        json = m.to_json()
        assert json == {"offset": 10, "stride": 3, "input_dimension": 1}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, DimensionMap)
        assert restored.input_dimension == 1
        assert restored.offset == 10
        assert restored.stride == 3

    def test_dimension_stride_1_written(self) -> None:
        """Canonical form writes stride even at its default of 1."""
        m = DimensionMap(input_dimension=0)
        json = m.to_json()
        assert json == {"offset": 0, "stride": 1, "input_dimension": 0}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, DimensionMap)
        assert restored.stride == 1

    def test_array(self) -> None:
        arr = np.array([1, 5, 9], dtype=np.intp)
        m = ArrayMap(index_array=arr, offset=2, stride=3)
        json = m.to_json()
        # Canonical: stride/offset present, index_array_bounds present, and
        # no input_dimension (ndsel/TensorStore reject it beside index_array).
        assert json == {
            "offset": 2,
            "stride": 3,
            "index_array": [1, 5, 9],
            "index_array_bounds": ["-inf", "+inf"],
        }
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        np.testing.assert_array_equal(restored.index_array, arr)
        assert restored.offset == 2
        assert restored.stride == 3

    def test_array_stride_1_written(self) -> None:
        arr = np.array([0, 1, 2], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        json = m.to_json()
        assert json["stride"] == 1
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        assert restored.stride == 1

    def test_array_2d(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        json = m.to_json()
        assert json["index_array"] == [[1, 2], [3, 4]]
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        np.testing.assert_array_equal(restored.index_array, arr)

    def test_degenerate_singleton_array_collapses_to_constant(self) -> None:
        """An all-singleton index_array selects one coordinate -> constant map."""
        m = ArrayMap(index_array=np.array([[4]], dtype=np.intp), offset=1, stride=2)
        json = m.to_json()
        assert json == {"offset": 1 + 2 * 4}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 9


class TestIndexTransformJSON:
    def test_identity(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        json = t.to_json()
        assert json == {
            "input_rank": 2,
            "input_inclusive_min": [0, 0],
            "input_exclusive_max": [10, 20],
            "input_labels": ["", ""],
            "output": [
                {"offset": 0, "stride": 1, "input_dimension": 0},
                {"offset": 0, "stride": 1, "input_dimension": 1},
            ],
        }
        restored = IndexTransform.from_json(json)
        assert restored.domain == t.domain
        assert len(restored.output) == 2
        for orig, rest in zip(t.output, restored.output, strict=True):
            assert type(orig) is type(rest)

    def test_sliced(self) -> None:
        t = IndexTransform.from_shape((100,))[10:50:2]
        json = t.to_json()
        restored = IndexTransform.from_json(json)
        assert restored.domain.shape == t.domain.shape
        assert isinstance(restored.output[0], DimensionMap)
        orig = t.output[0]
        assert isinstance(orig, DimensionMap)
        assert restored.output[0].offset == orig.offset
        assert restored.output[0].stride == orig.stride

    def test_with_constant(self) -> None:
        t = IndexTransform.from_shape((10, 20))[3]
        json = t.to_json()
        restored = IndexTransform.from_json(json)
        assert isinstance(restored.output[0], ConstantMap)
        assert restored.output[0].offset == 3
        assert isinstance(restored.output[1], DimensionMap)

    def test_with_array(self) -> None:
        idx = np.array([1, 5, 9], dtype=np.intp)
        t = IndexTransform.from_shape((10, 20)).oindex[idx, :]
        json = t.to_json()
        # The oindex array must not carry input_dimension on the wire.
        assert "input_dimension" not in json["output"][0]
        restored = IndexTransform.from_json(json)
        assert isinstance(restored.output[0], ArrayMap)
        # Orthogonal arrays are normalized to full input rank with a singleton
        # axis on the dimension they do not vary over.
        assert restored.output[0].index_array.shape == (3, 1)
        np.testing.assert_array_equal(restored.output[0].index_array, idx.reshape(3, 1))
        assert isinstance(restored.output[1], DimensionMap)

    def test_roundtrip_preserves_singleton_axes(self) -> None:
        """Full-rank orthogonal arrays keep their singleton axes across JSON."""
        t = IndexTransform.from_shape((10, 20)).oindex[np.array([1, 3]), np.array([2, 4, 6])]
        restored = IndexTransform.from_json(t.to_json())
        orig0, orig1 = t.output[0], t.output[1]
        rest0, rest1 = restored.output[0], restored.output[1]
        assert isinstance(orig0, ArrayMap)
        assert isinstance(orig1, ArrayMap)
        assert isinstance(rest0, ArrayMap)
        assert isinstance(rest1, ArrayMap)
        assert rest0.index_array.shape == (2, 1)
        assert rest1.index_array.shape == (1, 3)
        np.testing.assert_array_equal(rest0.index_array, orig0.index_array)
        np.testing.assert_array_equal(rest1.index_array, orig1.index_array)

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        t = IndexTransform.identity(domain)
        json = t.to_json()
        assert json["input_labels"] == ["x", "y"]
        restored = IndexTransform.from_json(json)
        assert restored.domain.labels == ("x", "y")

    def test_tensorstore_compatible_format(self) -> None:
        """A canonical body loads and round-trips through the engine layer."""
        json: IndexTransformJSON = {
            "input_rank": 3,
            "input_inclusive_min": [0, 0, 0],
            "input_exclusive_max": [100, 200, 3],
            "input_labels": ["x", "y", "channel"],
            "output": [
                {"offset": 5},
                {"offset": 10, "stride": 2, "input_dimension": 1},
                # Full input rank, which is what TensorStore itself requires:
                # it rejects a rank-1 array over a rank-3 domain outright.
                {"offset": 0, "stride": 1, "index_array": [[[1, 2, 0]]]},
            ],
        }
        t = IndexTransform.from_json(json)
        assert t.domain.shape == (100, 200, 3)
        assert t.domain.labels == ("x", "y", "channel")
        assert isinstance(t.output[0], ConstantMap)
        assert t.output[0].offset == 5
        assert isinstance(t.output[1], DimensionMap)
        assert t.output[1].offset == 10
        assert t.output[1].stride == 2
        assert t.output[1].input_dimension == 1
        assert isinstance(t.output[2], ArrayMap)
        np.testing.assert_array_equal(t.output[2].index_array, [[[1, 2, 0]]])

        # Roundtrip
        json_rt = t.to_json()
        t_rt = IndexTransform.from_json(json_rt)
        assert t_rt.domain == t.domain


class TestCanonicalRoundTrips:
    """Round-trip `transform == from(to(transform))`, up to the documented
    degenerate-collapse (all-singleton ArrayMap -> ConstantMap)."""

    def test_oindex_multi_axis(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30)).oindex[np.array([1, 3]), :, np.array([2, 4, 6])]
        rt = IndexTransform.from_json(t.to_json())
        assert _transforms_equal(rt, t)

    def test_oindex_with_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20))[2:8].oindex[np.array([3, 5, 7]), :]
        rt = IndexTransform.from_json(t.to_json())
        assert _transforms_equal(rt, t)

    def test_vindex(self) -> None:
        t = IndexTransform.from_shape((10, 20)).vindex[np.array([1, 3, 5]), np.array([2, 4, 6])]
        rt = IndexTransform.from_json(t.to_json())
        assert _transforms_equal(rt, t)

    def test_vindex_with_residual_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30)).vindex[np.array([1, 3]), np.array([2, 4]), :]
        rt = IndexTransform.from_json(t.to_json())
        assert _transforms_equal(rt, t)

    def test_length1_degenerate_oindex_collapses(self) -> None:
        """A length-1 oindex selection is the ConstantMap it equals.

        The selection layer collapses it at construction; a hand-built
        all-singleton ArrayMap still collapses on serialize, so the canonical
        wire form is a `constant` map either way.
        """
        t = IndexTransform.from_shape((10, 20)).oindex[np.array([7]), :]
        m = t.output[0]
        assert isinstance(m, ConstantMap)
        assert m.offset == 7

        hand_built = IndexTransform(
            domain=t.domain,
            output=(ArrayMap(index_array=np.array([[7]], dtype=np.intp)), t.output[1]),
        )
        rt = IndexTransform.from_json(hand_built.to_json())
        rm = rt.output[0]
        assert isinstance(rm, ConstantMap)
        assert rm.offset == 7
        # The size-1 input dimension survives, unconsumed, in the domain.
        assert rt.domain == t.domain

    def test_slices_and_constants(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))[2:8:2, 5, :]
        rt = IndexTransform.from_json(t.to_json())
        assert _transforms_equal(rt, t)


def _index_array_body(index_array: Any, rank: int = 1, extent: int = 2) -> IndexTransformJSON:
    return {
        "input_rank": rank,
        "input_inclusive_min": [0] * rank,
        "input_exclusive_max": [extent] * rank,
        "input_labels": [""] * rank,
        "output": [{"offset": 0, "stride": 1, "index_array": index_array}],
    }


@pytest.mark.parametrize(
    ("index_array", "detail"),
    [
        ([0.9, 1.9], "float64"),
        ([0, 1.5], "float64"),
        ([True, False], "bool"),
        (["a", "b"], "str"),
        # Not lists at all, so they are turned away before their content is
        # looked at: a bare string would be iterated into characters, and a bare
        # integer would become a rank-0 array and then a length-1 map, so a
        # document naming no cells would select one.
        ("abc", "must be an array of integers"),
        (5, "must be an array of integers"),
        ([None, None], "object"),
    ],
    ids=["floats", "mixed", "bools", "strings", "string", "scalar", "nulls"],
)
def test_a_non_integer_index_array_is_rejected(index_array: Any, detail: str) -> None:
    """An `index_array` addresses output coordinates, so it must be integral.

    Lowering a float array silently truncated it (`[0.9, 1.9]` selected cells 0
    and 1), a bool array coerced to 0/1, and a string array leaked a raw NumPy
    `ValueError` from the middle of the conversion.
    """
    with pytest.raises(NdselError) as excinfo:
        IndexTransform.from_json(_index_array_body(index_array))
    assert excinfo.value.reason == "invalid_json"
    assert "index_array" in str(excinfo.value)
    assert detail in str(excinfo.value)


def test_a_ragged_index_array_is_rejected() -> None:
    """A nested list that is not rectangular is not an array at all."""
    with pytest.raises(NdselError) as excinfo:
        IndexTransform.from_json(_index_array_body([[0, 1], [2]]))
    assert excinfo.value.reason == "invalid_json"


@pytest.mark.parametrize(
    ("index_array", "rank", "extent"),
    [([0, 1], 1, 2), ([[0], [1]], 2, 2), ([], 1, 0)],
    ids=["1d", "2d", "empty"],
)
def test_an_integer_index_array_is_accepted(index_array: Any, rank: int, extent: int) -> None:
    """Integers of any nesting still lower, including an empty selection.

    An empty array selects nothing, so the domain it is read over is empty too;
    a domain with room for coordinates the array does not supply is rejected
    (see `test_an_index_array_that_does_not_span_its_domain_is_rejected`).
    """
    t = IndexTransform.from_json(_index_array_body(index_array, rank, extent))
    m = t.output[0]
    assert isinstance(m, ArrayMap)
    assert m.index_array.dtype == np.intp


def test_a_non_integer_index_array_is_rejected_by_the_map_loader() -> None:
    """The single-map loader enforces the same constraint as the transform one."""
    with pytest.raises(NdselError) as excinfo:
        output_index_map_from_json({"index_array": [0.5, 1.5]})
    assert excinfo.value.reason == "invalid_json"


def test_infinite_bound_rejected_on_lowering() -> None:
    body: IndexTransformJSON = {
        "input_rank": 1,
        "input_inclusive_min": [0],
        "input_exclusive_max": [["+inf"]],
        "input_labels": [""],
        "output": [{"offset": 0, "stride": 1, "input_dimension": 0}],
    }
    with pytest.raises(ValueError, match="infinite"):
        IndexTransform.from_json(body)


def test_a_lower_rank_index_array_is_widened_on_the_way_in() -> None:
    """External JSON may broadcast a lower-rank array; the engine never holds one.

    ndsel leaves index-array rank unvalidated, so a conformant producer may send
    an array of lower rank. It is widened at the boundary, which keeps the
    full-rank invariant true of every transform the engine builds.
    """
    # A rank-1 array widens into the trailing axis, so it spans that axis's
    # extent of four.
    body: IndexTransformJSON = {
        "input_inclusive_min": [0, 0],
        "input_exclusive_max": [3, 4],
        "output": [{"index_array": [1, 2, 0, 2]}, {"input_dimension": 1}],
    }
    transform = IndexTransform.from_json(body)
    array_map = transform.output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_array.shape == (1, 4)
    assert array_map.index_array.ndim == transform.domain.ndim


def test_an_index_array_of_the_wrong_rank_is_rejected() -> None:
    """Inside the engine, a rank that does not match the domain is a bug."""
    with pytest.raises(ValueError, match="index_array has 1 dims"):
        IndexTransform(
            domain=IndexDomain.from_shape((3, 4)),
            output=(ArrayMap(index_array=np.array([1, 2, 0], dtype=np.intp)),),
        )


def test_an_index_array_that_does_not_span_its_domain_is_rejected() -> None:
    """An array with entries for only part of an axis is not a smaller selection.

    Reading it that way is how a truncated index array turned into a partially
    written result rather than an error.
    """
    with pytest.raises(ValueError, match="neither 1 nor the domain's extent"):
        IndexTransform(
            domain=IndexDomain.from_shape((3, 2)),
            output=(
                ArrayMap(index_array=np.zeros((3, 0), dtype=np.intp)),
                DimensionMap(input_dimension=1),
            ),
        )


def test_an_empty_index_array_collapses_to_a_constant() -> None:
    """Selecting nothing must survive a trip through JSON.

    An empty index array names no cell, and can only be empty because an input
    dimension is, so nothing is ever read through it. It is degenerate in
    exactly the way a size-1 array is, and collapses the same way — which is
    also what TensorStore emits for `t[ts.d[0][[]]]`.

    Emitting the array instead produced a document nothing could load:
    `tolist()` renders every empty array as `[]` once the leading axis is the
    zero-length one, so the rank went with it, and the loader put the dependency
    back on a different axis by prepending singletons.
    """
    for shape, selection in (
        ((5, 3), (np.array([], dtype=np.intp), slice(None))),
        ((5, 5), (np.array([], dtype=np.intp), np.array([], dtype=np.intp))),
    ):
        transform = IndexTransform.from_shape(shape).oindex[selection]
        body = transform.to_json()

        assert all("index_array" not in m for m in body["output"])
        reloaded = IndexTransform.from_json(body)
        assert reloaded.domain == transform.domain
        assert reloaded.to_json() == body


def test_an_empty_index_array_from_elsewhere_is_recovered_from_the_domain() -> None:
    """A producer that does emit one is still readable when the domain settles it.

    This package never writes such a document, but ndsel does not forbid it, and
    the domain names the axis unambiguously when exactly one dimension is empty.
    """
    body: IndexTransformJSON = {
        "input_inclusive_min": [0, 0],
        "input_exclusive_max": [0, 4],
        "output": [{"index_array": []}, {"input_dimension": 1}],
    }
    array_map = IndexTransform.from_json(body).output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_array.shape == (0, 1)


def test_an_ambiguous_empty_index_array_is_rejected() -> None:
    """Two zero-length dimensions leave nothing to recover the axis from."""
    body: IndexTransformJSON = {
        "input_inclusive_min": [0, 0],
        "input_exclusive_max": [0, 0],
        "output": [{"index_array": []}, {"input_dimension": 1}],
    }
    with pytest.raises(NdselError) as excinfo:
        IndexTransform.from_json(body)
    assert excinfo.value.reason == "invalid_json"
    assert "zero-length" in str(excinfo.value)


@pytest.mark.parametrize(
    ("document", "reason", "detail"),
    [
        (
            {"input_inclusive_min": [0.0], "input_exclusive_max": [3], "input_labels": [""]},
            "invalid_json",
            "must be an integer",
        ),
        (
            {"input_inclusive_min": [0], "input_exclusive_max": ["3"], "input_labels": [""]},
            "invalid_json",
            "must be an integer",
        ),
        (
            {"input_inclusive_min": [False], "input_exclusive_max": [True], "input_labels": [""]},
            "invalid_json",
            "must be an integer",
        ),
        (
            {"input_inclusive_min": [0], "input_exclusive_max": [3], "input_labels": [5]},
            "invalid_json",
            "must be a string",
        ),
        (
            {"input_inclusive_min": [0], "input_exclusive_max": [2**200], "input_labels": [""]},
            "invalid_json",
            "64-bit signed range",
        ),
    ],
    ids=["float", "string", "bool", "non-string-label", "out-of-range"],
)
def test_a_malformed_domain_document_is_rejected(document: Any, reason: str, detail: str) -> None:
    """The domain loader validates what the message layer validates.

    Reading the keys directly was a second, undefended way into the same
    objects: a bare `int()` truncated `3.9` to 3, coerced `"3"` and `True`, and
    let a non-string label into a `tuple[str, ...]` — each building a domain
    that was not the document's, and re-dumping as a different document.
    """
    with pytest.raises(NdselError) as excinfo:
        IndexDomain.from_json(document)
    assert excinfo.value.reason == reason
    assert detail in str(excinfo.value)


def test_a_transform_body_cannot_reinterpret_itself_as_another_message() -> None:
    """A `kind` inside the body must not change which message is being read."""
    with pytest.raises(NdselError) as excinfo:
        IndexTransform.from_json({"kind": "points", "coords": [[1, 2], [3, 4]]})
    assert excinfo.value.reason == "invalid_json"
    assert "kind" in str(excinfo.value)


def test_an_engine_invariant_failure_leaves_the_loader_as_a_typed_error() -> None:
    """A document is invalid input however deep the check that catches it lives.

    The engine's rank and span invariants are the last gate a document passes,
    and they raised a bare `ValueError` written in the engine's vocabulary.
    """
    body: IndexTransformJSON = {
        "input_rank": 1,
        "input_inclusive_min": [0],
        "input_exclusive_max": [5],
        "input_labels": [""],
        "output": [{"index_array": [[1, 2], [3, 4]]}],
    }
    with pytest.raises(NdselError) as excinfo:
        IndexTransform.from_json(body)
    assert excinfo.value.reason == "rank_mismatch"
