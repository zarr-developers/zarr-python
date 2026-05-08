from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from tests.test_transforms.conftest import Expect, ExpectErr
from zarr.core._transforms.domain import IndexDomain
from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core._transforms.transform import (
    IndexTransform,
    _intersect_vectorized,
    selection_to_transform,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=IndexTransform.from_shape((10, 20)),
            expected={"input_rank": 2, "output_rank": 2, "domain_shape": (10, 20)},
            id="from_shape-2d",
        ),
        Expect(
            input=IndexTransform.from_shape(()),
            expected={"input_rank": 0, "output_rank": 0, "domain_shape": ()},
            id="from_shape-0d",
        ),
        Expect(
            input=IndexTransform.identity(IndexDomain(inclusive_min=(5,), exclusive_max=(15,))),
            expected={"input_rank": 1, "output_rank": 1, "domain_shape": (10,)},
            id="identity-non-zero-origin",
        ),
        Expect(
            input=IndexTransform(
                domain=IndexDomain.from_shape((10,)),
                output=(ConstantMap(offset=42), DimensionMap(input_dimension=0)),
            ),
            expected={"input_rank": 1, "output_rank": 2, "domain_shape": (10,)},
            id="custom-output-maps",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_success(case: Expect[IndexTransform, dict[str, Any]]) -> None:
    """IndexTransform constructors yield the expected ranks and domain shape."""
    t = case.input
    assert t.input_rank == case.expected["input_rank"]
    assert t.output_rank == case.expected["output_rank"]
    assert t.domain.shape == case.expected["domain_shape"]


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={
                "domain": IndexDomain.from_shape((10,)),
                "output": (DimensionMap(input_dimension=5),),
            },
            msg="input_dimension",
            exception_cls=ValueError,
            id="dimension-map-out-of-range",
        ),
        ExpectErr(
            input={
                "domain": IndexDomain.from_shape((10,)),
                "output": (DimensionMap(input_dimension=0, stride=0),),
            },
            msg="must be positive",
            exception_cls=ValueError,
            id="dimension-map-zero-stride",
        ),
        ExpectErr(
            input={
                "domain": IndexDomain.from_shape((10,)),
                "output": (DimensionMap(input_dimension=0, stride=-1),),
            },
            msg="must be positive",
            exception_cls=ValueError,
            id="dimension-map-negative-stride",
        ),
        ExpectErr(
            input={
                "domain": IndexDomain.from_shape((5, 3)),
                "output": (
                    ArrayMap(
                        index_array=np.array([0, 1, 2], dtype=np.intp),
                        input_dimensions=(7,),
                    ),
                ),
            },
            msg="out of range",
            exception_cls=ValueError,
            id="array-map-input-dim-out-of-range",
        ),
        ExpectErr(
            input={
                "domain": IndexDomain.from_shape((5, 3)),
                "output": (
                    ArrayMap(
                        index_array=np.zeros((5, 5), dtype=np.intp),
                        input_dimensions=(0, 0),
                    ),
                ),
            },
            msg="duplicate dimensions",
            exception_cls=ValueError,
            id="array-map-input-dims-duplicate",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_errors(case: ExpectErr[dict[str, Any]]) -> None:
    """IndexTransform construction with invalid output maps raises ValueError."""
    with pytest.raises(case.exception_cls, match=case.msg):
        IndexTransform(**case.input)


# ---------------------------------------------------------------------------
# from_shape produces an identity transform whose output maps are DimensionMaps
# pointing at the corresponding input dim with offset=0, stride=1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=(10, 20), expected=2, id="2d"),
        Expect(input=(7,), expected=1, id="1d"),
        Expect(input=(), expected=0, id="0d"),
    ],
    ids=lambda c: c.id,
)
def test_from_shape_produces_identity_dimension_maps(
    case: Expect[tuple[int, ...], int],
) -> None:
    """IndexTransform.from_shape produces DimensionMaps that map each output dim
    back to the corresponding input dim, with no offset and unit stride."""
    t = IndexTransform.from_shape(case.input)
    assert len(t.output) == case.expected
    for i, m in enumerate(t.output):
        assert isinstance(m, DimensionMap)
        assert m.input_dimension == i
        assert m.offset == 0
        assert m.stride == 1


# ---------------------------------------------------------------------------
# __getitem__ (basic indexing)
#
# Most successful branches are covered by selection_to_transform tests below;
# this set focuses on cases unique to the __getitem__ surface (composition,
# bare-int / bare-slice, ArrayMap interactions).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexTransform.from_shape((10, 20)), (slice(None), slice(None))),
            expected={"shape": (10, 20), "input_rank": 2, "output_rank": 2},
            id="identity-slice",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), (slice(2, 8), slice(5, 15))),
            expected={"shape": (6, 10), "input_rank": 2, "output_rank": 2},
            id="2d-narrowing-slices",
        ),
        Expect(
            input=(IndexTransform.from_shape((10,)), slice(None, None, 2)),
            expected={"shape": (5,), "input_rank": 1, "output_rank": 1},
            id="strided-slice",
        ),
        Expect(
            input=(IndexTransform.from_shape((10,)), slice(1, 9, 3)),
            expected={"shape": (3,), "input_rank": 1, "output_rank": 1},
            id="strided-slice-with-start",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), 3),
            expected={"shape": (20,), "input_rank": 1, "output_rank": 2},
            id="bare-int-drops-leading-dim",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20, 30)), (slice(None), 5, slice(None))),
            expected={"shape": (10, 30), "input_rank": 2, "output_rank": 3},
            id="int-drops-middle-dim",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20, 30)), (slice(2, 8), ...)),
            expected={"shape": (6, 20, 30), "input_rank": 3, "output_rank": 3},
            id="ellipsis-fills-trailing",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), (np.newaxis, slice(None), slice(None))),
            expected={"shape": (1, 10, 20), "input_rank": 3, "output_rank": 2},
            id="newaxis-prepends-axis",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), slice(2, 8)),
            expected={"shape": (6, 20), "input_rank": 2, "output_rank": 2},
            id="bare-slice-implicitly-fills-trailing",
        ),
    ],
    ids=lambda c: c.id,
)
def test_getitem_basic_success(
    case: Expect[tuple[IndexTransform, Any], dict[str, Any]],
) -> None:
    """IndexTransform.__getitem__ produces a sub-transform with the expected shape and rank."""
    transform, selection = case.input
    result = transform[selection]
    assert result.domain.shape == case.expected["shape"]
    assert result.input_rank == case.expected["input_rank"]
    assert result.output_rank == case.expected["output_rank"]


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), 10),
            msg="out of bounds",
            exception_cls=IndexError,
            id="int-at-upper-bound",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), -1),
            msg="out of bounds",
            exception_cls=IndexError,
            id="negative-int-out-of-domain",
        ),
    ],
    ids=lambda c: c.id,
)
def test_getitem_basic_errors(case: ExpectErr[tuple[IndexTransform, Any]]) -> None:
    """IndexTransform.__getitem__ rejects out-of-domain integer indices.

    Note: negative indices are LITERAL coordinates per TensorStore convention,
    not wrap-around. arr[-1] on a domain [0, 10) is out of bounds, not arr[9].
    """
    transform, selection = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        transform[selection]


def test_getitem_negative_int_valid_with_negative_origin() -> None:
    """A negative integer index is valid when the domain's origin is negative.

    Stand-alone test (not parametrized) because verifying the *literal-coordinate*
    semantics is the whole point — the assertion on the resulting ConstantMap
    offset is the load-bearing check, not the shape.
    """
    domain = IndexDomain(inclusive_min=(-5,), exclusive_max=(5,))
    t = IndexTransform.identity(domain)
    result = t[-3]
    assert isinstance(result.output[0], ConstantMap)
    assert result.output[0].offset == -3


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexTransform.from_shape((100,))[10:50], slice(5, 20)),
            expected={"shape": (15,), "offset": 15, "stride": 1},
            id="composed-slices",
        ),
        Expect(
            input=(IndexTransform.from_shape((100,))[::2], slice(None, None, 3)),
            expected={"shape": (17,), "offset": 0, "stride": 6},
            id="composed-strides",
        ),
    ],
    ids=lambda c: c.id,
)
def test_getitem_composition(
    case: Expect[tuple[IndexTransform, Any], dict[str, Any]],
) -> None:
    """Indexing a sliced transform composes offsets and strides on the DimensionMap."""
    transform, selection = case.input
    result = transform[selection]
    assert result.domain.shape == case.expected["shape"]
    assert isinstance(result.output[0], DimensionMap)
    assert result.output[0].offset == case.expected["offset"]
    assert result.output[0].stride == case.expected["stride"]


# Indexing into a transform whose output is already an ArrayMap — basic
# operations (int/slice/stride/newaxis) must transform the index_array itself
# rather than building a new map.
_array_map_1d = IndexTransform(
    domain=IndexDomain.from_shape((5,)),
    output=(
        ArrayMap(
            index_array=np.array([10, 20, 30, 40, 50], dtype=np.intp),
            input_dimensions=(0,),
        ),
    ),
)
_array_map_2d_3x2 = IndexTransform(
    domain=IndexDomain.from_shape((3, 2)),
    output=(
        ArrayMap(
            index_array=np.array([[10, 20], [30, 40], [50, 60]], dtype=np.intp),
            input_dimensions=(0, 1),
        ),
    ),
)
_array_map_2d_2x3 = IndexTransform(
    domain=IndexDomain.from_shape((2, 3)),
    output=(
        ArrayMap(
            index_array=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.intp),
            input_dimensions=(0, 1),
        ),
    ),
)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(_array_map_2d_3x2, 1),
            expected=np.array([30, 40], dtype=np.intp),
            id="int-on-array-map-drops-axis",
        ),
        Expect(
            input=(_array_map_1d, slice(1, 4)),
            expected=np.array([20, 30, 40], dtype=np.intp),
            id="slice-on-array-map",
        ),
        Expect(
            input=(_array_map_1d, slice(None, None, 2)),
            expected=np.array([10, 30, 50], dtype=np.intp),
            id="strided-slice-on-array-map",
        ),
        Expect(
            input=(_array_map_2d_2x3, (0, slice(1, 3))),
            expected=np.array([20, 30], dtype=np.intp),
            id="int-then-slice-on-2d-array-map",
        ),
    ],
    ids=lambda c: c.id,
)
def test_getitem_on_array_map(
    case: Expect[tuple[IndexTransform, Any], np.ndarray[Any, np.dtype[np.intp]]],
) -> None:
    """Basic indexing on a transform whose output is an ArrayMap reshapes the index array."""
    transform, selection = case.input
    result = transform[selection]
    assert isinstance(result.output[0], ArrayMap)
    np.testing.assert_array_equal(result.output[0].index_array, case.expected)


def test_getitem_newaxis_on_array_map() -> None:
    """np.newaxis on an ArrayMap inserts a new input dim into the domain but
    leaves the array's parameterization unchanged. The array's input_dimensions
    just shifts to point at the new index of the old dim."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(
                index_array=np.array([10, 20, 30], dtype=np.intp),
                input_dimensions=(0,),
            ),
        ),
    )
    result = t[np.newaxis, :]
    assert result.input_rank == 2
    assert result.domain.shape == (1, 3)
    assert isinstance(result.output[0], ArrayMap)
    # newaxis is at new dim 0; old dim 0 shifts to new dim 1.
    assert result.output[0].input_dimensions == (1,)
    assert result.output[0].index_array.shape == (3,)
    np.testing.assert_array_equal(result.output[0].index_array, np.array([10, 20, 30]))


# ---------------------------------------------------------------------------
# oindex (orthogonal indexing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (np.array([1, 3, 5], dtype=np.intp), slice(None)),
            ),
            expected={"shape": (3, 20), "out0_kind": ArrayMap, "out1_kind": DimensionMap},
            id="int-array-and-slice",
        ),
        Expect(
            input=(IndexTransform.from_shape((5,)), (np.array([True, False, True, False, True]),)),
            expected={"shape": (3,), "out0_kind": ArrayMap, "out1_kind": None},
            id="bool-mask",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (np.array([2, 4], dtype=np.intp), slice(5, 15)),
            ),
            expected={"shape": (2, 10), "out0_kind": ArrayMap, "out1_kind": DimensionMap},
            id="array-and-narrowing-slice",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20, 30)),
                (
                    np.array([1, 3], dtype=np.intp),
                    slice(None),
                    np.array([5, 10, 15], dtype=np.intp),
                ),
            ),
            expected={"shape": (2, 20, 3), "out0_kind": ArrayMap, "out1_kind": DimensionMap},
            id="three-dims-mixed",
        ),
    ],
    ids=lambda c: c.id,
)
def test_oindex_success(case: Expect[tuple[IndexTransform, Any], dict[str, Any]]) -> None:
    """IndexTransform.oindex combines array indices independently per dimension."""
    transform, selection = case.input
    result = transform.oindex[selection]
    assert result.domain.shape == case.expected["shape"]
    assert isinstance(result.output[0], case.expected["out0_kind"])
    if case.expected["out1_kind"] is not None:
        assert isinstance(result.output[1], case.expected["out1_kind"])


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), (slice(None, None, -1),)),
            msg="slice step must be positive",
            exception_cls=IndexError,
            id="negative-slice-step",
        ),
    ],
    ids=lambda c: c.id,
)
def test_oindex_errors(case: ExpectErr[tuple[IndexTransform, Any]]) -> None:
    """IndexTransform.oindex rejects non-positive slice steps."""
    transform, selection = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        transform.oindex[selection]


def test_oindex_on_1d_array_map_with_int_array() -> None:
    """oindex on a transform with a 1-D ArrayMap output indexes that ArrayMap's
    array along its single parameterizing input dim."""
    arr = np.array([10, 20, 30, 40, 50], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((5,)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0,)),),
    )
    result = t.oindex[np.array([0, 2, 4], dtype=np.intp)]
    assert result.input_rank == 1
    assert result.domain.shape == (3,)
    assert isinstance(result.output[0], ArrayMap)
    np.testing.assert_array_equal(result.output[0].index_array, np.array([10, 30, 50]))


def test_oindex_on_2d_array_map_all_slices() -> None:
    """oindex on a 2-D ArrayMap with slices on every axis is well-defined
    (no axes selected by integer arrays)."""
    arr = np.arange(12, dtype=np.intp).reshape(3, 4)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3, 4)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0, 1)),),
    )
    # Both axes sliced; no array indices.
    result = t.oindex[1:3, 0:3]
    assert result.input_rank == 2
    assert result.domain.shape == (2, 3)
    assert isinstance(result.output[0], ArrayMap)
    np.testing.assert_array_equal(result.output[0].index_array, arr[1:3, 0:3])


def test_oindex_on_multi_dim_array_map_with_two_array_axes_errors() -> None:
    """oindex on a multi-dim ArrayMap with two or more axes selected by
    integer arrays needs np.ix_-style outer-product semantics. Until that
    is implemented, raise NotImplementedError."""
    arr = np.arange(12, dtype=np.intp).reshape(3, 4)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3, 4)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0, 1)),),
    )
    with pytest.raises(NotImplementedError, match="multi-dimensional ArrayMap"):
        t.oindex[np.array([0, 2], dtype=np.intp), np.array([1, 3], dtype=np.intp)]


# ---------------------------------------------------------------------------
# vindex (vectorized indexing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexTransform.from_shape((10,)), np.array([1, 3, 5], dtype=np.intp)),
            expected=(3,),
            id="single-1d-array",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (
                    np.array([[1, 2], [3, 4]], dtype=np.intp),
                    np.array([[10, 11], [12, 13]], dtype=np.intp),
                ),
            ),
            expected=(2, 2),
            id="two-2d-arrays-broadcast",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20, 30)),
                (np.array([1, 3, 5], dtype=np.intp), slice(None), slice(None)),
            ),
            expected=(3, 20, 30),
            id="array-with-trailing-slices",
        ),
        Expect(
            input=(IndexTransform.from_shape((5,)), np.array([True, False, True, False, True])),
            expected=(3,),
            id="bool-mask",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (np.array([1, 2, 3], dtype=np.intp), np.array([[10], [11]], dtype=np.intp)),
            ),
            expected=(2, 3),
            id="broadcast-different-shapes",
        ),
    ],
    ids=lambda c: c.id,
)
def test_vindex_success(case: Expect[tuple[IndexTransform, Any], tuple[int, ...]]) -> None:
    """IndexTransform.vindex broadcasts array indices and produces correlated ArrayMaps."""
    transform, selection = case.input
    result = transform.vindex[selection]
    assert result.domain.shape == case.expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), (slice(None, None, -1),)),
            msg="slice step must be positive",
            exception_cls=IndexError,
            id="negative-slice-step",
        ),
    ],
    ids=lambda c: c.id,
)
def test_vindex_errors(case: ExpectErr[tuple[IndexTransform, Any]]) -> None:
    """IndexTransform.vindex rejects non-positive slice steps."""
    transform, selection = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        transform.vindex[selection]


# ---------------------------------------------------------------------------
# selection_to_transform — the public dispatch front door for all three modes.
# Sanity check that each mode produces the expected output kind.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexTransform.from_shape((10, 20)), (slice(2, 8), slice(5, 15)), "basic"),
            expected={"shape": (6, 10), "out0_kind": DimensionMap},
            id="basic-slices",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), (3, slice(None)), "basic"),
            expected={"shape": (20,), "out0_kind": ConstantMap},
            id="basic-int-and-slice",
        ),
        Expect(
            input=(IndexTransform.from_shape((10, 20)), Ellipsis, "basic"),
            expected={"shape": (10, 20), "out0_kind": DimensionMap},
            id="basic-bare-ellipsis",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (np.array([1, 3, 5], dtype=np.intp), slice(None)),
                "orthogonal",
            ),
            expected={"shape": (3, 20), "out0_kind": ArrayMap},
            id="orthogonal",
        ),
        Expect(
            input=(
                IndexTransform.from_shape((10, 20)),
                (np.array([1, 3], dtype=np.intp), np.array([5, 7], dtype=np.intp)),
                "vectorized",
            ),
            expected={"shape": (2,), "out0_kind": ArrayMap},
            id="vectorized",
        ),
        Expect(
            input=(IndexTransform.from_shape((100,))[10:50], slice(5, 20), "basic"),
            expected={"shape": (15,), "out0_kind": DimensionMap},
            id="composes-with-non-identity-base",
        ),
    ],
    ids=lambda c: c.id,
)
def test_selection_to_transform_success(
    case: Expect[
        tuple[IndexTransform, Any, Literal["basic", "orthogonal", "vectorized"]], dict[str, Any]
    ],
) -> None:
    """selection_to_transform dispatches to basic/orthogonal/vectorized correctly."""
    transform, selection, mode = case.input
    result = selection_to_transform(selection, transform, mode)
    assert result.domain.shape == case.expected["shape"]
    assert isinstance(result.output[0], case.expected["out0_kind"])


def test_selection_to_transform_unknown_mode_errors() -> None:
    """selection_to_transform rejects unknown indexing modes.

    The `mode` parameter is typed as `Literal["basic", "orthogonal", "vectorized"]`,
    so this test bypasses static type checking to exercise the runtime guard.
    """
    t = IndexTransform.from_shape((10,))
    with pytest.raises(ValueError, match="Unknown mode"):
        selection_to_transform(slice(None), t, "diagonal")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# intersect — restrict an output domain. Returns (sub_transform, surviving)
# or None when the intersection is empty.
# ---------------------------------------------------------------------------


def test_intersect_constant_inside() -> None:
    """A ConstantMap whose offset is inside the chunk survives unchanged."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ConstantMap(offset=5),),
    )
    result = t.intersect(IndexDomain(inclusive_min=(0,), exclusive_max=(10,)))
    assert result is not None
    restricted, surviving = result
    assert isinstance(restricted.output[0], ConstantMap)
    assert restricted.output[0].offset == 5
    assert surviving is None


def test_intersect_constant_outside() -> None:
    """A ConstantMap whose offset is outside the chunk yields None."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ConstantMap(offset=5),),
    )
    assert t.intersect(IndexDomain(inclusive_min=(10,), exclusive_max=(20,))) is None


def test_intersect_dimension_partial() -> None:
    """A DimensionMap whose storage-coord range partially overlaps the chunk
    narrows the input domain to the surviving slice."""
    t = IndexTransform.from_shape((10,))
    result = t.intersect(IndexDomain(inclusive_min=(5,), exclusive_max=(15,)))
    assert result is not None
    restricted, surviving = result
    assert restricted.domain.inclusive_min == (5,)
    assert restricted.domain.exclusive_max == (10,)
    assert surviving is None


def test_intersect_dimension_no_overlap() -> None:
    """A DimensionMap whose storage-coord range does not overlap the chunk yields None."""
    t = IndexTransform.from_shape((10,))
    assert t.intersect(IndexDomain(inclusive_min=(20,), exclusive_max=(30,))) is None


def test_intersect_dimension_strided() -> None:
    """Strided DimensionMap: storage = offset + stride * input. Only inputs that land
    in the chunk survive."""
    # offset=1, stride=2, input [0,5): storage = {1, 3, 5, 7, 9}. Chunk [4, 8) -> {5, 7}.
    t = IndexTransform(
        domain=IndexDomain.from_shape((5,)),
        output=(DimensionMap(input_dimension=0, offset=1, stride=2),),
    )
    result = t.intersect(IndexDomain(inclusive_min=(4,), exclusive_max=(8,)))
    assert result is not None
    restricted, _ = result
    assert restricted.domain.inclusive_min == (2,)
    assert restricted.domain.exclusive_max == (4,)


def test_intersect_array_partial() -> None:
    """An ArrayMap whose storage coords partially overlap the chunk yields a filtered ArrayMap
    plus a `surviving` mask of the input indices that survived."""
    arr = np.array([3, 8, 15, 22], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((4,)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0,)),),
    )
    result = t.intersect(IndexDomain(inclusive_min=(5,), exclusive_max=(20,)))
    assert result is not None
    restricted, surviving = result
    assert isinstance(restricted.output[0], ArrayMap)
    np.testing.assert_array_equal(restricted.output[0].index_array, np.array([8, 15]))
    assert surviving is not None
    np.testing.assert_array_equal(surviving, np.array([1, 2]))


def test_intersect_array_disjoint() -> None:
    """An ArrayMap whose storage coords are entirely outside the chunk yields None."""
    arr = np.array([1, 2, 3], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0,)),),
    )
    assert t.intersect(IndexDomain(inclusive_min=(10,), exclusive_max=(20,))) is None


def test_intersect_2d_mixed_constant_and_dimension() -> None:
    """2D output: ConstantMap on dim 0 (inside chunk), DimensionMap on dim 1 (overlaps chunk)."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(
            ConstantMap(offset=5),
            DimensionMap(input_dimension=0, offset=0, stride=1),
        ),
    )
    chunk = IndexDomain(inclusive_min=(0, 5), exclusive_max=(10, 15))
    result = t.intersect(chunk)
    assert result is not None
    restricted, _ = result
    assert isinstance(restricted.output[0], ConstantMap)
    assert restricted.output[0].offset == 5
    assert isinstance(restricted.output[1], DimensionMap)
    assert restricted.domain.inclusive_min == (5,)
    assert restricted.domain.exclusive_max == (10,)


def test_intersect_rank_mismatch_errors() -> None:
    """intersect rejects an output_domain whose rank differs from the transform's output rank."""
    t = IndexTransform.from_shape((10,))  # output rank 1
    chunk = IndexDomain.from_shape((10, 20))  # rank 2
    with pytest.raises(ValueError, match="output rank"):
        t.intersect(chunk)


# ---------------------------------------------------------------------------
# Direct tests for _intersect_vectorized.
#
# Public `intersect` only calls _intersect_vectorized when the transform has
# 2+ ArrayMap outputs (correlated indices). All public test cases use exactly
# one ArrayMap, so this branch is unreachable from public-surface tests.
# ---------------------------------------------------------------------------


def _vectorized_2d_array_map() -> IndexTransform:
    """Helper: a vectorized transform over a (3,) input domain with two
    correlated ArrayMaps. Storage coords: (1,10), (5,11), (9,12).

    Both ArrayMaps share input_dimensions=(0,) — that's what makes them
    correlated under the new design."""
    return IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=np.array([1, 5, 9], dtype=np.intp), input_dimensions=(0,)),
            ArrayMap(index_array=np.array([10, 11, 12], dtype=np.intp), input_dimensions=(0,)),
        ),
    )


def test_intersect_vectorized_partial_survival() -> None:
    """Two correlated ArrayMaps; only points where ALL coords are in-chunk survive."""
    t = _vectorized_2d_array_map()
    chunk = IndexDomain(inclusive_min=(0, 10), exclusive_max=(8, 12))
    # Storage points (1,10), (5,11), (9,12). In-chunk: (1,10), (5,11). (9,12) fails dim 1.
    result = _intersect_vectorized(t, chunk, [0, 1])
    assert result is not None
    restricted, surviving = result
    assert isinstance(restricted.output[0], ArrayMap)
    assert isinstance(restricted.output[1], ArrayMap)
    np.testing.assert_array_equal(restricted.output[0].index_array, np.array([1, 5]))
    np.testing.assert_array_equal(restricted.output[1].index_array, np.array([10, 11]))
    assert surviving is not None
    np.testing.assert_array_equal(surviving, np.array([0, 1]))


def test_intersect_vectorized_no_survival() -> None:
    """If no point is in-chunk on all dims, returns None."""
    t = _vectorized_2d_array_map()
    chunk = IndexDomain(inclusive_min=(20, 20), exclusive_max=(30, 30))
    assert _intersect_vectorized(t, chunk, [0, 1]) is None


def test_intersect_vectorized_with_constant_outside_drops_to_none() -> None:
    """When a ConstantMap output is outside the chunk, the entire transform fails."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=np.array([1, 2, 3], dtype=np.intp), input_dimensions=(0,)),
            ArrayMap(index_array=np.array([10, 11, 12], dtype=np.intp), input_dimensions=(0,)),
            ConstantMap(offset=99),
        ),
    )
    chunk = IndexDomain(inclusive_min=(0, 0, 0), exclusive_max=(10, 20, 5))
    assert _intersect_vectorized(t, chunk, [0, 1]) is None


# ---------------------------------------------------------------------------
# translate — shift every coordinate by an offset.
# ---------------------------------------------------------------------------

_translate_dimension_t = IndexTransform.from_shape((10,))
_translate_array_t = IndexTransform(
    domain=IndexDomain.from_shape((2,)),
    output=(
        ArrayMap(
            index_array=np.array([5, 10], dtype=np.intp),
            input_dimensions=(0,),
            offset=3,
        ),
    ),
)
_translate_constant_t = IndexTransform(
    domain=IndexDomain.from_shape((10,)),
    output=(ConstantMap(offset=5),),
)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(_translate_constant_t, (-5,)),
            expected={"out_kind": ConstantMap, "offset": 0},
            id="constant",
        ),
        Expect(
            input=(_translate_dimension_t, (-3,)),
            expected={"out_kind": DimensionMap, "offset": -3, "stride": 1},
            id="dimension",
        ),
        Expect(
            input=(_translate_array_t, (-3,)),
            expected={"out_kind": ArrayMap, "offset": 0},
            id="array",
        ),
    ],
    ids=lambda c: c.id,
)
def test_translate_success(
    case: Expect[tuple[IndexTransform, tuple[int, ...]], dict[str, Any]],
) -> None:
    """IndexTransform.translate adjusts each output map's offset uniformly."""
    transform, shift = case.input
    result = transform.translate(shift)
    out0 = result.output[0]
    assert isinstance(out0, case.expected["out_kind"])
    assert out0.offset == case.expected["offset"]
    if "stride" in case.expected:
        assert isinstance(out0, DimensionMap)
        assert out0.stride == case.expected["stride"]


def test_translate_2d() -> None:
    """A multi-dimensional translate shifts all output dims independently."""
    t = IndexTransform.from_shape((10, 20))
    result = t.translate((-5, -10))
    out0, out1 = result.output
    assert isinstance(out0, DimensionMap)
    assert out0.offset == -5
    assert isinstance(out1, DimensionMap)
    assert out1.offset == -10


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10, 20)), (1,)),
            msg="shift must have length",
            exception_cls=ValueError,
            id="shift-too-short",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), (1, 2)),
            msg="shift must have length",
            exception_cls=ValueError,
            id="shift-too-long",
        ),
    ],
    ids=lambda c: c.id,
)
def test_translate_errors(case: ExpectErr[tuple[IndexTransform, tuple[int, ...]]]) -> None:
    """IndexTransform.translate rejects shifts whose length doesn't match output_rank."""
    transform, shift = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        transform.translate(shift)


# ---------------------------------------------------------------------------
# selection_repr and __repr__: verify the human-readable strings cover each
# OutputIndexMap variant.
# ---------------------------------------------------------------------------


def test_selection_repr_covers_all_map_kinds() -> None:
    """selection_repr produces a TensorStore-style domain string with one
    entry per output dim, formatted differently for each map kind."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ConstantMap(offset=5),
            DimensionMap(input_dimension=0, offset=2, stride=1),
            DimensionMap(input_dimension=0, offset=0, stride=3),
            ArrayMap(
                index_array=np.array([1, 5, 9], dtype=np.intp),
                input_dimensions=(0,),
            ),
        ),
    )
    repr_str = t.selection_repr
    assert "5" in repr_str  # ConstantMap
    assert "[2, 5)" in repr_str  # DimensionMap stride=1 over input [0, 3)
    assert "step 3" in repr_str  # DimensionMap stride=3
    assert "{1, 5, 9}" in repr_str  # ArrayMap (small)


def test_selection_repr_array_map_large() -> None:
    """ArrayMaps with more than 5 elements show as `array(N)` rather than spelled out."""
    arr = np.arange(10, dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0,)),),
    )
    assert "array(10)" in t.selection_repr


def test_repr_covers_all_map_kinds() -> None:
    """__repr__ formats each output map with its kind-specific shape."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10, 5)),
        output=(
            ConstantMap(offset=7),
            DimensionMap(input_dimension=0, offset=1, stride=2),
            ArrayMap(
                index_array=np.array([0, 1, 2, 3, 4], dtype=np.intp),
                input_dimensions=(1,),
            ),
        ),
    )
    s = repr(t)
    assert "out[0] = 7" in s
    assert "out[1] = 1 + 2 * in[0]" in s
    assert "out[2] = 0 + 1 * arr(5,)[in[1]]" in s


# ---------------------------------------------------------------------------
# intersect() public dispatch: prior tests call _intersect_vectorized directly;
# the public IndexTransform.intersect() vectorized path was untested.
# ---------------------------------------------------------------------------


def test_intersect_dispatches_to_vectorized_when_arraymaps_correlated() -> None:
    """IndexTransform.intersect() uses the vectorized path when 2+ ArrayMaps
    share an input dimension. It uses the orthogonal path when ArrayMaps have
    disjoint input dimensions."""
    # Correlated: both ArrayMaps share input_dimensions=(0,) on a 1-D domain.
    t_correlated = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(
                index_array=np.array([1, 5, 9], dtype=np.intp),
                input_dimensions=(0,),
            ),
            ArrayMap(
                index_array=np.array([10, 11, 12], dtype=np.intp),
                input_dimensions=(0,),
            ),
        ),
    )
    chunk = IndexDomain(inclusive_min=(0, 10), exclusive_max=(8, 12))
    result = t_correlated.intersect(chunk)
    assert result is not None
    _, surviving = result
    # Both points (1,10), (5,11) survive; (9,12) fails dim 1.
    assert surviving is not None
    np.testing.assert_array_equal(surviving, np.array([0, 1]))


# ---------------------------------------------------------------------------
# _intersect_vectorized with a DimensionMap output on a NON-correlated input
# dim: the post-refactor path that preserves the non-broadcast input dim.
# ---------------------------------------------------------------------------


def test_intersect_vectorized_preserves_non_correlated_dim() -> None:
    """A vindex transform with a non-broadcast input dim produces a
    DimensionMap on that dim. Intersecting must remap that DimensionMap's
    input_dimension to the new domain (where the broadcast dim has been
    collapsed to (len(surviving),) at index 0)."""
    # Construct the transform that vindex-with-trailing-slice would produce:
    # (broadcast_dim=3, slice_dim=20).  output[0] = ArrayMap on broadcast,
    # output[1] = DimensionMap on slice_dim.
    t = IndexTransform(
        domain=IndexDomain.from_shape((3, 20)),
        output=(
            ArrayMap(
                index_array=np.array([1, 5, 9], dtype=np.intp),
                input_dimensions=(0,),
            ),
            DimensionMap(input_dimension=1, offset=0, stride=1),
        ),
    )
    # Two correlated outputs needed for vectorized path; add a second ArrayMap
    # on the same broadcast dim.
    t_with_two_arrays = IndexTransform(
        domain=IndexDomain.from_shape((3, 20)),
        output=(
            ArrayMap(
                index_array=np.array([1, 5, 9], dtype=np.intp),
                input_dimensions=(0,),
            ),
            ArrayMap(
                index_array=np.array([2, 6, 10], dtype=np.intp),
                input_dimensions=(0,),
            ),
            DimensionMap(input_dimension=1, offset=0, stride=1),
        ),
    )
    chunk = IndexDomain(inclusive_min=(0, 0, 0), exclusive_max=(20, 20, 20))
    result = t_with_two_arrays.intersect(chunk)
    assert result is not None
    restricted, surviving = result
    # Surviving points: all 3 (all storage coords in [0,20)).
    assert surviving is not None
    np.testing.assert_array_equal(surviving, np.array([0, 1, 2]))
    # New domain: (3 surviving, 20 from preserved slice dim).
    assert restricted.domain.shape == (3, 20)
    # output[2] (the DimensionMap) should have its input_dimension remapped
    # from old dim 1 to new dim 1 (broadcast dim is now new dim 0).
    out_dim_map = restricted.output[2]
    assert isinstance(out_dim_map, DimensionMap)
    assert out_dim_map.input_dimension == 1
    # silence unused-var: t was an intermediate construction reference
    assert t.output_rank == 2


# ---------------------------------------------------------------------------
# _apply_basic_indexing rejects negative slice steps.
# ---------------------------------------------------------------------------


def test_basic_indexing_rejects_negative_slice_step() -> None:
    t = IndexTransform.from_shape((10,))
    with pytest.raises(IndexError, match="slice step must be positive"):
        t[slice(None, None, -1)]


# ---------------------------------------------------------------------------
# _apply_vindex on an existing ArrayMap output raises NotImplementedError.
# ---------------------------------------------------------------------------


def test_vindex_on_existing_arraymap_errors() -> None:
    t = IndexTransform(
        domain=IndexDomain.from_shape((5,)),
        output=(
            ArrayMap(
                index_array=np.array([1, 2, 3, 4, 5], dtype=np.intp),
                input_dimensions=(0,),
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="ArrayMap"):
        t.vindex[np.array([0, 2], dtype=np.intp)]


# ---------------------------------------------------------------------------
# selection_to_transform validation: reject unsupported selection types.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), 1.5, "basic"),
            msg="unsupported selection type",
            exception_cls=IndexError,
            id="basic-rejects-float",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), 1.5, "orthogonal"),
            msg="unsupported selection type",
            exception_cls=IndexError,
            id="orthogonal-rejects-float",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), 1.5, "vectorized"),
            msg="unsupported selection type",
            exception_cls=IndexError,
            id="vectorized-rejects-float",
        ),
    ],
    ids=lambda c: c.id,
)
def test_selection_to_transform_rejects_unsupported_types(
    case: ExpectErr[tuple[IndexTransform, Any, Literal["basic", "orthogonal", "vectorized"]]],
) -> None:
    """selection_to_transform's validators reject types like float."""
    transform, selection, mode = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        selection_to_transform(selection, transform, mode)


# ---------------------------------------------------------------------------
# _apply_oindex parsing branches: bare int, list selection.
# ---------------------------------------------------------------------------


def test_oindex_bare_int_becomes_singleton_array() -> None:
    """oindex[3] on a 1-D transform converts the int to a 1-element array,
    producing an ArrayMap of length 1 (not a ConstantMap)."""
    t = IndexTransform.from_shape((10,))
    result = t.oindex[3]
    assert result.input_rank == 1
    assert result.domain.shape == (1,)
    assert isinstance(result.output[0], ArrayMap)
    np.testing.assert_array_equal(result.output[0].index_array, np.array([3]))


def test_oindex_list_selection() -> None:
    """oindex accepts a Python list and converts it to an integer array."""
    t = IndexTransform.from_shape((10,))
    result = t.oindex[[1, 3, 5]]
    assert result.input_rank == 1
    assert result.domain.shape == (3,)
    assert isinstance(result.output[0], ArrayMap)
    np.testing.assert_array_equal(result.output[0].index_array, np.array([1, 3, 5]))


# ---------------------------------------------------------------------------
# _apply_vindex parsing branches: ellipsis, 2D bool, list, bare int.
# ---------------------------------------------------------------------------


def test_vindex_ellipsis() -> None:
    """vindex[...] is a no-op identity."""
    t = IndexTransform.from_shape((4, 5))
    result = t.vindex[...]
    assert result.domain.shape == (4, 5)


def test_vindex_2d_bool_mask_consumes_two_dims() -> None:
    """A 2-D bool mask in vindex consumes both dims of a 2-D domain and
    expands into two correlated 1-D ArrayMaps."""
    t = IndexTransform.from_shape((3, 4))
    mask = np.array(
        [[True, False, True, False], [False, True, False, True], [True, True, False, False]]
    )
    result = t.vindex[mask]
    # 6 True entries; broadcast shape (6,).
    assert result.domain.shape == (6,)
    assert isinstance(result.output[0], ArrayMap)
    assert isinstance(result.output[1], ArrayMap)


def test_vindex_list_selection() -> None:
    """vindex accepts a Python list like oindex does."""
    t = IndexTransform.from_shape((10,))
    result = t.vindex[[1, 3, 5]]
    assert result.domain.shape == (3,)
    assert isinstance(result.output[0], ArrayMap)


def test_vindex_bare_int_becomes_singleton_array() -> None:
    """vindex[3] on a 1-D transform produces an ArrayMap of length 1."""
    t = IndexTransform.from_shape((10,))
    result = t.vindex[3]
    assert result.domain.shape == (1,)
    assert isinstance(result.output[0], ArrayMap)


def test_vindex_with_fewer_selections_than_dims_pads_with_slice() -> None:
    """vindex(arr) on a 2-D domain leaves trailing dims untouched (slice fill)."""
    t = IndexTransform.from_shape((3, 5))
    result = t.vindex[np.array([0, 1], dtype=np.intp)]
    # Broadcast dim (2,) prepended; trailing dim (5,) preserved.
    assert result.domain.shape == (2, 5)


# ---------------------------------------------------------------------------
# ConstantMap survives basic / oindex / vindex unchanged. The tests above
# exercise these paths for DimensionMap-only transforms; these cover the
# `output[i] is ConstantMap` branch in each of the three apply functions.
# ---------------------------------------------------------------------------


def test_basic_indexing_preserves_constant_map() -> None:
    """A ConstantMap output passes through basic indexing unchanged."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ConstantMap(offset=42), DimensionMap(input_dimension=0)),
    )
    result = t[2:8]
    assert isinstance(result.output[0], ConstantMap)
    assert result.output[0].offset == 42


def test_oindex_preserves_constant_map() -> None:
    """A ConstantMap output passes through oindex unchanged."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ConstantMap(offset=42), DimensionMap(input_dimension=0)),
    )
    result = t.oindex[np.array([1, 3, 5], dtype=np.intp)]
    assert isinstance(result.output[0], ConstantMap)
    assert result.output[0].offset == 42


def test_vindex_preserves_constant_map() -> None:
    """A ConstantMap output passes through vindex unchanged."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ConstantMap(offset=42), DimensionMap(input_dimension=0)),
    )
    result = t.vindex[np.array([1, 3, 5], dtype=np.intp)]
    assert isinstance(result.output[0], ConstantMap)
    assert result.output[0].offset == 42


def test_intersect_vectorized_constant_inside_chunk_passes() -> None:
    """In _intersect_vectorized, a ConstantMap whose offset is inside the
    chunk's range on its output dim is passed through. (The outside-chunk
    case yields None and is already tested.)"""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=np.array([1, 2, 3], dtype=np.intp), input_dimensions=(0,)),
            ArrayMap(index_array=np.array([10, 11, 12], dtype=np.intp), input_dimensions=(0,)),
            ConstantMap(offset=5),
        ),
    )
    chunk = IndexDomain(inclusive_min=(0, 0, 0), exclusive_max=(10, 20, 10))
    result = t.intersect(chunk)
    assert result is not None
    restricted, _ = result
    assert isinstance(restricted.output[2], ConstantMap)
    assert restricted.output[2].offset == 5


# ---------------------------------------------------------------------------
# Domain-level edge cases: empty-domain intersect, oindex with ellipsis or
# trailing-dim implicit fill.
# ---------------------------------------------------------------------------


def test_intersect_dimension_map_on_empty_domain_returns_none() -> None:
    """When a DimensionMap's input dim is already empty (input_lo >= input_hi),
    intersect returns None."""
    t = IndexTransform(
        domain=IndexDomain(inclusive_min=(0,), exclusive_max=(0,)),
        output=(DimensionMap(input_dimension=0, offset=0, stride=1),),
    )
    assert t.intersect(IndexDomain.from_shape((10,))) is None


def test_oindex_with_ellipsis() -> None:
    """oindex with ellipsis fills missing dims with slice(None)."""
    t = IndexTransform.from_shape((4, 5, 6))
    result = t.oindex[np.array([0, 2], dtype=np.intp), ...]
    # ellipsis fills dims 1 and 2 with slice(None); domain becomes (2, 5, 6).
    assert result.domain.shape == (2, 5, 6)


def test_oindex_with_implicit_trailing_dim_fill() -> None:
    """oindex with fewer entries than ndim pads trailing dims with slice(None)."""
    t = IndexTransform.from_shape((4, 5, 6))
    result = t.oindex[np.array([0, 2], dtype=np.intp)]
    # Only the first dim is selected; trailing dims pad with slice(None).
    assert result.domain.shape == (2, 5, 6)


# ---------------------------------------------------------------------------
# IndexTransform.__post_init__ shape mismatch error: covered in test_construction_errors
# above? No — the shape mismatch is implicit (the __post_init__ check fires when
# ArrayMap shape != domain shape on input_dimensions), and it's hit by the
# multi-dim oindex test elsewhere. Add an explicit test.
# ---------------------------------------------------------------------------


def test_construction_rejects_shape_mismatch() -> None:
    """ArrayMap.index_array.shape must match the input domain's extents on
    input_dimensions (in order)."""
    with pytest.raises(ValueError, match="does not match expected shape"):
        IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(
                ArrayMap(
                    index_array=np.array([1, 2, 3], dtype=np.intp),
                    input_dimensions=(0,),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# _normalize_basic_selection error paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((2,)), (1, 2, 3)),
            msg="too many indices",
            exception_cls=IndexError,
            id="too-many-indices",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((3, 3, 3)), (..., 0, ...)),
            msg="single ellipsis",
            exception_cls=IndexError,
            id="double-ellipsis",
        ),
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), 1.5),
            msg="unsupported selection type",
            exception_cls=IndexError,
            id="float-not-supported",
        ),
    ],
    ids=lambda c: c.id,
)
def test_basic_indexing_rejects_malformed_selections(
    case: ExpectErr[tuple[IndexTransform, Any]],
) -> None:
    """_normalize_basic_selection error paths: too-many-indices, double-ellipsis,
    and unsupported types like float."""
    transform, selection = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        transform[selection]


# ---------------------------------------------------------------------------
# Transforms with ArrayMap NOT in the last output position.
#
# Several `for m in self.output:` loops in selection_repr, __repr__, basic /
# oindex / vindex apply functions, and _intersect's orthogonal path have an
# `elif isinstance(m, ArrayMap):` branch that, for branch coverage, needs to
# be exercised with an ArrayMap that is NOT the last output (i.e., the loop
# continues to a next iteration after the ArrayMap branch). The fixture below
# constructs a transform with ArrayMap-then-DimensionMap output ordering;
# the tests use it to hit those continuation branches.
# ---------------------------------------------------------------------------


def _arraymap_then_dimensionmap() -> IndexTransform:
    """Helper: a 2-D-input transform whose first output is an ArrayMap and
    whose second output is a DimensionMap. Ensures `for m in output` loops
    encounter an ArrayMap with a next iteration available."""
    return IndexTransform(
        domain=IndexDomain.from_shape((3, 5)),
        output=(
            ArrayMap(
                index_array=np.array([1, 4, 9], dtype=np.intp),
                input_dimensions=(0,),
            ),
            DimensionMap(input_dimension=1, offset=0, stride=1),
        ),
    )


def test_selection_repr_with_arraymap_not_last() -> None:
    """selection_repr output loop visits ArrayMap then continues."""
    t = _arraymap_then_dimensionmap()
    s = t.selection_repr
    assert "{1, 4, 9}" in s
    assert "[0, 5)" in s


def test_repr_with_arraymap_not_last() -> None:
    """__repr__ output loop visits ArrayMap then continues."""
    t = _arraymap_then_dimensionmap()
    s = repr(t)
    assert "out[0] = 0 + 1 * arr(3,)[in[0]]" in s
    assert "out[1] = 0 + 1 * in[1]" in s


def test_translate_with_arraymap_not_last() -> None:
    """IndexTransform.translate output loop visits ArrayMap then continues.

    The shift is applied to every output, so an ArrayMap-then-DimensionMap
    transform produces a (translated ArrayMap, translated DimensionMap)
    pair."""
    t = _arraymap_then_dimensionmap()
    result = t.translate((10, 100))
    assert isinstance(result.output[0], ArrayMap)
    assert result.output[0].offset == 10
    assert isinstance(result.output[1], DimensionMap)
    assert result.output[1].offset == 100


def test_basic_indexing_with_arraymap_not_last() -> None:
    """_apply_basic_indexing output loop visits ArrayMap then continues."""
    t = _arraymap_then_dimensionmap()
    result = t[:, 2:5]
    assert isinstance(result.output[0], ArrayMap)
    assert isinstance(result.output[1], DimensionMap)


def test_oindex_with_arraymap_not_last() -> None:
    """_apply_oindex output loop visits ArrayMap then continues."""
    t = _arraymap_then_dimensionmap()
    result = t.oindex[:, np.array([0, 2, 4], dtype=np.intp)]
    # Two outputs preserved: the original ArrayMap (untouched on its
    # parameterizing dim) and the new ArrayMap created from the DimensionMap.
    assert isinstance(result.output[0], ArrayMap)
    assert isinstance(result.output[1], ArrayMap)


def test_intersect_with_two_uncorrelated_arraymaps_uses_orthogonal_path() -> None:
    """When 2+ ArrayMaps have disjoint input_dimensions (no shared input dim),
    intersect detects no correlation and falls through to the orthogonal path,
    NOT the vectorized path. Also exercises the `for m in output` orthogonal
    loop visiting an ArrayMap that is not the last output."""
    # 2-D input domain (3, 4); two ArrayMaps with disjoint input_dimensions.
    t = IndexTransform(
        domain=IndexDomain.from_shape((3, 4)),
        output=(
            ArrayMap(
                index_array=np.array([0, 5, 10], dtype=np.intp),
                input_dimensions=(0,),
            ),
            ArrayMap(
                index_array=np.array([20, 30, 40, 50], dtype=np.intp),
                input_dimensions=(1,),
            ),
        ),
    )
    # Chunk that includes everything. The orthogonal path filters each
    # ArrayMap independently against its output dim's chunk range.
    chunk = IndexDomain.from_shape((100, 100))
    result = t.intersect(chunk)
    assert result is not None
    restricted, _ = result
    # Both outputs survive as ArrayMaps (orthogonal path preserves them).
    assert isinstance(restricted.output[0], ArrayMap)
    assert isinstance(restricted.output[1], ArrayMap)
