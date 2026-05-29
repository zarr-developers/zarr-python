from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_transforms.conftest import Expect, ExpectErr
from zarr.core._transforms.composition import compose
from zarr.core._transforms.domain import IndexDomain
from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core._transforms.transform import IndexTransform

# Inner = ConstantMap: result is always ConstantMap regardless of outer.
_constant_inner = IndexTransform(
    domain=IndexDomain.from_shape((5,)),
    output=(ConstantMap(offset=42),),
)
_identity_outer_5 = IndexTransform.from_shape((5,))

# Inner = DimensionMap with various outers.
_dimension_inner_0_10_3 = IndexTransform(
    domain=IndexDomain.from_shape((10,)),
    output=(DimensionMap(input_dimension=0, offset=10, stride=3),),
)
_constant_outer_5 = IndexTransform(
    domain=IndexDomain.from_shape((10,)),
    output=(ConstantMap(offset=5),),
)
_dimension_outer_0_5_2 = IndexTransform(
    domain=IndexDomain.from_shape((10,)),
    output=(DimensionMap(input_dimension=0, offset=5, stride=2),),
)
_array_outer_arr = np.array([0, 2, 4], dtype=np.intp)
_array_outer = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(ArrayMap(index_array=_array_outer_arr, input_dimensions=(0,), offset=5, stride=2),),
)

# Inner = ArrayMap with various outers.
_array_inner_arr = np.array([10, 20, 30], dtype=np.intp)
_array_inner = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(ArrayMap(index_array=_array_inner_arr, input_dimensions=(0,), offset=0, stride=1),),
)
_constant_outer_1 = IndexTransform(
    domain=IndexDomain.from_shape((5,)),
    output=(ConstantMap(offset=1),),
)
_array_outer_for_array_inner = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(
        ArrayMap(
            index_array=np.array([0, 2, 1], dtype=np.intp),
            input_dimensions=(0,),
            offset=0,
            stride=1,
        ),
    ),
)


@pytest.mark.parametrize(
    "case",
    [
        # Inner = ConstantMap. Result is constant with the inner's offset, regardless of outer.
        Expect(
            input=(_identity_outer_5, _constant_inner),
            expected={"kind": ConstantMap, "offset": 42},
            id="constant-inner-identity-outer",
        ),
        # Inner = DimensionMap.
        Expect(
            input=(_constant_outer_5, _dimension_inner_0_10_3),
            expected={"kind": ConstantMap, "offset": 25},
            id="dimension-inner-constant-outer",
        ),
        Expect(
            input=(_dimension_outer_0_5_2, _dimension_inner_0_10_3),
            expected={
                "kind": DimensionMap,
                "offset": 25,
                "stride": 6,
                "input_dimension": 0,
            },
            id="dimension-inner-dimension-outer",
        ),
        Expect(
            input=(_array_outer, _dimension_inner_0_10_3),
            expected={
                "kind": ArrayMap,
                "offset": 25,
                "stride": 6,
                "index_array": _array_outer_arr,
            },
            id="dimension-inner-array-outer",
        ),
        # Inner = ArrayMap.
        Expect(
            input=(_constant_outer_1, _array_inner),
            expected={"kind": ConstantMap, "offset": 20},
            id="array-inner-constant-outer",
        ),
        Expect(
            input=(
                # Outer: 1-D identity-ish, input domain (4,), DimensionMap with
                # offset=1 stride=1. Intermediate produced: [1, 2, 3, 4].
                IndexTransform(
                    domain=IndexDomain.from_shape((4,)),
                    output=(DimensionMap(input_dimension=0, offset=1, stride=1),),
                ),
                # Inner: ArrayMap of length 5 on intermediate dim 0.
                # arr[1..4] = [200, 300, 400, 500].
                IndexTransform(
                    domain=IndexDomain.from_shape((5,)),
                    output=(
                        ArrayMap(
                            index_array=np.array([100, 200, 300, 400, 500], dtype=np.intp),
                            input_dimensions=(0,),
                        ),
                    ),
                ),
            ),
            expected={
                "kind": ArrayMap,
                "offset": 0,
                "stride": 1,
                "index_array": np.array([200, 300, 400, 500], dtype=np.intp),
            },
            id="array-inner-dimension-outer",
        ),
        Expect(
            input=(_array_outer_for_array_inner, _array_inner),
            expected={
                "kind": ArrayMap,
                "offset": 0,
                "stride": 1,
                "index_array": np.array([10, 30, 20], dtype=np.intp),
            },
            id="array-inner-array-outer",
        ),
    ],
    ids=lambda c: c.id,
)
def test_compose_success(
    case: Expect[tuple[IndexTransform, IndexTransform], dict[str, Any]],
) -> None:
    """compose dispatches over (inner_kind, outer_kind) pairs and produces the expected result map."""
    outer, inner = case.input
    result = compose(outer, inner)
    assert len(result.output) == 1
    out0 = result.output[0]
    assert isinstance(out0, case.expected["kind"])
    if "offset" in case.expected:
        assert out0.offset == case.expected["offset"]
    if "stride" in case.expected:
        assert isinstance(out0, (DimensionMap, ArrayMap))
        assert out0.stride == case.expected["stride"]
    if "input_dimension" in case.expected:
        assert isinstance(out0, DimensionMap)
        assert out0.input_dimension == case.expected["input_dimension"]
    if "index_array" in case.expected:
        assert isinstance(out0, ArrayMap)
        np.testing.assert_array_equal(out0.index_array, case.expected["index_array"])


def test_compose_2d_identity() -> None:
    """Composing two identity 2D transforms yields a 2D identity."""
    a = IndexTransform.from_shape((10, 20))
    b = IndexTransform.from_shape((10, 20))
    result = compose(a, b)
    assert result.domain.shape == (10, 20)
    for i, m in enumerate(result.output):
        assert isinstance(m, DimensionMap)
        assert m.input_dimension == i
        assert m.offset == 0
        assert m.stride == 1


def test_compose_mixed_map_types() -> None:
    """Outer has heterogeneous output maps; each composes independently with its inner image."""
    outer = IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(
            ConstantMap(offset=5),
            DimensionMap(input_dimension=0, offset=0, stride=1),
        ),
    )
    inner = IndexTransform(
        domain=IndexDomain.from_shape((10, 10)),
        output=(
            DimensionMap(input_dimension=0, offset=2, stride=3),
            DimensionMap(input_dimension=1, offset=0, stride=1),
        ),
    )
    result = compose(outer, inner)
    assert isinstance(result.output[0], ConstantMap)
    assert result.output[0].offset == 17
    assert isinstance(result.output[1], DimensionMap)
    assert result.output[1].input_dimension == 0
    assert result.output[1].offset == 0
    assert result.output[1].stride == 1


def test_compose_chains_associatively() -> None:
    """compose(a, compose(b, c)) yields the same offsets/strides as composing in order."""
    a = IndexTransform.from_shape((100,))
    b = IndexTransform(
        domain=IndexDomain.from_shape((100,)),
        output=(DimensionMap(input_dimension=0, offset=10, stride=1),),
    )
    c = IndexTransform(
        domain=IndexDomain.from_shape((100,)),
        output=(DimensionMap(input_dimension=0, offset=5, stride=2),),
    )
    abc = compose(a, compose(b, c))
    assert isinstance(abc.output[0], DimensionMap)
    assert abc.output[0].offset == 25
    assert abc.output[0].stride == 2


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexTransform.from_shape((10,)), IndexTransform.from_shape((10, 20))),
            msg="rank",
            exception_cls=ValueError,
            id="outer-output-rank-vs-inner-input-rank-mismatch",
        ),
        ExpectErr(
            input=(
                # Outer is a non-constant 2D identity transform.
                IndexTransform.from_shape((3, 2)),
                # Inner has a 2D ArrayMap. _compose_array's general multi-dim
                # path raises NotImplementedError for this combination.
                IndexTransform(
                    domain=IndexDomain.from_shape((3, 2)),
                    output=(
                        ArrayMap(
                            index_array=np.array([[10, 20], [30, 40], [50, 60]], dtype=np.intp),
                            input_dimensions=(0, 1),
                        ),
                    ),
                ),
            ),
            msg="not yet supported",
            exception_cls=NotImplementedError,
            id="multi-d-array-inner-non-constant-outer",
        ),
        ExpectErr(
            input=(
                # Outer with mixed types: ConstantMap on dim 0, DimensionMap on dim 1.
                # Outer is NOT all-constant, so the early-return path is skipped.
                IndexTransform(
                    domain=IndexDomain.from_shape((4,)),
                    output=(
                        ConstantMap(offset=2),
                        DimensionMap(input_dimension=0, offset=0, stride=1),
                    ),
                ),
                # Inner: 1-D ArrayMap referencing outer's dim 0 (the ConstantMap).
                # _compose_array reaches the 1-D path; outer.output[0] is ConstantMap,
                # which falls through both inner elifs to NotImplementedError.
                IndexTransform(
                    domain=IndexDomain.from_shape((5, 4)),
                    output=(
                        ArrayMap(
                            index_array=np.array([10, 20, 30, 40, 50], dtype=np.intp),
                            input_dimensions=(0,),
                        ),
                    ),
                ),
            ),
            msg="not yet supported",
            exception_cls=NotImplementedError,
            id="single-input-dim-points-at-constantmap-with-mixed-outer",
        ),
    ],
    ids=lambda c: c.id,
)
def test_compose_errors(case: ExpectErr[tuple[IndexTransform, IndexTransform]]) -> None:
    """compose raises on rank mismatch and on the unsupported multi-d-array compose path."""
    outer, inner = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        compose(outer, inner)


# ---------------------------------------------------------------------------
# Associativity property test.
#
# An IndexTransform models a function from input coords to output coords;
# function composition is associative by definition. Verify the implementation
# preserves that algebraic property by sampling random affine triples
# `(a, b, c)` with compatible ranks and checking that
#   compose(compose(a, b), c)
# evaluates the same as
#   compose(a, compose(b, c))
# at randomly-chosen points in `a`'s domain.
#
# Restricted to DimensionMap + ConstantMap outputs (the affine subset).
# ArrayMap composition has implementation-level branching that depends on
# outer structure, and would need a more careful generator to avoid the
# NotImplementedError path; saved for a follow-up.
# ---------------------------------------------------------------------------

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def _evaluate(transform: IndexTransform, user_coord: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate a transform at a single input coordinate.

    Restricted to DimensionMap + ConstantMap outputs; `ArrayMap` is unsupported
    here because the property test only generates affine triples.
    """
    storage: list[int] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            storage.append(m.offset)
        elif isinstance(m, DimensionMap):
            storage.append(m.offset + m.stride * user_coord[m.input_dimension])
        else:
            raise TypeError(f"property test should not generate {type(m).__name__}; got {m!r}")
    return tuple(storage)


def _affine_output_map(input_rank: int, draw: st.DrawFn) -> ConstantMap | DimensionMap:
    """Generate one ConstantMap or DimensionMap output map.

    DimensionMap requires input_rank >= 1; falls back to ConstantMap otherwise.
    Offsets and strides are kept small to avoid integer overflow during
    repeated composition. Strides are positive (DimensionMap rejects
    non-positive strides at construction).
    """
    if input_rank == 0:
        return ConstantMap(offset=draw(st.integers(min_value=-10, max_value=10)))
    kind = draw(st.sampled_from(["constant", "dimension"]))
    if kind == "constant":
        return ConstantMap(offset=draw(st.integers(min_value=-10, max_value=10)))
    return DimensionMap(
        input_dimension=draw(st.integers(min_value=0, max_value=input_rank - 1)),
        offset=draw(st.integers(min_value=-10, max_value=10)),
        stride=draw(st.integers(min_value=1, max_value=3)),
    )


@st.composite
def _affine_transform(draw: st.DrawFn, input_rank: int, output_rank: int) -> IndexTransform:
    """Generate an affine IndexTransform with the requested ranks."""
    domain_shape = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(input_rank))
    domain = IndexDomain.from_shape(domain_shape)
    output = tuple(_affine_output_map(input_rank, draw) for _ in range(output_rank))
    return IndexTransform(domain=domain, output=output)


@st.composite
def _affine_triple(
    draw: st.DrawFn,
) -> tuple[IndexTransform, IndexTransform, IndexTransform]:
    """Generate three rank-compatible affine transforms (a, b, c)."""
    m = draw(st.integers(min_value=1, max_value=3))  # a's input rank
    n = draw(st.integers(min_value=1, max_value=3))  # a's output / b's input rank
    p = draw(st.integers(min_value=1, max_value=3))  # b's output / c's input rank
    q = draw(st.integers(min_value=1, max_value=3))  # c's output rank
    a = draw(_affine_transform(input_rank=m, output_rank=n))
    b = draw(_affine_transform(input_rank=n, output_rank=p))
    c = draw(_affine_transform(input_rank=p, output_rank=q))
    return a, b, c


@settings(max_examples=200, deadline=None)
@given(triple=_affine_triple(), data=st.data())
def test_compose_is_associative(
    triple: tuple[IndexTransform, IndexTransform, IndexTransform],
    data: st.DataObject,
) -> None:
    """For affine transforms, compose(compose(a,b),c) and compose(a,compose(b,c))
    evaluate identically at every point in a's domain."""
    a, b, c = triple
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))

    # Sanity: both compositions agree on rank and domain.
    assert left.input_rank == right.input_rank
    assert left.output_rank == right.output_rank
    assert left.domain == right.domain

    # Sample several points from a's domain and compare evaluations at each.
    # 5 coords per triple raises probabilistic coverage at negligible cost.
    for _ in range(5):
        if a.input_rank == 0:
            coord: tuple[int, ...] = ()
        else:
            coord = tuple(
                data.draw(
                    st.integers(
                        min_value=a.domain.inclusive_min[d],
                        max_value=a.domain.exclusive_max[d] - 1,
                    )
                )
                for d in range(a.input_rank)
            )
            assume(a.domain.contains(coord))
        assert _evaluate(left, coord) == _evaluate(right, coord)
