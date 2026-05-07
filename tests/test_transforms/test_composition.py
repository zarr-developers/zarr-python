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
    output=(ArrayMap(index_array=_array_outer_arr, offset=5, stride=2),),
)

# Inner = ArrayMap with various outers.
_array_inner_arr = np.array([10, 20, 30], dtype=np.intp)
_array_inner = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(ArrayMap(index_array=_array_inner_arr, offset=0, stride=1),),
)
_constant_outer_1 = IndexTransform(
    domain=IndexDomain.from_shape((5,)),
    output=(ConstantMap(offset=1),),
)
_array_outer_for_array_inner = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(ArrayMap(index_array=np.array([0, 2, 1], dtype=np.intp), offset=0, stride=1),),
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
                            index_array=np.array([[10, 20], [30, 40], [50, 60]], dtype=np.intp)
                        ),
                    ),
                ),
            ),
            msg="not yet supported",
            exception_cls=NotImplementedError,
            id="multi-d-array-inner-non-constant-outer",
        ),
    ],
    ids=lambda c: c.id,
)
def test_compose_errors(case: ExpectErr[tuple[IndexTransform, IndexTransform]]) -> None:
    """compose raises on rank mismatch and on the unsupported multi-d-array compose path."""
    outer, inner = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        compose(outer, inner)
