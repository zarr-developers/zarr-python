from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np
import numpy.typing as npt
import pytest

import zarr_indexing.reader as reader_module
from zarr_indexing import (
    ArrayMap,
    ChunkProjection,
    ConstantMap,
    DimensionMap,
    IndexDomain,
    IndexTransform,
    LazyArray,
    ReadContext,
    plan_chunks,
)
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.reader import basic_reader, numpy_reader, unit_step_reader

if TYPE_CHECKING:
    from collections.abc import Callable


class BasicOnlySource:
    def __init__(self, data: np.ndarray[Any, Any]) -> None:
        self.data = data
        self.keys: list[tuple[Any, ...]] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def __getitem__(self, key: tuple[Any, ...]) -> np.ndarray[Any, Any]:
        assert all(isinstance(item, slice) and (item.step or 1) > 0 for item in key)
        self.keys.append(key)
        return self.data[key]


class UnitStepOnlySource(BasicOnlySource):
    """A source that rejects everything but ascending unit-step slices."""

    def __getitem__(self, key: tuple[Any, ...]) -> np.ndarray[Any, Any]:
        assert all(
            isinstance(item, slice) and item.step == 1 and 0 <= item.start <= item.stop <= size
            for item, size in zip(key, self.data.shape, strict=True)
        )
        self.keys.append(key)
        return self.data[key]


SOURCE = np.arange(6 * 7 * 8).reshape(6, 7, 8)
BASE = IndexTransform.from_shape(SOURCE.shape)


SUCCESSFUL_CONTRACT_CASES = (
    pytest.param(
        np.arange(8),
        IndexTransform.from_shape((8,))[3],
        (3,),
        np.zeros((1, 0), dtype=np.intp),
        np.array([[3]], dtype=np.intp),
        np.array(3),
        lambda array: array.lazy[3],
        id="constant",
    ),
    pytest.param(
        np.arange(8),
        IndexTransform.from_shape((8,))[1:8:2],
        (3,),
        np.array([[0], [1], [2], [3]], dtype=np.intp),
        np.array([[1], [3], [5], [7]], dtype=np.intp),
        np.array([1, 3, 5, 7]),
        lambda array: array.lazy[1:8:2],
        id="positive-affine",
    ),
    pytest.param(
        np.arange(8),
        IndexTransform.from_shape((8,))[::-2],
        (3,),
        np.array([[-3], [-2], [-1], [0]], dtype=np.intp),
        np.array([[7], [5], [3], [1]], dtype=np.intp),
        np.array([7, 5, 3, 1]),
        lambda array: array.lazy[::-2],
        id="negative-affine",
    ),
    pytest.param(
        np.arange(8),
        IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(DimensionMap(input_dimension=0, offset=2, stride=0),),
        ),
        (3,),
        np.array([[0], [1], [2]], dtype=np.intp),
        np.array([[2], [2], [2]], dtype=np.intp),
        np.array([2, 2, 2]),
        lambda array: array.lazy.oindex[[2, 2, 2]],
        id="zero-affine",
    ),
    pytest.param(
        np.arange(20).reshape(4, 5),
        IndexTransform.from_shape((4, 5)).oindex[[3, 1, 1], [4, 0]],
        (2, 3),
        np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.intp),
        np.array([[3, 4], [3, 0], [1, 4], [1, 0], [1, 4], [1, 0]], dtype=np.intp),
        np.array([[19, 15], [9, 5], [9, 5]]),
        lambda array: array.lazy.oindex[[3, 1, 1], [4, 0]],
        id="orthogonal-array",
    ),
    pytest.param(
        np.arange(20).reshape(4, 5),
        IndexTransform.from_shape((4, 5)).vindex[[3, 1, 1], [4, 0, 4]],
        (2, 3),
        np.array([[0], [1], [2]], dtype=np.intp),
        np.array([[3, 4], [1, 0], [1, 4]], dtype=np.intp),
        np.array([19, 5, 9]),
        lambda array: array.lazy.vindex[[3, 1, 1], [4, 0, 4]],
        id="correlated-array",
    ),
    pytest.param(
        np.arange(8),
        IndexTransform(
            domain=IndexDomain((4,), (7,)),
            output=(DimensionMap(input_dimension=0, offset=-3, stride=1),),
        ),
        (3,),
        np.array([[4], [5], [6]], dtype=np.intp),
        np.array([[1], [2], [3]], dtype=np.intp),
        np.array([1, 2, 3]),
        None,
        id="non-zero-origin",
    ),
    pytest.param(
        np.arange(8),
        IndexTransform.from_shape((8,))[2:2],
        (3,),
        np.empty((0, 1), dtype=np.intp),
        np.empty((0, 1), dtype=np.intp),
        np.array([], dtype=np.intp),
        lambda array: array.lazy[2:2],
        id="empty",
    ),
)


@pytest.mark.parametrize(
    (
        "source_data",
        "transform",
        "chunk_shape",
        "request_coordinates",
        "storage_coordinates",
        "expected_values",
        "lazy_selection",
    ),
    SUCCESSFUL_CONTRACT_CASES,
)
def test_successful_transform_contract_across_planning_readers_and_lazy_array(
    source_data: np.ndarray[Any, Any],
    transform: IndexTransform,
    chunk_shape: tuple[int, ...],
    request_coordinates: npt.NDArray[np.intp],
    storage_coordinates: npt.NDArray[np.intp],
    expected_values: np.ndarray[Any, Any],
    lazy_selection: Callable[[LazyArray], LazyArray] | None,
) -> None:
    """One literal matrix keeps transform, planning, readers, and wrappers aligned."""
    np.testing.assert_array_equal(transform.apply_many(request_coordinates), storage_coordinates)

    grids = dimension_grids_from_chunks(chunk_shape, source_data.shape)
    reconstructed_pairs = [
        (
            tuple(projection.cell_transform.apply(cell_coordinate)),
            tuple(
                local_coordinate + chunk_origin
                for local_coordinate, chunk_origin in zip(
                    projection.chunk_transform.apply(cell_coordinate),
                    projection.chunk_domain.inclusive_min,
                    strict=True,
                )
            ),
        )
        for projection in plan_chunks(transform, grids)
        for cell_coordinate in _domain_coordinates(projection.cell_transform.domain)
    ]
    expected_pairs = list(
        zip(
            map(tuple, request_coordinates.tolist()),
            map(tuple, storage_coordinates.tolist()),
            strict=True,
        )
    )
    assert sorted(reconstructed_pairs) == sorted(expected_pairs)

    for reader, source in (
        (basic_reader, BasicOnlySource(source_data)),
        (numpy_reader, source_data),
        (unit_step_reader, UnitStepOnlySource(source_data)),
    ):
        out = np.empty(transform.domain.shape, dtype=source_data.dtype)
        assert reader.read_into(source, ReadContext(transform), out) is None
        np.testing.assert_array_equal(out, expected_values)

    if lazy_selection is not None:
        view = lazy_selection(LazyArray.from_numpy(source_data).with_parts(chunk_shape))
        np.testing.assert_array_equal(view.result(), expected_values)


def _domain_coordinates(domain: IndexDomain) -> list[tuple[int, ...]]:
    return [
        tuple(
            position + origin for position, origin in zip(index, domain.inclusive_min, strict=True)
        )
        for index in np.ndindex(*domain.shape)
    ]


def test_read_context_public_annotations_resolve() -> None:
    assert get_type_hints(ReadContext)["projection"] == ChunkProjection | None


READER_CASES = (
    pytest.param(
        SOURCE,
        BASE[1:6:2, 2, ::-2],
        np.array(
            [
                [79, 77, 75, 73],
                [191, 189, 187, 185],
                [303, 301, 299, 297],
            ]
        ),
        id="strided-reversed-nonzero-origin",
    ),
    pytest.param(
        SOURCE,
        BASE.oindex[[5, 1, 1], slice(1, 6), [7, 2]],
        np.array(
            [
                [[295, 290], [303, 298], [311, 306], [319, 314], [327, 322]],
                [[71, 66], [79, 74], [87, 82], [95, 90], [103, 98]],
                [[71, 66], [79, 74], [87, 82], [95, 90], [103, 98]],
            ]
        ),
        id="orthogonal-nonzero-origin",
    ),
    pytest.param(
        SOURCE,
        BASE.vindex[np.array([[5], [1]]), np.array([[2, 4, 0]])],
        np.array(
            [
                [
                    [296, 297, 298, 299, 300, 301, 302, 303],
                    [312, 313, 314, 315, 316, 317, 318, 319],
                    [280, 281, 282, 283, 284, 285, 286, 287],
                ],
                [
                    [72, 73, 74, 75, 76, 77, 78, 79],
                    [88, 89, 90, 91, 92, 93, 94, 95],
                    [56, 57, 58, 59, 60, 61, 62, 63],
                ],
            ]
        ),
        id="correlated-broadcast",
    ),
    pytest.param(
        SOURCE,
        BASE.vindex[[5, 1], [2, 2]],
        np.array(
            [
                [296, 297, 298, 299, 300, 301, 302, 303],
                [72, 73, 74, 75, 76, 77, 78, 79],
            ]
        ),
        id="correlated-vector",
    ),
    pytest.param(
        SOURCE,
        BASE[:, 0:0, :],
        np.empty((6, 0, 8), dtype=SOURCE.dtype),
        id="empty",
    ),
    pytest.param(SOURCE, BASE[2, 3, 4], np.array(140), id="scalar"),
    pytest.param(
        np.arange(8),
        IndexTransform(
            domain=IndexDomain((4,), (7,)),
            output=(DimensionMap(input_dimension=0, offset=2, stride=0),),
        ),
        np.array([2, 2, 2]),
        id="zero-stride-nonzero-origin",
    ),
)


@pytest.mark.parametrize(("source_data", "transform", "expected"), READER_CASES)
@pytest.mark.parametrize("reader_name", ["basic", "numpy", "unit-step"])
def test_builtin_readers_match_the_transform(
    source_data: np.ndarray[Any, Any],
    transform: IndexTransform,
    expected: np.ndarray[Any, Any],
    reader_name: str,
) -> None:
    out = np.empty(transform.domain.shape, dtype=source_data.dtype)
    if reader_name == "basic":
        source = BasicOnlySource(source_data)
        result = basic_reader.read_into(source, ReadContext(transform), out)
        assert len(source.keys) == 1
    elif reader_name == "unit-step":
        source = UnitStepOnlySource(source_data)
        result = unit_step_reader.read_into(source, ReadContext(transform), out)
        assert len(source.keys) == 1
    else:
        result = numpy_reader.read_into(source_data, ReadContext(transform), out)
    assert result is None
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("reader_name", ["basic", "numpy", "unit-step"])
def test_builtin_readers_share_transform_affine_overflow(reader_name: str) -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((1,)),
        output=(ArrayMap(np.array([2**62], dtype=np.intp), stride=4),),
    )

    with pytest.raises(OverflowError, match="outside np.intp"):
        transform.apply_many(np.array([[0]], dtype=np.intp))

    out = np.empty(transform.domain.shape, dtype=np.intp)
    if reader_name == "basic":
        with pytest.raises(OverflowError, match="outside np.intp"):
            basic_reader.read_into(BasicOnlySource(np.arange(1)), ReadContext(transform), out)
    elif reader_name == "unit-step":
        with pytest.raises(OverflowError, match="outside np.intp"):
            unit_step_reader.read_into(
                UnitStepOnlySource(np.arange(1)), ReadContext(transform), out
            )
    else:
        with pytest.raises(OverflowError, match="outside np.intp"):
            numpy_reader.read_into(np.arange(1), ReadContext(transform), out)


def test_empty_domain_composed_fancy_transform_reads_as_empty() -> None:
    """An ArrayMap composed over an empty domain resolves like any other map.

    The composed map is legitimately empty along the vanished axis; the
    resolvers used to fail reshaping it instead of noticing that an empty
    domain selects nothing.
    """
    source_data = np.arange(6).reshape(2, 3)
    view = LazyArray.from_numpy(source_data).lazy.oindex[slice(0, 0), np.array([2, 1, 2, 0])]
    transform = view.lazy.oindex[slice(None), np.array([1, 3, 1])].transform
    assert transform.domain.shape == (0, 3)

    for reader, source in (
        (basic_reader, BasicOnlySource(source_data)),
        (numpy_reader, source_data),
        (unit_step_reader, UnitStepOnlySource(source_data)),
    ):
        out = np.empty(transform.domain.shape, dtype=source_data.dtype)
        assert reader.read_into(source, ReadContext(transform), out) is None


def test_unit_step_reader_reads_through_lazy_array() -> None:
    """The full dialect resolves through a source that only accepts unit-step slices.

    `UnitStepOnlySource` asserts the shape of every key it receives, so each
    selection here also proves no strided, descending, or non-slice key
    reached the source — partitioned and unpartitioned alike.
    """
    selections: tuple[Callable[[LazyArray], LazyArray], ...] = (
        lambda v: v.lazy[1:5, ::2, ::-1],
        lambda v: v.lazy[5:1:-2, None, 3, ::3],
        lambda v: v.lazy.oindex[[3, 0, 3], ::-2, [7, 7]],
        lambda v: v.lazy.vindex[np.array([[0, 5]]), np.array([[6], [0]]), 2],
        lambda v: v.lazy[2:2, :, ::-1],
        lambda v: v.lazy[::5, 6, 1:8:4],
    )
    for select in selections:
        for parts in (None, (2, 3, 8), (6, 7, 1)):
            source = UnitStepOnlySource(SOURCE)
            view = LazyArray(source).with_reader(unit_step_reader)
            if parts is not None:
                view = view.with_parts(parts)
            expected = select(LazyArray.from_numpy(SOURCE)).result()
            np.testing.assert_array_equal(select(view).result(), expected)
            # An empty view allocates without reading; every other one must read.
            assert (len(source.keys) > 0) == (expected.size > 0)


def test_constant_outside_intp_has_transform_planner_reader_error_parity() -> None:
    outside_intp = int(np.iinfo(np.intp).max) + 1
    transform = IndexTransform(
        domain=IndexDomain((), ()),
        output=(ConstantMap(offset=outside_intp),),
    )

    with pytest.raises(OverflowError, match="outside np.intp range"):
        transform.apply_many(np.zeros((1, 0), dtype=np.intp))

    grids = dimension_grids_from_chunks((1,), (1,))
    with pytest.raises(OverflowError, match="outside np.intp range"):
        list(plan_chunks(transform, grids))

    for reader, source in (
        (basic_reader, BasicOnlySource(np.arange(1))),
        (numpy_reader, np.arange(1)),
    ):
        with pytest.raises(OverflowError, match="outside np.intp range"):
            reader.read_into(source, ReadContext(transform), np.empty((), dtype=np.intp))


def test_numpy_reader_narrows_basic_slab_before_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(1_000_000).reshape(1000, 1000)
    transform = IndexTransform.from_shape(source.shape).oindex[:10, [0, 999]]
    seen: list[tuple[int, ...]] = []
    real_take = reader_module._take  # pyright: ignore[reportPrivateUsage]

    def recording_take(array: Any, indices: npt.NDArray[np.intp], axis: int) -> Any:
        seen.append(tuple(int(value) for value in array.shape))
        return real_take(array, indices, axis)

    monkeypatch.setattr(reader_module, "_take", recording_take)
    out = np.empty(transform.domain.shape, dtype=source.dtype)
    assert numpy_reader.read_into(source, ReadContext(transform), out) is None
    assert out.shape == (10, 2)
    np.testing.assert_array_equal(
        out,
        np.array(
            [
                [0, 999],
                [1000, 1999],
                [2000, 2999],
                [3000, 3999],
                [4000, 4999],
                [5000, 5999],
                [6000, 6999],
                [7000, 7999],
                [8000, 8999],
                [9000, 9999],
            ]
        ),
    )
    assert seen
    assert all(math.prod(shape) <= 20_000 for shape in seen), seen


def test_numpy_reader_preserves_a_mask_in_the_supplied_buffer() -> None:
    source = np.ma.masked_greater(np.arange(12).reshape(3, 4), 7)
    transform = IndexTransform.from_shape(source.shape)[:, 1:]
    out = np.ma.masked_all(transform.domain.shape, dtype=source.dtype)
    assert numpy_reader.read_into(source, ReadContext(transform), out) is None
    np.testing.assert_array_equal(np.ma.getmaskarray(out), np.ma.getmaskarray(source[:, 1:]))
    np.testing.assert_array_equal(np.ma.filled(out, 0), np.ma.filled(source[:, 1:], 0))


# ---------------------------------------------------------------------------
# Diagonal gathers — output maps sharing an input axis
# ---------------------------------------------------------------------------


def test_reading_a_diagonal_gather_transform() -> None:
    """Two index arrays bound to the same input axis resolve pointwise.

    No selection dialect produces this transform — it is the hand-built
    diagonal-extraction form — but the reader resolves it through the same
    pointwise path as a correlated gather.
    """
    data = np.arange(30).reshape(5, 6)
    rows = np.array([4, 0, 2])
    cols = np.array([1, 5, 2])
    transform = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=rows),
            ArrayMap(index_array=cols),
        ),
    )

    out = np.empty((3,), dtype=data.dtype)
    basic_reader.read_into(data, ReadContext(transform), out)
    np.testing.assert_array_equal(out, data[rows, cols])

    out = np.empty((3,), dtype=data.dtype)
    numpy_reader.read_into(data, ReadContext(transform), out)
    np.testing.assert_array_equal(out, data[rows, cols])


def test_reading_a_diagonal_gather_with_a_residual_slice_axis() -> None:
    data = np.arange(60).reshape(5, 6, 2)
    rows = np.array([[4], [0], [2]])
    cols = np.array([[1], [5], [2]])
    transform = IndexTransform(
        domain=IndexDomain.from_shape((3, 2)),
        output=(
            ArrayMap(index_array=rows),
            ArrayMap(index_array=cols),
            DimensionMap(input_dimension=1),
        ),
    )

    out = np.empty((3, 2), dtype=data.dtype)
    basic_reader.read_into(data, ReadContext(transform), out)
    np.testing.assert_array_equal(out, data[rows[:, 0], cols[:, 0], :])
