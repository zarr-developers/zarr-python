"""The wrapper defers selection; explicit execution touches the source."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from zarr_indexing import EagerArrayAdapter, IndexDomain, IndexTransform, LazyArray
from zarr_indexing.errors import BoundsCheckError


class RecordingArray:
    def __init__(self) -> None:
        self.data = np.arange(30).reshape(5, 6)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.reads: list[Any] = []

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return self.data[key]


@pytest.mark.parametrize("mode", ["basic", "oindex", "vindex", "iteration"])
def test_selection_defers_reads(mode: str) -> None:
    source = RecordingArray()
    array = LazyArray(source)
    if mode == "basic":
        view = array[1:][::-1, ::2]
        expected = source.data[1:][::-1, ::2]
    elif mode == "oindex":
        view = array.oindex[[4, 1], :][:, ::2]
        expected = source.data[[4, 1]][:, ::2]
    elif mode == "vindex":
        view = array.vindex[[4, 1], [2, 0]][::-1]
        expected = source.data[[4, 1], [2, 0]][::-1]
    else:
        view = list(array)[2]
        expected = source.data[2]
    assert isinstance(view, LazyArray)
    assert source.reads == []
    np.testing.assert_array_equal(view.result(), expected)
    assert source.reads


def test_no_redundant_lazy_accessor() -> None:
    assert not hasattr(LazyArray(np.arange(3)), "lazy")


@pytest.mark.parametrize("selection", [slice(None), slice(0, 0), 0])
def test_eager_adapter_executes_selection(selection: Any) -> None:
    source = RecordingArray()
    view = LazyArray(source)[1:, ::2]
    adapter = EagerArrayAdapter(view)
    assert (adapter.shape, adapter.ndim, adapter.dtype) == (view.shape, view.ndim, view.dtype)
    assert source.reads == []
    result = adapter[selection]
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, source.data[1:, ::2][selection])
    assert not np.shares_memory(result, source.data)


def test_eager_adapter_distinguishes_token_semantics() -> None:
    pytest.importorskip("dask")
    view = LazyArray(np.arange(8))[::2]
    assert EagerArrayAdapter(view).__dask_tokenize__() != view.__dask_tokenize__()


def test_eager_adapter_refuses_no_copy_conversion() -> None:
    with pytest.raises(ValueError, match="copy"):
        EagerArrayAdapter(LazyArray(np.arange(3))).__array__(copy=False)


def test_views_keep_literal_domains_while_keys_stay_positional() -> None:
    source = np.arange(30)
    view = LazyArray(source)[10:20]
    assert view.transform.domain == IndexDomain((10,), (20,))
    nested = view[2:5]
    assert nested.transform.domain == IndexDomain((12,), (15,))
    np.testing.assert_array_equal(nested.result(), source[12:15])
    # NumPy keys are positions in the current view, not literal coordinates.
    assert view[0].result() == source[10]
    assert view[-1].result() == source[19]
    assert nested[-1].result() == source[14]
    np.testing.assert_array_equal(view[::-1][:3].result(), source[10:20][::-1][:3])
    # Equivalent selections built by different routes share one domain.
    assert LazyArray(source)[0:20][5:10].transform == LazyArray(source)[5:10].transform


def test_domain_key_selects_literal_coordinates() -> None:
    source = np.arange(30)
    view = LazyArray(source)[10:20]
    inner = view[IndexDomain((12,), (15,))]
    assert inner.transform.domain == IndexDomain((12,), (15,))
    np.testing.assert_array_equal(inner.result(), source[12:15])
    np.testing.assert_array_equal(LazyArray(source)[IndexDomain((2,), (5,))].result(), source[2:5])
    with pytest.raises(BoundsCheckError):
        view[IndexDomain((0,), (3,))]
    # An empty domain outside the view is refused too: the key names the view's coordinates.
    with pytest.raises(BoundsCheckError):
        view[IndexDomain((50,), (50,))]
    assert view[IndexDomain((15,), (15,))].shape == (0,)
    with pytest.raises(ValueError, match="rank"):
        view[IndexDomain((0, 0), (1, 1))]
    view[IndexDomain((12,), (15,))] = [-1, -2, -3]
    np.testing.assert_array_equal(source[12:15], [-1, -2, -3])


def test_transform_key_composes_onto_the_view() -> None:
    source = np.arange(30)
    view = LazyArray(source)[10:20]
    reversed_key = IndexTransform.identity(view.transform.domain)[::-1]
    np.testing.assert_array_equal(view[reversed_key].result(), source[10:20][::-1])
    rebased = IndexTransform.identity(IndexDomain((12,), (15,))).translate_domain_to((0,))
    composed = view[rebased]
    assert composed.transform.domain == IndexDomain((0,), (3,))
    np.testing.assert_array_equal(composed.result(), source[12:15])
    with pytest.raises((ValueError, BoundsCheckError)):
        view[IndexTransform.identity(IndexDomain((0,), (3,)))]


def test_box_part_views_are_sub_domains_of_their_parent() -> None:
    source = np.arange(40).reshape(4, 10)
    view = LazyArray(source).with_parts((2, 4))[1:, 3:9]
    origin = view.transform.domain.inclusive_min
    for part in view.parts():
        domain = part.view.transform.domain
        assert all(
            lo >= parent_lo and hi <= parent_hi
            for lo, hi, parent_lo, parent_hi in zip(
                domain.inclusive_min,
                domain.exclusive_max,
                view.transform.domain.inclusive_min,
                view.transform.domain.exclusive_max,
                strict=True,
            )
        )
        # Placement is readable from the domain: it is `out_selection`.
        assert part.out_selection == tuple(
            slice(lo - o, hi - o, 1)
            for lo, hi, o in zip(domain.inclusive_min, domain.exclusive_max, origin, strict=True)
        )
        np.testing.assert_array_equal(part.view.result(), source[1:, 3:9][part.out_selection])
        for child in part.view.parts():
            assert child.view.transform.domain == domain
    # A part placed by index arrays cannot carry the request's coordinates: its
    # domain is fresh and zero-origin, and `out_selection` is the placement.
    fancy = LazyArray(source).with_parts((2, 4))[1:, 3:9].oindex[[2, 0], :]
    assembled = np.empty(fancy.shape, dtype=fancy.dtype)
    for part in fancy.parts():
        assert part.view.transform.domain.inclusive_min == (0,) * part.view.ndim
        assembled[part.out_selection] = part.view.result()
    np.testing.assert_array_equal(assembled, source[1:, 3:9][[2, 0], :])


def test_reader_contexts_are_re_based_whatever_the_view_domain() -> None:
    from zarr_indexing.reader import ReadContext

    view = LazyArray(np.arange(30))[10:20][::-1]
    context = ReadContext(view.transform)
    assert context.transform.domain == IndexDomain((0,), (10,))
    assert context.transform.apply((0,)) == view.transform.apply(
        view.transform.domain.inclusive_min
    )


DOMAIN_CASES: list[
    tuple[str, Callable[[LazyArray], LazyArray], tuple[int, ...], tuple[int, ...]]
] = [
    ("whole", lambda v: v[:], (0, 0), (6, 8)),
    ("ellipsis", lambda v: v[...], (0, 0), (6, 8)),
    ("slice", lambda v: v[2:5], (2, 0), (5, 8)),
    ("nested slice", lambda v: v[2:5][1:], (3, 0), (5, 8)),
    ("negative stop", lambda v: v[2:5][:-1], (2, 0), (4, 8)),
    ("empty", lambda v: v[2:5][1:1], (3, 0), (3, 8)),
    ("int drops axis", lambda v: v[2:5][-1], (0,), (8,)),
    ("strided", lambda v: v[:, 1:7:2], (0, 0), (6, 3)),
    ("reversed", lambda v: v[::-1], (-5, 0), (1, 8)),
    ("reversed then slice", lambda v: v[::-1][1:3], (-4, 0), (-2, 8)),
    ("reversed twice", lambda v: v[1:4, 2:6][::-1, ::-1], (-3, -5), (0, -1)),
    ("newaxis", lambda v: v[None, 2:4], (0, 2, 0), (1, 4, 8)),
    ("oindex", lambda v: v.oindex[[3, 1], 2:6], (0, 2), (2, 6)),
    ("oindex on slice", lambda v: v[2:5].oindex[[1, 0], :], (0, 0), (2, 8)),
    ("vindex", lambda v: v.vindex[[1, 2], [3, 4]], (0,), (2,)),
    ("domain key", lambda v: v[2:5][IndexDomain((3, 0), (4, 8))], (3, 0), (4, 8)),
]


@pytest.mark.parametrize(
    ("case", "select", "lo", "hi"), DOMAIN_CASES, ids=[c[0] for c in DOMAIN_CASES]
)
def test_indexing_preserves_the_literal_domain(
    case: str, select: Callable[[LazyArray], LazyArray], lo: tuple[int, ...], hi: tuple[int, ...]
) -> None:
    """A view's domain is the literal domain TensorStore-style composition gives it.

    Slices keep their coordinates, nested slices compose them, integers drop
    axes, `None` and index arrays open fresh zero-origin axes, and reversals
    run negative. Values are unaffected: the same positional keys select the
    same NumPy elements.
    """
    source = np.arange(48).reshape(6, 8)
    view = select(LazyArray(source))
    expected = IndexDomain(lo, hi)
    assert view.transform.domain == expected, case
    assert view.shape == expected.shape
    # The literal frame is live: the domain's own first coordinate addresses the
    # view's first element, and `ReadContext` still hands readers a zero origin.
    if view.size:
        np.testing.assert_array_equal(
            view.result().flat[0], source[view.transform.apply(expected.inclusive_min)]
        )
