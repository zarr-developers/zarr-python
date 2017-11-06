# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, eq_ as eq


from zarr.indexing import normalize_integer_selection, normalize_slice_selection, \
    replace_ellipsis, ix_, oindex
import zarr


def test_normalize_integer_selection():

    eq(1, normalize_integer_selection(1, 100))
    eq(99, normalize_integer_selection(-1, 100))
    with assert_raises(IndexError):
        normalize_integer_selection(100, 100)
    with assert_raises(IndexError):
        normalize_integer_selection(1000, 100)
    with assert_raises(IndexError):
        normalize_integer_selection(-1000, 100)


def test_normalize_slice_selection():

    eq(slice(0, 100, 1), normalize_slice_selection(slice(None), 100))
    eq(slice(0, 100, 1), normalize_slice_selection(slice(None, 100), 100))
    eq(slice(0, 100, 1), normalize_slice_selection(slice(0, None), 100))
    eq(slice(0, 100, 1), normalize_slice_selection(slice(0, 1000), 100))
    eq(slice(99, 100, 1), normalize_slice_selection(slice(-1, None), 100))
    eq(slice(98, 99, 1), normalize_slice_selection(slice(-2, -1), 100))
    eq(slice(10, 10, 1), normalize_slice_selection(slice(10, 0), 100))
    with assert_raises(IndexError):
        normalize_slice_selection(slice(100, None), 100)
    with assert_raises(IndexError):
        normalize_slice_selection(slice(1000, 2000), 100)
    with assert_raises(IndexError):
        normalize_slice_selection(slice(-1000, 0), 100)


def test_replace_ellipsis():

    # 1D, single item
    eq((0,), replace_ellipsis(0, (100,)))

    # 1D
    eq((slice(None),), replace_ellipsis(Ellipsis, (100,)))
    eq((slice(None),), replace_ellipsis(slice(None), (100,)))
    eq((slice(None, 100),), replace_ellipsis(slice(None, 100), (100,)))
    eq((slice(0, None),), replace_ellipsis(slice(0, None), (100,)))
    eq((slice(None),), replace_ellipsis((slice(None), Ellipsis), (100,)))
    eq((slice(None),), replace_ellipsis((Ellipsis, slice(None)), (100,)))

    # 2D, single item
    eq((0, 0), replace_ellipsis((0, 0), (100, 100)))
    eq((-1, 1), replace_ellipsis((-1, 1), (100, 100)))

    # 2D, single col/row
    eq((0, slice(None)), replace_ellipsis((0, slice(None)), (100, 100)))
    eq((0, slice(None)), replace_ellipsis((0,), (100, 100)))
    eq((slice(None), 0), replace_ellipsis((slice(None), 0), (100, 100)))

    # 2D slice
    eq((slice(None), slice(None)),
       replace_ellipsis(Ellipsis, (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis(slice(None), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((slice(None), slice(None)), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((Ellipsis, slice(None)), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((slice(None), Ellipsis), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((slice(None), Ellipsis, slice(None)), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((Ellipsis, slice(None), slice(None)), (100, 100)))
    eq((slice(None), slice(None)),
       replace_ellipsis((slice(None), slice(None), Ellipsis), (100, 100)))


def _test_get_orthogonal_selection_1d_common(a, z, ix):
    expect = a[ix]
    actual = z.get_orthogonal_selection(ix)
    assert_array_equal(expect, actual)
    actual = z.oindex[ix]
    assert_array_equal(expect, actual)
    # for 1d arrays, also available via __getitem__
    actual = z[ix]
    assert_array_equal(expect, actual)


# noinspection PyStatementEffect
def test_get_orthogonal_selection_1d_bool():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        _test_get_orthogonal_selection_1d_common(a, z, ix)

    # test errors
    with assert_raises(IndexError):
        z.oindex[np.zeros(50, dtype=bool)]  # too short
    with assert_raises(IndexError):
        z.oindex[np.zeros(2000, dtype=bool)]  # too long
    with assert_raises(IndexError):
        z.oindex[[[True, False], [False, True]]]  # too many dimensions


# noinspection PyStatementEffect
def test_get_orthogonal_selection_1d_int():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        _test_get_orthogonal_selection_1d_common(a, z, ix)
        ix.sort()
        _test_get_orthogonal_selection_1d_common(a, z, ix)

    # test wraparound
    ix = [0, 3, 10, -23, -12, -1]
    expect = a[ix]
    actual = z.oindex[ix]
    assert_array_equal(expect, actual)

    # explicit test not sorted
    ix = [3, 105, 23, 127]  # not monotonically increasing
    expect = a[ix]
    actual = z.oindex[ix]
    assert_array_equal(expect, actual)

    # test errors
    with assert_raises(IndexError):
        ix = [a.shape[0] + 1]  # out of bounds
        z.oindex[ix]
    with assert_raises(IndexError):
        ix = [-(a.shape[0] + 1)]  # out of bounds
        z.oindex[ix]
    with assert_raises(IndexError):
        ix = [[2, 4], [6, 8]]  # too many dimensions
        z.oindex[ix]


def test_get_orthogonal_selection_1d_slice_with_step():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    selections = [
        slice(0, 1050),
        slice(0, 1050, 1),
        slice(0, 1050, 10),
        slice(0, 1050, 100),
        slice(0, 1050, 1000),
        slice(50, 150, 1),
        slice(50, 150, 10),
        slice(50, 150, 100),
    ]
    for selection in selections:
        expect = a[selection]
        actual = z.get_orthogonal_selection(selection)
        assert_array_equal(expect, actual)
        actual = z.oindex[selection]
        assert_array_equal(expect, actual)
        # for 1d arrays also available via __getitem__
        actual = z[selection]
        assert_array_equal(expect, actual)


def _test_get_orthogonal_selection_2d_common(a, z, ix0, ix1):

    selections = [
        # index both axes with array
        (ix0, ix1),
        # mixed indexing with array / slice
        (ix0, slice(1, 5)),
        (ix0, slice(1, 5, 2)),
        (slice(250, 350), ix1),
        (slice(250, 350, 10), ix1),
        # mixed indexing with array / int
        (ix0, 4),
        (42, ix1),
    ]

    for selection in selections:
        expect = oindex(a, selection)
        actual = z.get_orthogonal_selection(selection)
        assert_array_equal(expect, actual)
        actual = z.oindex[selection]
        assert_array_equal(expect, actual)


def test_get_orthogonal_selection_2d_bool():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, 0.5, size=a.shape[1]).astype(bool)

        # main tests
        _test_get_orthogonal_selection_2d_common(a, z, ix0, ix1)

        # mixed int array / bool array
        selections = (
            (ix0, np.nonzero(ix1)[0]),
            (np.nonzero(ix0)[0], ix1),
        )
        for selection in selections:
            expect = oindex(a, selection)
            actual = z.oindex[selection]
            assert_array_equal(expect, actual)


def test_get_orthogonal_selection_2d_int():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        _test_get_orthogonal_selection_2d_common(a, z, ix0, ix1)
        ix0.sort()
        ix1.sort()
        _test_get_orthogonal_selection_2d_common(a, z, ix0, ix1)


def _test_get_orthogonal_selection_3d_common(a, z, ix0, ix1, ix2):

    selections = [
        # index all axes with array
        (ix0, ix1, ix2),
        # mixed indexing with single array / slices
        (ix0, slice(15, 25), slice(1, 5)),
        (slice(50, 70), ix1, slice(1, 5)),
        (slice(50, 70), slice(15, 25), ix2),
        (ix0, slice(15, 25, 5), slice(1, 5, 2)),
        (slice(50, 70, 3), ix1, slice(1, 5, 2)),
        (slice(50, 70, 3), slice(15, 25, 5), ix2),
        # mixed indexing with single array / ints
        (ix0, 42, 4),
        (84, ix1, 4),
        (84, 42, ix2),
        # mixed indexing with single array / slice / int
        (ix0, slice(15, 25), 4),
        (42, ix1, slice(1, 5)),
        (slice(50, 70), 42, ix2),
        # mixed indexing with two array / slice
        (ix0, ix1, slice(1, 5)),
        (slice(50, 70), ix1, ix2),
        (ix0, slice(15, 25), ix2),
        # mixed indexing with two array / integer
        (ix0, ix1, 4),
        (42, ix1, ix2),
        (ix0, 42, ix2),
    ]
    for selection in selections:
        expect = oindex(a, selection)
        actual = z.get_orthogonal_selection(selection)
        assert_array_equal(expect, actual)
        actual = z.oindex[selection]
        assert_array_equal(expect, actual)


def test_get_orthogonal_selection_3d_bool():

    # setup
    a = np.arange(100000, dtype=int).reshape(200, 50, 10)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        ix2 = np.random.binomial(1, .5, size=a.shape[2]).astype(bool)
        _test_get_orthogonal_selection_3d_common(a, z, ix0, ix1, ix2)


def test_orthogonal_indexing_edge_cases():

    a = np.arange(6).reshape(1, 2, 3)
    z = zarr.create(shape=a.shape, chunks=(1, 2, 3), dtype=a.dtype)
    z[:] = a

    expect = a[ix_([0], range(2), [0, 1, 2])].squeeze(axis=0)
    actual = z.oindex[0, :, [0, 1, 2]]
    assert_array_equal(expect, actual)

    expect = a[ix_([0], range(2), [True, True, True])].squeeze(axis=0)
    actual = z.oindex[0, :, [True, True, True]]
    assert_array_equal(expect, actual)


def test_get_orthogonal_selection_3d_int():

    # setup
    a = np.arange(100000, dtype=int).reshape(200, 50, 10)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        ix2 = np.random.choice(a.shape[2], size=int(a.shape[2] * .5), replace=True)
        _test_get_orthogonal_selection_3d_common(a, z, ix0, ix1, ix2)
        ix0.sort()
        ix1.sort()
        ix2.sort()
        _test_get_orthogonal_selection_3d_common(a, z, ix0, ix1, ix2)


def _test_set_orthogonal_selection_1d_common(v, a, z, ix):
    a[:] = 0
    a[ix] = v[ix]
    z[:] = 0
    z.oindex[ix] = v[ix]
    assert_array_equal(a, z[:])
    z[:] = 0
    z.set_orthogonal_selection(ix, v[ix])
    assert_array_equal(a, z[:])
    # also available via __getitem__ for 1d arrays
    z[:] = 0
    z[ix] = v[ix]
    assert_array_equal(a, z[:])


def test_set_orthogonal_selection_1d_bool():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty(v.shape, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        _test_set_orthogonal_selection_1d_common(v, a, z, ix)


def test_set_orthogonal_selection_1d_int():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty(v.shape, dtype=v.dtype)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        _test_set_orthogonal_selection_1d_common(v, a, z, ix)
        ix.sort()
        _test_set_orthogonal_selection_1d_common(v, a, z, ix)


def _test_set_orthogonal_selection_2d_common(v, a, z, ix0, ix1):

    selections = (
        # index both axes with array
        (ix0, ix1),
        # mixed indexing with array / slice or int
        (ix0, slice(1, 5)),
        (slice(250, 350), ix1),
        (ix0, 4),
        (42, ix1),
    )
    for selection in selections:
        a[:] = 0
        a[ix_(*selection)] = v[ix_(*selection)]
        z[:] = 0
        z.oindex[selection] = oindex(v, selection)
        assert_array_equal(a, z[:])
        z[:] = 0
        z.set_orthogonal_selection(selection, oindex(v, selection))
        assert_array_equal(a, z[:])


def test_set_orthogonal_selection_2d_bool():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        _test_set_orthogonal_selection_2d_common(v, a, z, ix0, ix1)


def test_set_orthogonal_selection_2d_int():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        _test_set_orthogonal_selection_2d_common(v, a, z, ix0, ix1)
        ix0.sort()
        ix1.sort()
        _test_set_orthogonal_selection_2d_common(v, a, z, ix0, ix1)


def _test_set_orthogonal_selection_3d_common(v, a, z, ix0, ix1, ix2):

    selections = (
        # index all axes with bool array
        (ix0, ix1, ix2),
        # mixed indexing with single bool array / slice or int
        (ix0, slice(15, 25), slice(1, 5)),
        (slice(50, 70), ix1, slice(1, 5)),
        (slice(50, 70), slice(15, 25), ix2),
        (ix0, 42, 4),
        (84, ix1, 4),
        (84, 42, ix2),
        (ix0, slice(15, 25), 4),
        (slice(50, 70), ix1, 4),
        (slice(50, 70), 42, ix2),
        # indexing with two arrays / slice
        (ix0, ix1, slice(1, 5)),
        # indexing with two arrays / integer
        (ix0, ix1, 4),
    )
    for selection in selections:
        a[:] = 0
        a[ix_(*selection)] = v[ix_(*selection)]
        z[:] = 0
        z.oindex[selection] = oindex(v, selection)
        assert_array_equal(a, z[:])
        z[:] = 0
        z.set_orthogonal_selection(selection, oindex(v, selection))
        assert_array_equal(a, z[:])


def test_set_orthogonal_selection_3d_bool():

    # setup
    v = np.arange(100000, dtype=int).reshape(200, 50, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        ix2 = np.random.binomial(1, .5, size=a.shape[2]).astype(bool)
        _test_set_orthogonal_selection_3d_common(v, a, z, ix0, ix1, ix2)


def test_set_orthogonal_selection_3d_int():

    # setup
    v = np.arange(100000, dtype=int).reshape(200, 50, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        ix2 = np.random.choice(a.shape[2], size=int(a.shape[2] * .5), replace=True)
        _test_set_orthogonal_selection_3d_common(v, a, z, ix0, ix1, ix2)
        ix0.sort()
        ix1.sort()
        ix2.sort()
        _test_set_orthogonal_selection_3d_common(v, a, z, ix0, ix1, ix2)


# noinspection PyStatementEffect
def test_get_coordinate_selection_1d():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix = np.random.choice(a.shape[0], size=n, replace=True)
        expect = a[ix]
        actual = z.get_coordinate_selection(ix)
        assert_array_equal(expect, actual)
        actual = z.vindex[ix]
        assert_array_equal(expect, actual)
        ix.sort()
        expect = a[ix]
        actual = z.get_coordinate_selection(ix)
        assert_array_equal(expect, actual)
        actual = z.vindex[ix]
        assert_array_equal(expect, actual)

    # test single item
    ix = 42
    expect = a[ix]
    actual = z.get_coordinate_selection(ix)
    assert_array_equal(expect, actual)

    # test wraparound
    ix = [0, 3, 10, -23, -12, -1]
    expect = a[ix]
    actual = z.get_coordinate_selection(ix)
    assert_array_equal(expect, actual)

    # test out of order
    ix = [3, 105, 23, 127]  # not monotonically increasing
    expect = a[ix]
    actual = z.get_coordinate_selection(ix)
    assert_array_equal(expect, actual)

    # test errors
    with assert_raises(IndexError):
        ix = [a.shape[0] + 1]  # out of bounds
        z.get_coordinate_selection(ix)
    with assert_raises(IndexError):
        ix = [-(a.shape[0] + 1)]  # out of bounds
        z.get_coordinate_selection(ix)
    with assert_raises(IndexError):
        ix = [[2, 4], [6, 8]]  # too many dimensions
        z.get_coordinate_selection(ix)
    with assert_raises(IndexError):
        ix = slice(5, 15)
        z.get_coordinate_selection(ix)
    with assert_raises(IndexError):
        ix = Ellipsis
        z.get_coordinate_selection(ix)


def test_get_coordinate_selection_2d():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix0 = np.random.choice(a.shape[0], size=n, replace=True)
        ix1 = np.random.choice(a.shape[1], size=n, replace=True)
        selections = [
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
            (42, 4),
        ]

        for selection in selections:
            expect = a[selection]
            actual = z.get_coordinate_selection(selection)
            assert_array_equal(expect, actual)
            actual = z.vindex[selection]
            assert_array_equal(expect, actual)

        srt = np.lexsort((ix0, ix1))
        ix0 = ix0[srt]
        ix1 = ix1[srt]
        selections = [
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
            (42, 4),
        ]

        for selection in selections:
            expect = a[selection]
            actual = z.get_coordinate_selection(selection)
            assert_array_equal(expect, actual)
            actual = z.vindex[selection]
            assert_array_equal(expect, actual)

    # not monotonically increasing (first dim)
    ix0 = [3, 3, 4, 2, 5]
    ix1 = [1, 3, 5, 7, 9]
    expect = a[ix0, ix1]
    actual = z.get_coordinate_selection((ix0, ix1))
    assert_array_equal(expect, actual)

    # not monotonically increasing (second dim)
    ix0 = [1, 1, 2, 2, 5]
    ix1 = [1, 3, 2, 1, 0]
    expect = a[ix0, ix1]
    actual = z.get_coordinate_selection((ix0, ix1))
    assert_array_equal(expect, actual)

    with assert_raises(IndexError):
        selection = slice(5, 15), [1, 2, 3]
        z.get_coordinate_selection(selection)
    with assert_raises(IndexError):
        selection = [1, 2, 3], slice(5, 15)
        z.get_coordinate_selection(selection)
    with assert_raises(IndexError):
        selection = Ellipsis, [1, 2, 3]
        z.get_coordinate_selection(selection)


def test_set_coordinate_selection_1d_int():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty(v.shape, dtype=v.dtype)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix = np.random.choice(a.shape[0], size=n, replace=True)

        a[:] = 0
        a[ix] = v[ix]
        z[:] = 0
        z.vindex[ix] = v[ix]
        assert_array_equal(a, z[:])
        z[:] = 0
        z.set_coordinate_selection(ix, v[ix])
        assert_array_equal(a, z[:])


def test_set_coordinate_selection_2d_int():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix0 = np.random.choice(a.shape[0], size=n, replace=True)
        ix1 = np.random.choice(a.shape[1], size=n, replace=True)

        selections = (
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
        )

        for selection in selections:
            a[:] = 0
            a[selection] = v[selection]
            z[:] = 0
            z.vindex[selection] = v[selection]
            assert_array_equal(a, z[:])
            z[:] = 0
            z.set_coordinate_selection(selection, v[selection])
            assert_array_equal(a, z[:])


# noinspection PyStatementEffect
def test_get_mask_selection_1d():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)

        expect = a[ix]
        actual = z.get_mask_selection(ix)
        assert_array_equal(expect, actual)
        actual = z.vindex[ix]
        assert_array_equal(expect, actual)
        # for 1d arrays, also available via __getitem__
        actual = z[ix]
        assert_array_equal(expect, actual)

    # test errors
    with assert_raises(IndexError):
        z.vindex[np.zeros(50, dtype=bool)]  # too short
    with assert_raises(IndexError):
        z.vindex[np.zeros(2000, dtype=bool)]  # too long
    with assert_raises(IndexError):
        z.vindex[[[True, False], [False, True]]]  # too many dimensions


# noinspection PyStatementEffect
def test_get_mask_selection_2d():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.size).astype(bool).reshape(a.shape)
        expect = a[ix]
        actual = z.get_mask_selection(ix)
        assert_array_equal(expect, actual)
        actual = z.vindex[ix]
        assert_array_equal(expect, actual)

    # test errors
    with assert_raises(IndexError):
        z.vindex[np.zeros((1000, 5), dtype=bool)]  # too short
    with assert_raises(IndexError):
        z.vindex[np.zeros((2000, 10), dtype=bool)]  # too long
    with assert_raises(IndexError):
        z.vindex[[True, False]]  # wrong no. dimensions


def test_set_mask_selection_1d():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)

        a[:] = 0
        z[:] = 0
        a[ix] = v[ix]
        z.set_mask_selection(ix, v[ix])
        assert_array_equal(a, z[:])
        z[:] = 0
        z.vindex[ix] = v[ix]
        assert_array_equal(a, z[:])
        # for 1d arrays, also available via __setitem__
        z[:] = 0
        z[ix] = v[ix]
        assert_array_equal(a, z[:])


def test_set_mask_selection_2d():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.size).astype(bool).reshape(a.shape)

        a[:] = 0
        z[:] = 0
        a[ix] = v[ix]
        z.set_mask_selection(ix, v[ix])
        assert_array_equal(a, z[:])
        z[:] = 0
        z.vindex[ix] = v[ix]
        assert_array_equal(a, z[:])


def test_get_selection_out():

    # basic selections
    a = np.arange(1050)
    z = zarr.create(shape=1050, chunks=100, dtype=a.dtype)
    z[:] = a
    selections = [
        slice(50, 150),
        slice(0, 1050),
        slice(1, 2),
    ]
    for selection in selections:
        expect = a[selection]
        out = zarr.create(shape=expect.shape, chunks=10, dtype=expect.dtype, fill_value=0)
        z.get_basic_selection(selection, out=out)
        assert_array_equal(expect, out[:])

    # orthogonal selections
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a
    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        selections = [
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / slice
            (ix0, slice(1, 5)),
            (slice(250, 350), ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
            # mixed int array / bool array
            (ix0, np.nonzero(ix1)[0]),
            (np.nonzero(ix0)[0], ix1),
        ]
        for selection in selections:
            expect = oindex(a, selection)
            # out = zarr.create(shape=expect.shape, chunks=10, dtype=expect.dtype,
            #                         fill_value=0)
            out = np.zeros(expect.shape, dtype=expect.dtype)
            z.get_orthogonal_selection(selection, out=out)
            assert_array_equal(expect, out[:])

    # coordinate selections
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a
    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix0 = np.random.choice(a.shape[0], size=n, replace=True)
        ix1 = np.random.choice(a.shape[1], size=n, replace=True)
        selections = [
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
        ]
        for selection in selections:
            expect = a[selection]
            out = np.zeros(expect.shape, dtype=expect.dtype)
            z.get_coordinate_selection(selection, out=out)
            assert_array_equal(expect, out[:])


# TODO selection with fields
