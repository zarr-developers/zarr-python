import numpy
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.indexing import (
    make_slice_selection,
    normalize_integer_selection,
    oindex,
    oindex_set,
    replace_ellipsis,
    PartialChunkIterator,
)

from zarr.tests.util import CountingDict


def test_normalize_integer_selection():

    assert 1 == normalize_integer_selection(1, 100)
    assert 99 == normalize_integer_selection(-1, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(100, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(1000, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(-1000, 100)


def test_replace_ellipsis():

    # 1D, single item
    assert (0,) == replace_ellipsis(0, (100,))

    # 1D
    assert (slice(None),) == replace_ellipsis(Ellipsis, (100,))
    assert (slice(None),) == replace_ellipsis(slice(None), (100,))
    assert (slice(None, 100),) == replace_ellipsis(slice(None, 100), (100,))
    assert (slice(0, None),) == replace_ellipsis(slice(0, None), (100,))
    assert (slice(None),) == replace_ellipsis((slice(None), Ellipsis), (100,))
    assert (slice(None),) == replace_ellipsis((Ellipsis, slice(None)), (100,))

    # 2D, single item
    assert (0, 0) == replace_ellipsis((0, 0), (100, 100))
    assert (-1, 1) == replace_ellipsis((-1, 1), (100, 100))

    # 2D, single col/row
    assert (0, slice(None)) == replace_ellipsis((0, slice(None)), (100, 100))
    assert (0, slice(None)) == replace_ellipsis((0,), (100, 100))
    assert (slice(None), 0) == replace_ellipsis((slice(None), 0), (100, 100))

    # 2D slice
    assert ((slice(None), slice(None)) ==
            replace_ellipsis(Ellipsis, (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis(slice(None), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((slice(None), slice(None)), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((Ellipsis, slice(None)), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((slice(None), Ellipsis), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((slice(None), Ellipsis, slice(None)), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((Ellipsis, slice(None), slice(None)), (100, 100)))
    assert ((slice(None), slice(None)) ==
            replace_ellipsis((slice(None), slice(None), Ellipsis), (100, 100)))


def test_get_basic_selection_0d():

    # setup
    a = np.array(42)
    z = zarr.create(shape=a.shape, dtype=a.dtype, fill_value=None)
    z[...] = a

    assert_array_equal(a, z.get_basic_selection(Ellipsis))
    assert_array_equal(a, z[...])
    assert 42 == z.get_basic_selection(())
    assert 42 == z[()]

    # test out param
    b = np.zeros_like(a)
    z.get_basic_selection(Ellipsis, out=b)
    assert_array_equal(a, b)

    # test structured array
    value = (b'aaa', 1,  4.2)
    a = np.array(value, dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    z = zarr.create(shape=a.shape, dtype=a.dtype, fill_value=None)
    z[()] = value
    assert_array_equal(a, z.get_basic_selection(Ellipsis))
    assert_array_equal(a, z[...])
    assert a[()] == z.get_basic_selection(())
    assert a[()] == z[()]
    assert b'aaa' == z.get_basic_selection((), fields='foo')
    assert b'aaa' == z['foo']
    assert a[['foo', 'bar']] == z.get_basic_selection((), fields=['foo', 'bar'])
    assert a[['foo', 'bar']] == z['foo', 'bar']
    # test out param
    b = np.zeros_like(a)
    z.get_basic_selection(Ellipsis, out=b)
    assert_array_equal(a, b)
    c = np.zeros_like(a[['foo', 'bar']])
    z.get_basic_selection(Ellipsis, out=c, fields=['foo', 'bar'])
    assert_array_equal(a[['foo', 'bar']], c)


basic_selections_1d = [
    # single value
    42,
    -1,
    # slices
    slice(0, 1050),
    slice(50, 150),
    slice(0, 2000),
    slice(-150, -50),
    slice(-2000, 2000),
    slice(0, 0),  # empty result
    slice(-1, 0),  # empty result
    # total selections
    slice(None),
    Ellipsis,
    (),
    (Ellipsis, slice(None)),
    # slice with step
    slice(None),
    slice(None, None),
    slice(None, None, 1),
    slice(None, None, 10),
    slice(None, None, 100),
    slice(None, None, 1000),
    slice(None, None, 10000),
    slice(0, 1050),
    slice(0, 1050, 1),
    slice(0, 1050, 10),
    slice(0, 1050, 100),
    slice(0, 1050, 1000),
    slice(0, 1050, 10000),
    slice(1, 31, 3),
    slice(1, 31, 30),
    slice(1, 31, 300),
    slice(81, 121, 3),
    slice(81, 121, 30),
    slice(81, 121, 300),
    slice(50, 150),
    slice(50, 150, 1),
    slice(50, 150, 10),
]


basic_selections_1d_bad = [
    # only positive step supported
    slice(None, None, -1),
    slice(None, None, -10),
    slice(None, None, -100),
    slice(None, None, -1000),
    slice(None, None, -10000),
    slice(1050, -1, -1),
    slice(1050, -1, -10),
    slice(1050, -1, -100),
    slice(1050, -1, -1000),
    slice(1050, -1, -10000),
    slice(1050, 0, -1),
    slice(1050, 0, -10),
    slice(1050, 0, -100),
    slice(1050, 0, -1000),
    slice(1050, 0, -10000),
    slice(150, 50, -1),
    slice(150, 50, -10),
    slice(31, 1, -3),
    slice(121, 81, -3),
    slice(-1, 0, -1),
    # bad stuff
    2.3,
    'foo',
    b'xxx',
    None,
    (0, 0),
    (slice(None), slice(None)),
]


def _test_get_basic_selection(a, z, selection):
    expect = a[selection]
    actual = z.get_basic_selection(selection)
    assert_array_equal(expect, actual)
    actual = z[selection]
    assert_array_equal(expect, actual)


# noinspection PyStatementEffect
def test_get_basic_selection_1d():

    # setup
    a = np.arange(1050, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)
    z[:] = a

    for selection in basic_selections_1d:
        _test_get_basic_selection(a, z, selection)

    for selection in basic_selections_1d_bad:
        with pytest.raises(IndexError):
            z.get_basic_selection(selection)
        with pytest.raises(IndexError):
            z[selection]

    with pytest.raises(IndexError):
        z.get_basic_selection([1, 0])


basic_selections_2d = [
    # single row
    42,
    -1,
    (42, slice(None)),
    (-1, slice(None)),
    # single col
    (slice(None), 4),
    (slice(None), -1),
    # row slices
    slice(None),
    slice(0, 1000),
    slice(250, 350),
    slice(0, 2000),
    slice(-350, -250),
    slice(0, 0),  # empty result
    slice(-1, 0),  # empty result
    slice(-2000, 0),
    slice(-2000, 2000),
    # 2D slices
    (slice(None), slice(1, 5)),
    (slice(250, 350), slice(None)),
    (slice(250, 350), slice(1, 5)),
    (slice(250, 350), slice(-5, -1)),
    (slice(250, 350), slice(-50, 50)),
    (slice(250, 350, 10), slice(1, 5)),
    (slice(250, 350), slice(1, 5, 2)),
    (slice(250, 350, 33), slice(1, 5, 3)),
    # total selections
    (slice(None), slice(None)),
    Ellipsis,
    (),
    (Ellipsis, slice(None)),
    (Ellipsis, slice(None), slice(None)),
]


basic_selections_2d_bad = [
    # bad stuff
    2.3,
    'foo',
    b'xxx',
    None,
    (2.3, slice(None)),
    # only positive step supported
    slice(None, None, -1),
    (slice(None, None, -1), slice(None)),
    (0, 0, 0),
    (slice(None), slice(None), slice(None)),
]


# noinspection PyStatementEffect
def test_get_basic_selection_2d():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    for selection in basic_selections_2d:
        _test_get_basic_selection(a, z, selection)

    bad_selections = basic_selections_2d_bad + [
        # integer arrays
        [0, 1],
        (slice(None), [0, 1]),
    ]
    for selection in bad_selections:
        with pytest.raises(IndexError):
            z.get_basic_selection(selection)
        with pytest.raises(IndexError):
            z[selection]
    # check fallback on fancy indexing
    fancy_selection = ([0, 1], [0, 1])
    np.testing.assert_array_equal(z[fancy_selection], [0, 11])


def test_fancy_indexing_fallback_on_get_setitem():
    z = zarr.zeros((20, 20))
    z[[1, 2, 3], [1, 2, 3]] = 1
    np.testing.assert_array_equal(
        z[:4, :4],
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    np.testing.assert_array_equal(
        z[[1, 2, 3], [1, 2, 3]], 1
    )
    # test broadcasting
    np.testing.assert_array_equal(
        z[1, [1, 2, 3]], [1, 0, 0]
    )
    # test 1D fancy indexing
    z2 = zarr.zeros(5)
    z2[[1, 2, 3]] = 1
    np.testing.assert_array_equal(
        z2, [0, 1, 1, 1, 0]
    )


def test_fancy_indexing_doesnt_mix_with_slicing():
    z = zarr.zeros((20, 20))
    with pytest.raises(IndexError):
        z[[1, 2, 3], :] = 2
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(
            z[[1, 2, 3], :], 0
        )


def test_fancy_indexing_doesnt_mix_with_implicit_slicing():
    z2 = zarr.zeros((5, 5, 5))
    with pytest.raises(IndexError):
        z2[[1, 2, 3], [1, 2, 3]] = 2
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(
            z2[[1, 2, 3], [1, 2, 3]], 0
        )
    with pytest.raises(IndexError):
        z2[[1, 2, 3]] = 2
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(
            z2[[1, 2, 3]], 0
        )
    with pytest.raises(IndexError):
        z2[..., [1, 2, 3]] = 2
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(
            z2[..., [1, 2, 3]], 0
        )


def test_set_basic_selection_0d():

    # setup
    v = np.array(42)
    a = np.zeros_like(v)
    z = zarr.zeros_like(v)
    assert_array_equal(a, z)

    # tests
    z.set_basic_selection(Ellipsis, v)
    assert_array_equal(v, z)
    z[...] = 0
    assert_array_equal(a, z)
    z[...] = v
    assert_array_equal(v, z)

    # test structured array
    value = (b'aaa', 1,  4.2)
    v = np.array(value, dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    a = np.zeros_like(v)
    z = zarr.create(shape=a.shape, dtype=a.dtype, fill_value=None)

    # tests
    z.set_basic_selection(Ellipsis, v)
    assert_array_equal(v, z)
    z.set_basic_selection(Ellipsis, a)
    assert_array_equal(a, z)
    z[...] = v
    assert_array_equal(v, z)
    z[...] = a
    assert_array_equal(a, z)
    # with fields
    z.set_basic_selection(Ellipsis, v['foo'], fields='foo')
    assert v['foo'] == z['foo']
    assert a['bar'] == z['bar']
    assert a['baz'] == z['baz']
    z['bar'] = v['bar']
    assert v['foo'] == z['foo']
    assert v['bar'] == z['bar']
    assert a['baz'] == z['baz']
    # multiple field assignment not supported
    with pytest.raises(IndexError):
        z.set_basic_selection(Ellipsis, v[['foo', 'bar']], fields=['foo', 'bar'])
    with pytest.raises(IndexError):
        z[..., 'foo', 'bar'] = v[['foo', 'bar']]


def _test_get_orthogonal_selection(a, z, selection):
    expect = oindex(a, selection)
    actual = z.get_orthogonal_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.oindex[selection]
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
        _test_get_orthogonal_selection(a, z, ix)

    # test errors
    with pytest.raises(IndexError):
        z.oindex[np.zeros(50, dtype=bool)]  # too short
    with pytest.raises(IndexError):
        z.oindex[np.zeros(2000, dtype=bool)]  # too long
    with pytest.raises(IndexError):
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
        # unordered
        ix = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        _test_get_orthogonal_selection(a, z, ix)
        # increasing
        ix.sort()
        _test_get_orthogonal_selection(a, z, ix)
        # decreasing
        ix = ix[::-1]
        _test_get_orthogonal_selection(a, z, ix)

    selections = basic_selections_1d + [
        # test wraparound
        [0, 3, 10, -23, -12, -1],
        # explicit test not sorted
        [3, 105, 23, 127],

    ]
    for selection in selections:
        _test_get_orthogonal_selection(a, z, selection)

    bad_selections = basic_selections_1d_bad + [
        [a.shape[0] + 1],  # out of bounds
        [-(a.shape[0] + 1)],  # out of bounds
        [[2, 4], [6, 8]],  # too many dimensions
    ]
    for selection in bad_selections:
        with pytest.raises(IndexError):
            z.get_orthogonal_selection(selection)
        with pytest.raises(IndexError):
            z.oindex[selection]


def _test_get_orthogonal_selection_2d(a, z, ix0, ix1):
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
        _test_get_orthogonal_selection(a, z, selection)


# noinspection PyStatementEffect
def test_get_orthogonal_selection_2d():

    # setup
    a = np.arange(10000, dtype=int).reshape(1000, 10)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:

        # boolean arrays
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, 0.5, size=a.shape[1]).astype(bool)
        _test_get_orthogonal_selection_2d(a, z, ix0, ix1)

        # mixed int array / bool array
        selections = (
            (ix0, np.nonzero(ix1)[0]),
            (np.nonzero(ix0)[0], ix1),
        )
        for selection in selections:
            _test_get_orthogonal_selection(a, z, selection)

        # integer arrays
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        _test_get_orthogonal_selection_2d(a, z, ix0, ix1)
        ix0.sort()
        ix1.sort()
        _test_get_orthogonal_selection_2d(a, z, ix0, ix1)
        ix0 = ix0[::-1]
        ix1 = ix1[::-1]
        _test_get_orthogonal_selection_2d(a, z, ix0, ix1)

    for selection in basic_selections_2d:
        _test_get_orthogonal_selection(a, z, selection)

    for selection in basic_selections_2d_bad:
        with pytest.raises(IndexError):
            z.get_orthogonal_selection(selection)
        with pytest.raises(IndexError):
            z.oindex[selection]


def _test_get_orthogonal_selection_3d(a, z, ix0, ix1, ix2):
    selections = [
        # single value
        (84, 42, 4),
        (-1, -1, -1),
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
        _test_get_orthogonal_selection(a, z, selection)


def test_get_orthogonal_selection_3d():

    # setup
    a = np.arange(100000, dtype=int).reshape(200, 50, 10)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)
    z[:] = a

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:

        # boolean arrays
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        ix2 = np.random.binomial(1, .5, size=a.shape[2]).astype(bool)
        _test_get_orthogonal_selection_3d(a, z, ix0, ix1, ix2)

        # integer arrays
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        ix2 = np.random.choice(a.shape[2], size=int(a.shape[2] * .5), replace=True)
        _test_get_orthogonal_selection_3d(a, z, ix0, ix1, ix2)
        ix0.sort()
        ix1.sort()
        ix2.sort()
        _test_get_orthogonal_selection_3d(a, z, ix0, ix1, ix2)
        ix0 = ix0[::-1]
        ix1 = ix1[::-1]
        ix2 = ix2[::-1]
        _test_get_orthogonal_selection_3d(a, z, ix0, ix1, ix2)


def test_orthogonal_indexing_edge_cases():

    a = np.arange(6).reshape(1, 2, 3)
    z = zarr.create(shape=a.shape, chunks=(1, 2, 3), dtype=a.dtype)
    z[:] = a

    expect = oindex(a, (0, slice(None), [0, 1, 2]))
    actual = z.oindex[0, :, [0, 1, 2]]
    assert_array_equal(expect, actual)

    expect = oindex(a, (0, slice(None), [True, True, True]))
    actual = z.oindex[0, :, [True, True, True]]
    assert_array_equal(expect, actual)


def _test_set_orthogonal_selection(v, a, z, selection):
    for value in 42, oindex(v, selection), oindex(v, selection).tolist():
        if isinstance(value, list) and value == []:
            # skip these cases as cannot preserve all dimensions
            continue
        # setup expectation
        a[:] = 0
        oindex_set(a, selection, value)
        # long-form API
        z[:] = 0
        z.set_orthogonal_selection(selection, value)
        assert_array_equal(a, z[:])
        # short-form API
        z[:] = 0
        z.oindex[selection] = value
        assert_array_equal(a, z[:])


def test_set_orthogonal_selection_1d():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty(v.shape, dtype=int)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    # test with different degrees of sparseness
    np.random.seed(42)
    for p in 0.5, 0.1, 0.01:

        # boolean arrays
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        _test_set_orthogonal_selection(v, a, z, ix)

        # integer arrays
        ix = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        _test_set_orthogonal_selection(v, a, z, ix)
        ix.sort()
        _test_set_orthogonal_selection(v, a, z, ix)
        ix = ix[::-1]
        _test_set_orthogonal_selection(v, a, z, ix)

    # basic selections
    for selection in basic_selections_1d:
        _test_set_orthogonal_selection(v, a, z, selection)


def _test_set_orthogonal_selection_2d(v, a, z, ix0, ix1):

    selections = [
        # index both axes with array
        (ix0, ix1),
        # mixed indexing with array / slice or int
        (ix0, slice(1, 5)),
        (slice(250, 350), ix1),
        (ix0, 4),
        (42, ix1),
    ]
    for selection in selections:
        _test_set_orthogonal_selection(v, a, z, selection)


def test_set_orthogonal_selection_2d():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:

        # boolean arrays
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        _test_set_orthogonal_selection_2d(v, a, z, ix0, ix1)

        # integer arrays
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        _test_set_orthogonal_selection_2d(v, a, z, ix0, ix1)
        ix0.sort()
        ix1.sort()
        _test_set_orthogonal_selection_2d(v, a, z, ix0, ix1)
        ix0 = ix0[::-1]
        ix1 = ix1[::-1]
        _test_set_orthogonal_selection_2d(v, a, z, ix0, ix1)

    for selection in basic_selections_2d:
        _test_set_orthogonal_selection(v, a, z, selection)


def _test_set_orthogonal_selection_3d(v, a, z, ix0, ix1, ix2):

    selections = (
        # single value
        (84, 42, 4),
        (-1, -1, -1),
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
        _test_set_orthogonal_selection(v, a, z, selection)


def test_set_orthogonal_selection_3d():

    # setup
    v = np.arange(100000, dtype=int).reshape(200, 50, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(60, 20, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:

        # boolean arrays
        ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        ix1 = np.random.binomial(1, .5, size=a.shape[1]).astype(bool)
        ix2 = np.random.binomial(1, .5, size=a.shape[2]).astype(bool)
        _test_set_orthogonal_selection_3d(v, a, z, ix0, ix1, ix2)

        # integer arrays
        ix0 = np.random.choice(a.shape[0], size=int(a.shape[0] * p), replace=True)
        ix1 = np.random.choice(a.shape[1], size=int(a.shape[1] * .5), replace=True)
        ix2 = np.random.choice(a.shape[2], size=int(a.shape[2] * .5), replace=True)
        _test_set_orthogonal_selection_3d(v, a, z, ix0, ix1, ix2)

        # sorted increasing
        ix0.sort()
        ix1.sort()
        ix2.sort()
        _test_set_orthogonal_selection_3d(v, a, z, ix0, ix1, ix2)

        # sorted decreasing
        ix0 = ix0[::-1]
        ix1 = ix1[::-1]
        ix2 = ix2[::-1]
        _test_set_orthogonal_selection_3d(v, a, z, ix0, ix1, ix2)


def _test_get_coordinate_selection(a, z, selection):
    expect = a[selection]
    actual = z.get_coordinate_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.vindex[selection]
    assert_array_equal(expect, actual)


coordinate_selections_1d_bad = [
    # slice not supported
    slice(5, 15),
    slice(None),
    Ellipsis,
    # bad stuff
    2.3,
    'foo',
    b'xxx',
    None,
    (0, 0),
    (slice(None), slice(None)),
]


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
        _test_get_coordinate_selection(a, z, ix)
        ix.sort()
        _test_get_coordinate_selection(a, z, ix)
        ix = ix[::-1]
        _test_get_coordinate_selection(a, z, ix)

    selections = [
        # test single item
        42,
        -1,
        # test wraparound
        [0, 3, 10, -23, -12, -1],
        # test out of order
        [3, 105, 23, 127],  # not monotonically increasing
        # test multi-dimensional selection
        np.array([[2, 4], [6, 8]]),
    ]
    for selection in selections:
        _test_get_coordinate_selection(a, z, selection)

    # test errors
    bad_selections = coordinate_selections_1d_bad + [
        [a.shape[0] + 1],  # out of bounds
        [-(a.shape[0] + 1)],  # out of bounds
    ]
    for selection in bad_selections:
        with pytest.raises(IndexError):
            z.get_coordinate_selection(selection)
        with pytest.raises(IndexError):
            z.vindex[selection]


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
            # single value
            (42, 4),
            (-1, -1),
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
            (42, 4),
        ]
        for selection in selections:
            _test_get_coordinate_selection(a, z, selection)

    # not monotonically increasing (first dim)
    ix0 = [3, 3, 4, 2, 5]
    ix1 = [1, 3, 5, 7, 9]
    _test_get_coordinate_selection(a, z, (ix0, ix1))

    # not monotonically increasing (second dim)
    ix0 = [1, 1, 2, 2, 5]
    ix1 = [1, 3, 2, 1, 0]
    _test_get_coordinate_selection(a, z, (ix0, ix1))

    # multi-dimensional selection
    ix0 = np.array([[1, 1, 2],
                    [2, 2, 5]])
    ix1 = np.array([[1, 3, 2],
                    [1, 0, 0]])
    _test_get_coordinate_selection(a, z, (ix0, ix1))

    with pytest.raises(IndexError):
        selection = slice(5, 15), [1, 2, 3]
        z.get_coordinate_selection(selection)
    with pytest.raises(IndexError):
        selection = [1, 2, 3], slice(5, 15)
        z.get_coordinate_selection(selection)
    with pytest.raises(IndexError):
        selection = Ellipsis, [1, 2, 3]
        z.get_coordinate_selection(selection)
    with pytest.raises(IndexError):
        selection = Ellipsis
        z.get_coordinate_selection(selection)


def _test_set_coordinate_selection(v, a, z, selection):
    for value in 42, v[selection], v[selection].tolist():
        # setup expectation
        a[:] = 0
        a[selection] = value
        # test long-form API
        z[:] = 0
        z.set_coordinate_selection(selection, value)
        assert_array_equal(a, z[:])
        # test short-form API
        z[:] = 0
        z.vindex[selection] = value
        assert_array_equal(a, z[:])


def test_set_coordinate_selection_1d():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty(v.shape, dtype=v.dtype)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 2, 0.5, 0.1, 0.01:
        n = int(a.size * p)
        ix = np.random.choice(a.shape[0], size=n, replace=True)
        _test_set_coordinate_selection(v, a, z, ix)

    # multi-dimensional selection
    ix = np.array([[2, 4], [6, 8]])
    _test_set_coordinate_selection(v, a, z, ix)

    for selection in coordinate_selections_1d_bad:
        with pytest.raises(IndexError):
            z.set_coordinate_selection(selection, 42)
        with pytest.raises(IndexError):
            z.vindex[selection] = 42


def test_set_coordinate_selection_2d():

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
            (42, 4),
            (-1, -1),
            # index both axes with array
            (ix0, ix1),
            # mixed indexing with array / int
            (ix0, 4),
            (42, ix1),
        )
        for selection in selections:
            _test_set_coordinate_selection(v, a, z, selection)

    # multi-dimensional selection
    ix0 = np.array([[1, 2, 3],
                    [4, 5, 6]])
    ix1 = np.array([[1, 3, 2],
                    [2, 0, 5]])
    _test_set_coordinate_selection(v, a, z, (ix0, ix1))


def _test_get_mask_selection(a, z, selection):
    expect = a[selection]
    actual = z.get_mask_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.vindex[selection]
    assert_array_equal(expect, actual)


mask_selections_1d_bad = [
    # slice not supported
    slice(5, 15),
    slice(None),
    Ellipsis,
    # bad stuff
    2.3,
    'foo',
    b'xxx',
    None,
    (0, 0),
    (slice(None), slice(None)),
]


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
        _test_get_mask_selection(a, z, ix)

    # test errors
    bad_selections = mask_selections_1d_bad + [
        np.zeros(50, dtype=bool),  # too short
        np.zeros(2000, dtype=bool),  # too long
        [[True, False], [False, True]],  # too many dimensions
    ]
    for selection in bad_selections:
        with pytest.raises(IndexError):
            z.get_mask_selection(selection)
        with pytest.raises(IndexError):
            z.vindex[selection]


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
        _test_get_mask_selection(a, z, ix)

    # test errors
    with pytest.raises(IndexError):
        z.vindex[np.zeros((1000, 5), dtype=bool)]  # too short
    with pytest.raises(IndexError):
        z.vindex[np.zeros((2000, 10), dtype=bool)]  # too long
    with pytest.raises(IndexError):
        z.vindex[[True, False]]  # wrong no. dimensions


def _test_set_mask_selection(v, a, z, selection):
    a[:] = 0
    z[:] = 0
    a[selection] = v[selection]
    z.set_mask_selection(selection, v[selection])
    assert_array_equal(a, z[:])
    z[:] = 0
    z.vindex[selection] = v[selection]
    assert_array_equal(a, z[:])


def test_set_mask_selection_1d():

    # setup
    v = np.arange(1050, dtype=int)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=100, dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
        _test_set_mask_selection(v, a, z, ix)

    for selection in mask_selections_1d_bad:
        with pytest.raises(IndexError):
            z.set_mask_selection(selection, 42)
        with pytest.raises(IndexError):
            z.vindex[selection] = 42


def test_set_mask_selection_2d():

    # setup
    v = np.arange(10000, dtype=int).reshape(1000, 10)
    a = np.empty_like(v)
    z = zarr.create(shape=a.shape, chunks=(300, 3), dtype=a.dtype)

    np.random.seed(42)
    # test with different degrees of sparseness
    for p in 0.5, 0.1, 0.01:
        ix = np.random.binomial(1, p, size=a.size).astype(bool).reshape(a.shape)
        _test_set_mask_selection(v, a, z, ix)


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

    with pytest.raises(TypeError):
        z.get_basic_selection(Ellipsis, out=[])

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


def test_get_selections_with_fields():

    a = [('aaa', 1, 4.2),
         ('bbb', 2, 8.4),
         ('ccc', 3, 12.6)]
    a = np.array(a, dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    z = zarr.create(shape=a.shape, chunks=2, dtype=a.dtype, fill_value=None)
    z[:] = a

    fields_fixture = [
        'foo',
        ['foo'],
        ['foo', 'bar'],
        ['foo', 'baz'],
        ['bar', 'baz'],
        ['foo', 'bar', 'baz'],
        ['bar', 'foo'],
        ['baz', 'bar', 'foo'],
    ]

    for fields in fields_fixture:

        # total selection
        expect = a[fields]
        actual = z.get_basic_selection(Ellipsis, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[fields[0], fields[1]]
            assert_array_equal(expect, actual)
        if isinstance(fields, str):
            actual = z[..., fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[..., fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # basic selection with slice
        expect = a[fields][0:2]
        actual = z.get_basic_selection(slice(0, 2), fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[0:2, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[0:2, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # basic selection with single item
        expect = a[fields][1]
        actual = z.get_basic_selection(1, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[1, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[1, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # orthogonal selection
        ix = [0, 2]
        expect = a[fields][ix]
        actual = z.get_orthogonal_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.oindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.oindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # coordinate selection
        ix = [0, 2]
        expect = a[fields][ix]
        actual = z.get_coordinate_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.vindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.vindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # mask selection
        ix = [True, False, True]
        expect = a[fields][ix]
        actual = z.get_mask_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.vindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.vindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

    # missing/bad fields
    with pytest.raises(IndexError):
        z.get_basic_selection(Ellipsis, fields=['notafield'])
    with pytest.raises(IndexError):
        z.get_basic_selection(Ellipsis, fields=slice(None))


def test_set_selections_with_fields():

    v = [('aaa', 1, 4.2),
         ('bbb', 2, 8.4),
         ('ccc', 3, 12.6)]
    v = np.array(v, dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    a = np.empty_like(v)
    z = zarr.empty_like(v, chunks=2)

    fields_fixture = [
        'foo',
        [],
        ['foo'],
        ['foo', 'bar'],
        ['foo', 'baz'],
        ['bar', 'baz'],
        ['foo', 'bar', 'baz'],
        ['bar', 'foo'],
        ['baz', 'bar', 'foo'],
    ]

    for fields in fields_fixture:

        # currently multi-field assignment is not supported in numpy, so we won't support
        # it either
        if isinstance(fields, list) and len(fields) > 1:
            with pytest.raises(IndexError):
                z.set_basic_selection(Ellipsis, v, fields=fields)
            with pytest.raises(IndexError):
                z.set_orthogonal_selection([0, 2], v, fields=fields)
            with pytest.raises(IndexError):
                z.set_coordinate_selection([0, 2], v, fields=fields)
            with pytest.raises(IndexError):
                z.set_mask_selection([True, False, True], v, fields=fields)

        else:

            if isinstance(fields, list) and len(fields) == 1:
                # work around numpy does not support multi-field assignment even if there
                # is only one field
                key = fields[0]
            elif isinstance(fields, list) and len(fields) == 0:
                # work around numpy ambiguity about what is a field selection
                key = Ellipsis
            else:
                key = fields

            # setup expectation
            a[:] = ('', 0, 0)
            z[:] = ('', 0, 0)
            assert_array_equal(a, z[:])
            a[key] = v[key]
            # total selection
            z.set_basic_selection(Ellipsis, v[key], fields=fields)
            assert_array_equal(a, z[:])

            # basic selection with slice
            a[:] = ('', 0, 0)
            z[:] = ('', 0, 0)
            a[key][0:2] = v[key][0:2]
            z.set_basic_selection(slice(0, 2), v[key][0:2], fields=fields)
            assert_array_equal(a, z[:])

            # orthogonal selection
            a[:] = ('', 0, 0)
            z[:] = ('', 0, 0)
            ix = [0, 2]
            a[key][ix] = v[key][ix]
            z.set_orthogonal_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])

            # coordinate selection
            a[:] = ('', 0, 0)
            z[:] = ('', 0, 0)
            ix = [0, 2]
            a[key][ix] = v[key][ix]
            z.set_coordinate_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])

            # mask selection
            a[:] = ('', 0, 0)
            z[:] = ('', 0, 0)
            ix = [True, False, True]
            a[key][ix] = v[key][ix]
            z.set_mask_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])


@pytest.mark.parametrize(
    "selection, arr, expected",
    [
        (
            (slice(5, 8, 1), slice(2, 4, 1), slice(0, 100, 1)),
            np.arange(2, 100_002).reshape((100, 10, 100)),
            [
                (5200, 200, (slice(5, 6, 1), slice(2, 4, 1))),
                (6200, 200, (slice(6, 7, 1), slice(2, 4, 1))),
                (7200, 200, (slice(7, 8, 1), slice(2, 4, 1))),
            ],
        ),
        (
            (slice(5, 8, 1), slice(2, 4, 1), slice(0, 5, 1)),
            np.arange(2, 100_002).reshape((100, 10, 100)),
            [
                (5200.0, 5.0, (slice(5, 6, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (5300.0, 5.0, (slice(5, 6, 1), slice(3, 4, 1), slice(0, 5, 1))),
                (6200.0, 5.0, (slice(6, 7, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (6300.0, 5.0, (slice(6, 7, 1), slice(3, 4, 1), slice(0, 5, 1))),
                (7200.0, 5.0, (slice(7, 8, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (7300.0, 5.0, (slice(7, 8, 1), slice(3, 4, 1), slice(0, 5, 1))),
            ],
        ),
        (
            (slice(5, 8, 1), slice(2, 4, 1), slice(0, 5, 1)),
            np.asfortranarray(np.arange(2, 100_002).reshape((100, 10, 100))),
            [
                (5200.0, 5.0, (slice(5, 6, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (5300.0, 5.0, (slice(5, 6, 1), slice(3, 4, 1), slice(0, 5, 1))),
                (6200.0, 5.0, (slice(6, 7, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (6300.0, 5.0, (slice(6, 7, 1), slice(3, 4, 1), slice(0, 5, 1))),
                (7200.0, 5.0, (slice(7, 8, 1), slice(2, 3, 1), slice(0, 5, 1))),
                (7300.0, 5.0, (slice(7, 8, 1), slice(3, 4, 1), slice(0, 5, 1))),
            ],
        ),
        (
            (slice(5, 8, 1), slice(2, 4, 1)),
            np.arange(2, 100_002).reshape((100, 10, 100)),
            [
                (5200, 200, (slice(5, 6, 1), slice(2, 4, 1))),
                (6200, 200, (slice(6, 7, 1), slice(2, 4, 1))),
                (7200, 200, (slice(7, 8, 1), slice(2, 4, 1))),
            ],
        ),
        (
            (slice(0, 10, 1),),
            np.arange(0, 10).reshape((10)),
            [(0, 10, (slice(0, 10, 1),))],
        ),
        ((0,), np.arange(0, 100).reshape((10, 10)), [(0, 10, (slice(0, 1, 1),))]),
        (
            (
                0,
                0,
            ),
            np.arange(0, 100).reshape((10, 10)),
            [(0, 1, (slice(0, 1, 1), slice(0, 1, 1)))],
        ),
        ((0,), np.arange(0, 10).reshape((10)), [(0, 1, (slice(0, 1, 1),))]),
        pytest.param(
            (slice(5, 8, 1), slice(2, 4, 1), slice(0, 5, 1)),
            np.arange(2, 100002).reshape((10, 1, 10000)),
            None,
            marks=[pytest.mark.xfail(reason="slice 2 is out of range")],
        ),
        pytest.param(
            (slice(5, 8, 1), slice(2, 4, 1), slice(0, 5, 1)),
            np.arange(2, 100_002).reshape((10, 10_000)),
            None,
            marks=[pytest.mark.xfail(reason="slice 2 is out of range")],
        ),
    ],
)
def test_PartialChunkIterator(selection, arr, expected):
    PCI = PartialChunkIterator(selection, arr.shape)
    results = list(PCI)
    assert results == expected


def test_slice_selection_uints():
    arr = np.arange(24).reshape((4, 6))
    idx = np.uint64(3)
    slice_sel = make_slice_selection((idx,))
    assert arr[tuple(slice_sel)].shape == (1, 6)


def test_numpy_int_indexing():
    a = np.arange(1050)
    z = zarr.create(shape=1050, chunks=100, dtype=a.dtype)
    z[:] = a
    assert a[42] == z[42]
    assert a[numpy.int64(42)] == z[numpy.int64(42)]


@pytest.mark.parametrize(
    "shape, chunks, ops",
    [
        # 1D test cases
        ((1070,), (50,), [("__getitem__", (slice(200, 400),))]),
        ((1070,), (50,), [("__getitem__", (slice(200, 400, 100),))]),
        ((1070,), (50,), [
            ("__getitem__", (slice(200, 400),)),
            ("__setitem__", (slice(200, 400, 100),)),
        ]),

        # 2D test cases
        ((40, 50), (5, 8), [
            ("__getitem__", (slice(6, 37, 13), (slice(4, 10)))),
            ("__setitem__", (slice(None), (slice(None)))),
        ]),
    ]
)
def test_accessed_chunks(shape, chunks, ops):
    # Test that only the required chunks are accessed during basic selection operations
    # shape: array shape
    # chunks: chunk size
    # ops: list of tuples with (optype, tuple of slices)
    # optype = "__getitem__" or "__setitem__", tuple length must match number of dims
    import itertools

    # Use a counting dict as the backing store so we can track the items access
    store = CountingDict()
    z = zarr.create(shape=shape, chunks=chunks, store=store)

    for ii, (optype, slices) in enumerate(ops):

        # Resolve the slices into the accessed chunks for each dimension
        chunks_per_dim = []
        for N, C, sl in zip(shape, chunks, slices):
            chunk_ind = np.arange(N, dtype=int)[sl] // C
            chunks_per_dim.append(np.unique(chunk_ind))

        # Combine and generate the cartesian product to determine the chunks keys that
        # will be accessed
        chunks_accessed = []
        for comb in itertools.product(*chunks_per_dim):
            chunks_accessed.append(".".join([str(ci) for ci in comb]))

        counts_before = store.counter.copy()

        # Perform the operation
        if optype == "__getitem__":
            z[slices]
        else:
            z[slices] = ii

        # Get the change in counts
        delta_counts = store.counter - counts_before

        # Check that the access counts for the operation have increased by one for all
        # the chunks we expect to be included
        for ci in chunks_accessed:
            assert delta_counts.pop((optype, ci)) == 1

            # If the chunk was partially written to it will also have been read once. We
            # don't determine if the chunk was actually partial here, just that the
            # counts are consistent that this might have happened
            if optype == "__setitem__":
                assert (
                    ("__getitem__", ci) not in delta_counts or
                    delta_counts.pop(("__getitem__", ci)) == 1
                )
        # Check that no other chunks were accessed
        assert len(delta_counts) == 0
