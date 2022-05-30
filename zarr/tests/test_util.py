import re
from unittest import mock

import numpy as np
import pytest

from zarr.core import Array
from zarr.util import (all_equal, flatten, guess_chunks, human_readable_size,
                       info_html_report, info_text_report, is_total_slice,
                       json_dumps, normalize_chunks,
                       normalize_dimension_separator,
                       normalize_fill_value, normalize_order,
                       normalize_resize_args, normalize_shape, retry_call,
                       tree_array_icon, tree_group_icon, tree_get_icon,
                       tree_widget)


def test_normalize_dimension_separator():
    assert None is normalize_dimension_separator(None)
    assert '/' == normalize_dimension_separator('/')
    assert '.' == normalize_dimension_separator('.')
    with pytest.raises(ValueError):
        normalize_dimension_separator('X')


def test_normalize_shape():
    assert (100,) == normalize_shape((100,))
    assert (100,) == normalize_shape([100])
    assert (100,) == normalize_shape(100)
    with pytest.raises(TypeError):
        normalize_shape(None)
    with pytest.raises(ValueError):
        normalize_shape('foo')


def test_normalize_chunks():
    assert (10,) == normalize_chunks((10,), (100,), 1)
    assert (10,) == normalize_chunks([10], (100,), 1)
    assert (10,) == normalize_chunks(10, (100,), 1)
    assert (10, 10) == normalize_chunks((10, 10), (100, 10), 1)
    assert (10, 10) == normalize_chunks(10, (100, 10), 1)
    assert (10, 10) == normalize_chunks((10, None), (100, 10), 1)
    assert (30, 30, 30) == normalize_chunks(30, (100, 20, 10), 1)
    assert (30, 20, 10) == normalize_chunks((30,), (100, 20, 10), 1)
    assert (30, 20, 10) == normalize_chunks((30, None), (100, 20, 10), 1)
    assert (30, 20, 10) == normalize_chunks((30, None, None), (100, 20, 10), 1)
    assert (30, 20, 10) == normalize_chunks((30, 20, None), (100, 20, 10), 1)
    assert (30, 20, 10) == normalize_chunks((30, 20, 10), (100, 20, 10), 1)
    with pytest.raises(ValueError):
        normalize_chunks('foo', (100,), 1)
    with pytest.raises(ValueError):
        normalize_chunks((100, 10), (100,), 1)

    # test auto-chunking
    assert (100,) == normalize_chunks(None, (100,), 1)
    assert (100,) == normalize_chunks(-1, (100,), 1)
    assert (30, 20, 10) == normalize_chunks((30, -1, None), (100, 20, 10), 1)


def test_is_total_slice():

    # 1D
    assert is_total_slice(Ellipsis, (100,))
    assert is_total_slice(slice(None), (100,))
    assert is_total_slice(slice(0, 100), (100,))
    assert not is_total_slice(slice(0, 50), (100,))
    assert not is_total_slice(slice(0, 100, 2), (100,))

    # 2D
    assert is_total_slice(Ellipsis, (100, 100))
    assert is_total_slice(slice(None), (100, 100))
    assert is_total_slice((slice(None), slice(None)), (100, 100))
    assert is_total_slice((slice(0, 100), slice(0, 100)), (100, 100))
    assert not is_total_slice((slice(0, 100), slice(0, 50)), (100, 100))
    assert not is_total_slice((slice(0, 50), slice(0, 100)), (100, 100))
    assert not is_total_slice((slice(0, 50), slice(0, 50)), (100, 100))
    assert not is_total_slice((slice(0, 100, 2), slice(0, 100)), (100, 100))

    with pytest.raises(TypeError):
        is_total_slice('foo', (100,))


def test_normalize_resize_args():

    # 1D
    assert (200,) == normalize_resize_args((100,), 200)
    assert (200,) == normalize_resize_args((100,), (200,))

    # 2D
    assert (200, 100) == normalize_resize_args((100, 100), (200, 100))
    assert (200, 100) == normalize_resize_args((100, 100), (200, None))
    assert (200, 100) == normalize_resize_args((100, 100), 200, 100)
    assert (200, 100) == normalize_resize_args((100, 100), 200, None)

    with pytest.raises(ValueError):
        normalize_resize_args((100,), (200, 100))


def test_human_readable_size():
    assert '100' == human_readable_size(100)
    assert '1.0K' == human_readable_size(2**10)
    assert '1.0M' == human_readable_size(2**20)
    assert '1.0G' == human_readable_size(2**30)
    assert '1.0T' == human_readable_size(2**40)
    assert '1.0P' == human_readable_size(2**50)


def test_normalize_order():
    assert 'F' == normalize_order('F')
    assert 'C' == normalize_order('C')
    assert 'F' == normalize_order('f')
    assert 'C' == normalize_order('c')
    with pytest.raises(ValueError):
        normalize_order('foo')


def test_normalize_fill_value():
    assert b'' == normalize_fill_value(0, dtype=np.dtype('S1'))
    structured_dtype = np.dtype([('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    expect = np.array((b'', 0, 0.), dtype=structured_dtype)[()]
    assert expect == normalize_fill_value(0, dtype=structured_dtype)
    assert '' == normalize_fill_value(0, dtype=np.dtype('U1'))


def test_guess_chunks():
    shapes = (
        (100,),
        (100, 100),
        (1000000,),
        (1000000000,),
        (10000000000000000000000,),
        (10000, 10000),
        (10000000, 1000),
        (1000, 10000000),
        (10000000, 1000, 2),
        (1000, 10000000, 2),
        (10000, 10000, 10000),
        (100000, 100000, 100000),
        (1000000000, 1000000000, 1000000000),
        (0,),
        (0, 0),
        (10, 0),
        (0, 10),
        (1, 2, 0, 4, 5),
    )
    for shape in shapes:
        chunks = guess_chunks(shape, 1)
        assert isinstance(chunks, tuple)
        assert len(chunks) == len(shape)
        # doesn't make any sense to allow chunks to have zero length dimension
        assert all(0 < c <= max(s, 1) for c, s in zip(chunks, shape))

    # ludicrous itemsize
    chunks = guess_chunks((1000000,), 40000000000)
    assert isinstance(chunks, tuple)
    assert (1,) == chunks


def test_info_text_report():
    items = [('foo', 'bar'), ('baz', 'qux')]
    expect = "foo : bar\nbaz : qux\n"
    assert expect == info_text_report(items)


def test_info_html_report():
    items = [('foo', 'bar'), ('baz', 'qux')]
    actual = info_html_report(items)
    assert '<table' == actual[:6]
    assert '</table>' == actual[-8:]


def test_tree_get_icon():
    assert tree_get_icon("Array") == tree_array_icon
    assert tree_get_icon("Group") == tree_group_icon
    with pytest.raises(ValueError):
        tree_get_icon("Baz")


@mock.patch.dict("sys.modules", {"ipytree": None})
def test_tree_widget_missing_ipytree():
    pattern = (
        "Run `pip install zarr[jupyter]` or `conda install ipytree`"
        "to get the required ipytree dependency for displaying the tree "
        "widget. If using jupyterlab<3, you also need to run "
        "`jupyter labextension install ipytree`"
        )
    with pytest.raises(ImportError, match=re.escape(pattern)):
        tree_widget(None, None, None)


def test_retry_call():

    class Fixture:

        def __init__(self, pass_on=1):
            self.c = 0
            self.pass_on = pass_on

        def __call__(self):
            self.c += 1
            if self.c != self.pass_on:
                raise PermissionError()

    for x in range(1, 11):
        # Any number of failures less than 10 will be accepted.
        fixture = Fixture(pass_on=x)
        retry_call(fixture, exceptions=(PermissionError,), wait=0)
        assert fixture.c == x

    def fail(x):
        # Failures after 10 will cause an error to be raised.
        retry_call(Fixture(pass_on=x), exceptions=(Exception,), wait=0)

    for x in range(11, 15):
        pytest.raises(PermissionError, fail, x)


def test_flatten():
    assert list(flatten(['0', ['1', ['2', ['3', [4, ]]]]])) == ['0', '1', '2', '3', 4]
    assert list(flatten('foo')) == ['f', 'o', 'o']
    assert list(flatten(['foo'])) == ['foo']


def test_all_equal():
    assert all_equal(0, np.zeros((10, 10, 10)))
    assert not all_equal(1, np.zeros((10, 10, 10)))

    assert all_equal(1, np.ones((10, 10, 10)))
    assert not all_equal(1, 1 + np.ones((10, 10, 10)))

    assert all_equal(np.nan, np.array([np.nan, np.nan]))
    assert not all_equal(np.nan, np.array([np.nan, 1.0]))

    assert all_equal({'a': -1}, np.array([{'a': -1}, {'a': -1}], dtype='object'))
    assert not all_equal({'a': -1}, np.array([{'a': -1}, {'a': 2}], dtype='object'))

    assert all_equal(np.timedelta64(999, 'D'), np.array([999, 999], dtype='timedelta64[D]'))
    assert not all_equal(np.timedelta64(999, 'D'), np.array([999, 998], dtype='timedelta64[D]'))

    # all_equal(None, *) always returns False
    assert not all_equal(None, np.array([None, None]))
    assert not all_equal(None, np.array([None, 10]))


def test_json_dumps_numpy_dtype():
    assert json_dumps(np.int64(0)) == json_dumps(0)
    assert json_dumps(np.float32(0)) == json_dumps(float(0))
    # Check that we raise the error of the superclass for unsupported object
    with pytest.raises(TypeError):
        json_dumps(Array)
