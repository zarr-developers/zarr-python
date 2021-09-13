import inspect
import json
import math
import numbers
from textwrap import TextWrapper
import mmap
import time

import numpy as np
from asciitree import BoxStyle, LeftAligned
from asciitree.traversal import Traversal
from numcodecs.compat import ensure_ndarray, ensure_text
from numcodecs.registry import codec_registry
from numcodecs.blosc import cbuffer_sizes, cbuffer_metainfo

from typing import Any, Callable, Dict, Optional, Tuple, Union


# codecs to use for object dtype convenience API
object_codecs = {
    str.__name__: 'vlen-utf8',
    bytes.__name__: 'vlen-bytes',
    'array': 'vlen-array',
}


def json_dumps(o: Any) -> bytes:
    """Write JSON in a consistent, human-readable way."""
    return json.dumps(o, indent=4, sort_keys=True, ensure_ascii=True,
                      separators=(',', ': ')).encode('ascii')


def json_loads(s: str) -> Dict[str, Any]:
    """Read JSON in a consistent way."""
    return json.loads(ensure_text(s, 'ascii'))


def normalize_shape(shape) -> Tuple[int]:
    """Convenience function to normalize the `shape` argument."""

    if shape is None:
        raise TypeError('shape is None')

    # handle 1D convenience form
    if isinstance(shape, numbers.Integral):
        shape = (int(shape),)

    # normalize
    shape = tuple(int(s) for s in shape)
    return shape


# code to guess chunk shape, adapted from h5py

CHUNK_BASE = 256*1024  # Multiplier by which chunks are adjusted
CHUNK_MIN = 128*1024  # Soft lower limit (128k)
CHUNK_MAX = 64*1024*1024  # Hard upper limit


def guess_chunks(shape: Tuple[int, ...], typesize: int) -> Tuple[int, ...]:
    """
    Guess an appropriate chunk layout for an array, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """

    ndims = len(shape)
    # require chunks to have non-zero length for all dimensions
    chunks = np.maximum(np.array(shape, dtype='=f8'), 1)

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.product(chunks)*typesize
    target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))

    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        # 2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.product(chunks)*typesize

        if (chunk_bytes < target_size or
                abs(chunk_bytes-target_size)/target_size < 0.5) and \
                chunk_bytes < CHUNK_MAX:
            break

        if np.product(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        chunks[idx % ndims] = math.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)


def normalize_chunks(
    chunks: Any, shape: Tuple[int, ...], typesize: int
) -> Tuple[int, ...]:
    """Convenience function to normalize the `chunks` argument for an array
    with the given `shape`."""

    # N.B., expect shape already normalized

    # handle auto-chunking
    if chunks is None or chunks is True:
        return guess_chunks(shape, typesize)

    # handle no chunking
    if chunks is False:
        return shape

    # handle 1D convenience form
    if isinstance(chunks, numbers.Integral):
        chunks = tuple(int(chunks) for _ in shape)

    # handle bad dimensionality
    if len(chunks) > len(shape):
        raise ValueError('too many dimensions in chunks')

    # handle underspecified chunks
    if len(chunks) < len(shape):
        # assume chunks across remaining dimensions
        chunks += shape[len(chunks):]

    # handle None or -1 in chunks
    if -1 in chunks or None in chunks:
        chunks = tuple(s if c == -1 or c is None else int(c)
                       for s, c in zip(shape, chunks))

    return tuple(chunks)


def normalize_dtype(dtype: Union[str, np.dtype], object_codec) -> Tuple[np.dtype, Any]:

    # convenience API for object arrays
    if inspect.isclass(dtype):
        dtype = dtype.__name__  # type: ignore
    if isinstance(dtype, str):
        # allow ':' to delimit class from codec arguments
        tokens = dtype.split(':')
        key = tokens[0]
        if key in object_codecs:
            dtype = np.dtype(object)
            if object_codec is None:
                codec_id = object_codecs[key]
                if len(tokens) > 1:
                    args = tokens[1].split(',')
                else:
                    args = []
                try:
                    object_codec = codec_registry[codec_id](*args)
                except KeyError:  # pragma: no cover
                    raise ValueError('codec %r for object type %r is not '
                                     'available; please provide an '
                                     'object_codec manually' % (codec_id, key))
            return dtype, object_codec

    dtype = np.dtype(dtype)

    # don't allow generic datetime64 or timedelta64, require units to be specified
    if dtype == np.dtype('M8') or dtype == np.dtype('m8'):
        raise ValueError('datetime64 and timedelta64 dtypes with generic units '
                         'are not supported, please specify units (e.g., "M8[ns]")')

    return dtype, object_codec


# noinspection PyTypeChecker
def is_total_slice(item, shape: Tuple[int]) -> bool:
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimize __setitem__ operations on the Chunk
    class."""

    # N.B., assume shape is normalized

    if item == Ellipsis:
        return True
    if item == slice(None):
        return True
    if isinstance(item, slice):
        item = item,
    if isinstance(item, tuple):
        return all(
            (isinstance(s, slice) and
                ((s == slice(None)) or
                 ((s.stop - s.start == l) and (s.step in [1, None]))))
            for s, l in zip(item, shape)
        )
    else:
        raise TypeError('expected slice or tuple of slices, found %r' % item)


def normalize_resize_args(old_shape, *args):

    # normalize new shape argument
    if len(args) == 1:
        new_shape = args[0]
    else:
        new_shape = args
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    else:
        new_shape = tuple(new_shape)
    if len(new_shape) != len(old_shape):
        raise ValueError('new shape must have same number of dimensions')

    # handle None in new_shape
    new_shape = tuple(s if n is None else int(n)
                      for s, n in zip(old_shape, new_shape))

    return new_shape


def human_readable_size(size) -> str:
    if size < 2**10:
        return '%s' % size
    elif size < 2**20:
        return '%.1fK' % (size / float(2**10))
    elif size < 2**30:
        return '%.1fM' % (size / float(2**20))
    elif size < 2**40:
        return '%.1fG' % (size / float(2**30))
    elif size < 2**50:
        return '%.1fT' % (size / float(2**40))
    else:
        return '%.1fP' % (size / float(2**50))


def normalize_order(order: str) -> str:
    order = str(order).upper()
    if order not in ['C', 'F']:
        raise ValueError("order must be either 'C' or 'F', found: %r" % order)
    return order


def normalize_dimension_separator(sep: Optional[str]) -> Optional[str]:
    if sep in (".", "/", None):
        return sep
    else:
        raise ValueError(
            "dimension_separator must be either '.' or '/', found: %r" % sep)


def normalize_fill_value(fill_value, dtype: np.dtype):

    if fill_value is None or dtype.hasobject:
        # no fill value
        pass
    elif fill_value == 0:
        # this should be compatible across numpy versions for any array type, including
        # structured arrays
        fill_value = np.zeros((), dtype=dtype)[()]

    elif dtype.kind == 'U':
        # special case unicode because of encoding issues on Windows if passed through numpy
        # https://github.com/alimanfoo/zarr/pull/172#issuecomment-343782713

        if not isinstance(fill_value, str):
            raise ValueError('fill_value {!r} is not valid for dtype {}; must be a '
                             'unicode string'.format(fill_value, dtype))

    else:
        try:
            if isinstance(fill_value, bytes) and dtype.kind == 'V':
                # special case for numpy 1.14 compatibility
                fill_value = np.array(fill_value, dtype=dtype.str).view(dtype)[()]
            else:
                fill_value = np.array(fill_value, dtype=dtype)[()]

        except Exception as e:
            # re-raise with our own error message to be helpful
            raise ValueError('fill_value {!r} is not valid for dtype {}; nested '
                             'exception: {}'.format(fill_value, dtype, e))

    return fill_value


def normalize_storage_path(path: Union[str, bytes, None]) -> str:

    # handle bytes
    if isinstance(path, bytes):
        path = str(path, 'ascii')

    # ensure str
    if path is not None and not isinstance(path, str):
        path = str(path)

    if path:

        # convert backslash to forward slash
        path = path.replace('\\', '/')

        # ensure no leading slash
        while len(path) > 0 and path[0] == '/':
            path = path[1:]

        # ensure no trailing slash
        while len(path) > 0 and path[-1] == '/':
            path = path[:-1]

        # collapse any repeated slashes
        previous_char = None
        collapsed = ''
        for char in path:
            if char == '/' and previous_char == '/':
                pass
            else:
                collapsed += char
            previous_char = char
        path = collapsed

        # don't allow path segments with just '.' or '..'
        segments = path.split('/')
        if any([s in {'.', '..'} for s in segments]):
            raise ValueError("path containing '.' or '..' segment not allowed")

    else:
        path = ''

    return path


def buffer_size(v) -> int:
    return ensure_ndarray(v).nbytes


def info_text_report(items: Dict[Any, Any]) -> str:
    keys = [k for k, v in items]
    max_key_len = max(len(k) for k in keys)
    report = ''
    for k, v in items:
        wrapper = TextWrapper(width=80,
                              initial_indent=k.ljust(max_key_len) + ' : ',
                              subsequent_indent=' '*max_key_len + ' : ')
        text = wrapper.fill(str(v))
        report += text + '\n'
    return report


def info_html_report(items) -> str:
    report = '<table class="zarr-info">'
    report += '<tbody>'
    for k, v in items:
        report += '<tr>' \
                  '<th style="text-align: left">%s</th>' \
                  '<td style="text-align: left">%s</td>' \
                  '</tr>' \
                  % (k, v)
    report += '</tbody>'
    report += '</table>'
    return report


class InfoReporter(object):

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        items = self.obj.info_items()
        return info_text_report(items)

    def _repr_html_(self):
        items = self.obj.info_items()
        return info_html_report(items)


class TreeNode(object):

    def __init__(self, obj, depth=0, level=None):
        self.obj = obj
        self.depth = depth
        self.level = level

    def get_children(self):
        if hasattr(self.obj, 'values'):
            if self.level is None or self.depth < self.level:
                depth = self.depth + 1
                return [TreeNode(o, depth=depth, level=self.level)
                        for o in self.obj.values()]
        return []

    def get_text(self):
        name = self.obj.name.split("/")[-1] or "/"
        if hasattr(self.obj, 'shape'):
            name += ' {} {}'.format(self.obj.shape, self.obj.dtype)
        return name

    def get_type(self):
        return type(self.obj).__name__


class TreeTraversal(Traversal):

    def get_children(self, node):
        return node.get_children()

    def get_root(self, tree):
        return tree

    def get_text(self, node):
        return node.get_text()


tree_group_icon = 'folder'
tree_array_icon = 'table'


def tree_get_icon(stype: str) -> str:
    if stype == "Array":
        return tree_array_icon
    elif stype == "Group":
        return tree_group_icon
    else:
        raise ValueError("Unknown type: %s" % stype)


def tree_widget_sublist(node, root=False, expand=False):
    import ipytree

    result = ipytree.Node()
    result.icon = tree_get_icon(node.get_type())
    if root or (expand is True) or (isinstance(expand, int) and node.depth < expand):
        result.opened = True
    else:
        result.opened = False
    result.name = node.get_text()
    result.nodes = [tree_widget_sublist(c, expand=expand) for c in node.get_children()]
    result.disabled = True

    return result


def tree_widget(group, expand, level):
    try:
        import ipytree
    except ImportError as error:
        raise ImportError(
            "{}: Run `pip install zarr[jupyter]` or `conda install ipytree`"
            "to get the required ipytree dependency for displaying the tree "
            "widget. If using jupyterlab<3, you also need to run "
            "`jupyter labextension install ipytree`".format(error)
        )

    result = ipytree.Tree()
    root = TreeNode(group, level=level)
    result.add_node(tree_widget_sublist(root, root=True, expand=expand))

    return result


class TreeViewer(object):

    def __init__(self, group, expand=False, level=None):

        self.group = group
        self.expand = expand
        self.level = level

        self.text_kwargs = dict(
            horiz_len=2,
            label_space=1,
            indent=1
        )

        self.bytes_kwargs = dict(
            UP_AND_RIGHT="+",
            HORIZONTAL="-",
            VERTICAL="|",
            VERTICAL_AND_RIGHT="+"
        )

        self.unicode_kwargs = dict(
            UP_AND_RIGHT="\u2514",
            HORIZONTAL="\u2500",
            VERTICAL="\u2502",
            VERTICAL_AND_RIGHT="\u251C"
        )

    def __bytes__(self):
        drawer = LeftAligned(
            traverse=TreeTraversal(),
            draw=BoxStyle(gfx=self.bytes_kwargs, **self.text_kwargs)
        )
        root = TreeNode(self.group, level=self.level)
        result = drawer(root)

        # Unicode characters slip in on Python 3.
        # So we need to straighten that out first.
        result = result.encode()

        return result

    def __unicode__(self):
        drawer = LeftAligned(
            traverse=TreeTraversal(),
            draw=BoxStyle(gfx=self.unicode_kwargs, **self.text_kwargs)
        )
        root = TreeNode(self.group, level=self.level)
        return drawer(root)

    def __repr__(self):
        return self.__unicode__()

    def _ipython_display_(self):
        tree = tree_widget(self.group, expand=self.expand, level=self.level)
        tree._ipython_display_()
        return tree


def check_array_shape(param, array, shape):
    if not hasattr(array, 'shape'):
        raise TypeError('parameter {!r}: expected an array-like object, got {!r}'
                        .format(param, type(array)))
    if array.shape != shape:
        raise ValueError('parameter {!r}: expected array with shape {!r}, got {!r}'
                         .format(param, shape, array.shape))


def is_valid_python_name(name):
    from keyword import iskeyword
    return name.isidentifier() and not iskeyword(name)


class NoLock(object):
    """A lock that doesn't lock."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


nolock = NoLock()


class PartialReadBuffer:
    def __init__(self, store_key, chunk_store):
        self.chunk_store = chunk_store
        # is it fsstore or an actual fsspec map object
        assert hasattr(self.chunk_store, "map")
        self.map = self.chunk_store.map
        self.fs = self.chunk_store.fs
        self.store_key = store_key
        self.buff = None
        self.nblocks = None
        self.start_points = None
        self.n_per_block = None
        self.start_points_max = None
        self.read_blocks = set()

        _key_path = self.map._key_to_str(store_key)
        _key_path = _key_path.split('/')
        _chunk_path = [self.chunk_store._normalize_key(_key_path[-1])]
        _key_path = '/'.join(_key_path[:-1] + _chunk_path)
        self.key_path = _key_path

    def prepare_chunk(self):
        assert self.buff is None
        header = self.fs.read_block(self.key_path, 0, 16)
        nbytes, self.cbytes, blocksize = cbuffer_sizes(header)
        typesize, _shuffle, _memcpyd = cbuffer_metainfo(header)
        self.buff = mmap.mmap(-1, self.cbytes)
        self.buff[0:16] = header
        self.nblocks = nbytes / blocksize
        self.nblocks = (
            int(self.nblocks)
            if self.nblocks == int(self.nblocks)
            else int(self.nblocks + 1)
        )
        if self.nblocks == 1:
            self.buff = self.read_full()
            return
        start_points_buffer = self.fs.read_block(
            self.key_path, 16, int(self.nblocks * 4)
        )
        self.start_points = np.frombuffer(
            start_points_buffer, count=self.nblocks, dtype=np.int32
        )
        self.start_points_max = self.start_points.max()
        self.buff[16: (16 + (self.nblocks * 4))] = start_points_buffer
        self.n_per_block = blocksize / typesize

    def read_part(self, start, nitems):
        assert self.buff is not None
        if self.nblocks == 1:
            return
        blocks_to_decompress = nitems / self.n_per_block
        blocks_to_decompress = (
            blocks_to_decompress
            if blocks_to_decompress == int(blocks_to_decompress)
            else int(blocks_to_decompress + 1)
        )
        start_block = int(start / self.n_per_block)
        wanted_decompressed = 0
        while wanted_decompressed < nitems:
            if start_block not in self.read_blocks:
                start_byte = self.start_points[start_block]
                if start_byte == self.start_points_max:
                    stop_byte = self.cbytes
                else:
                    stop_byte = self.start_points[self.start_points > start_byte].min()
                length = stop_byte - start_byte
                data_buff = self.fs.read_block(self.key_path, start_byte, length)
                self.buff[start_byte:stop_byte] = data_buff
                self.read_blocks.add(start_block)
            if wanted_decompressed == 0:
                wanted_decompressed += ((start_block + 1) * self.n_per_block) - start
            else:
                wanted_decompressed += self.n_per_block
            start_block += 1

    def read_full(self):
        return self.chunk_store[self.store_key]


def retry_call(callabl: Callable,
               args=None,
               kwargs=None,
               exceptions: Tuple[Any, ...] = (),
               retries: int = 10,
               wait: float = 0.1) -> Any:
    """
    Make several attempts to invoke the callable. If one of the given exceptions
    is raised, wait the given period of time and retry up to the given number of
    retries.
    """

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    for attempt in range(1, retries+1):
        try:
            return callabl(*args, **kwargs)
        except exceptions:
            if attempt < retries:
                time.sleep(wait)
            else:
                raise
