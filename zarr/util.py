# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from textwrap import TextWrapper, dedent
import numbers
import uuid
import inspect


from asciitree import BoxStyle, LeftAligned
from asciitree.traversal import Traversal
import numpy as np
from numcodecs.compat import ensure_ndarray
from numcodecs.registry import codec_registry


from zarr.compat import PY2, text_type, binary_type


# codecs to use for object dtype convenience API
object_codecs = {
    text_type.__name__: 'vlen-utf8',
    binary_type.__name__: 'vlen-bytes',
    'array': 'vlen-array',
}


def normalize_shape(shape):
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


def guess_chunks(shape, typesize):
    """
    Guess an appropriate chunk layout for a dataset, given its shape and
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

        chunks[idx % ndims] = np.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)


def normalize_chunks(chunks, shape, typesize):
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
        chunks = (int(chunks),)

    # handle bad dimensionality
    if len(chunks) > len(shape):
        raise ValueError('too many dimensions in chunks')

    # handle underspecified chunks
    if len(chunks) < len(shape):
        # assume chunks across remaining dimensions
        chunks += shape[len(chunks):]

    # handle None in chunks
    chunks = tuple(s if c is None else int(c)
                   for s, c in zip(shape, chunks))

    return chunks


def normalize_dtype(dtype, object_codec):

    # convenience API for object arrays
    if inspect.isclass(dtype):
        dtype = dtype.__name__
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
                    args = ()
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
def is_total_slice(item, shape):
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


def human_readable_size(size):
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


def normalize_order(order):
    order = str(order).upper()
    if order not in ['C', 'F']:
        raise ValueError("order must be either 'C' or 'F', found: %r" % order)
    return order


def normalize_fill_value(fill_value, dtype):

    if fill_value is None:
        # no fill value
        pass

    elif fill_value == 0:
        # this should be compatible across numpy versions for any array type, including
        # structured arrays
        fill_value = np.zeros((), dtype=dtype)[()]

    elif dtype.kind == 'U':
        # special case unicode because of encoding issues on Windows if passed through numpy
        # https://github.com/alimanfoo/zarr/pull/172#issuecomment-343782713

        if PY2 and isinstance(fill_value, binary_type):  # pragma: py3 no cover
            # this is OK on PY2, can be written as JSON
            pass

        elif not isinstance(fill_value, text_type):
            raise ValueError('fill_value {!r} is not valid for dtype {}; must be a '
                             'unicode string'.format(fill_value, dtype))

    else:
        try:
            if isinstance(fill_value, binary_type) and dtype.kind == 'V':
                # special case for numpy 1.14 compatibility
                fill_value = np.array(fill_value, dtype=dtype.str).view(dtype)[()]
            else:
                fill_value = np.array(fill_value, dtype=dtype)[()]

        except Exception as e:
            # re-raise with our own error message to be helpful
            raise ValueError('fill_value {!r} is not valid for dtype {}; nested '
                             'exception: {}'.format(fill_value, dtype, e))

    return fill_value


def normalize_storage_path(path):

    # handle bytes
    if not PY2 and isinstance(path, bytes):  # pragma: py2 no cover
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


def buffer_size(v):
    return ensure_ndarray(v).nbytes


def info_text_report(items):
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


def info_html_report(items):
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


def tree_html_sublist(node, root=False, expand=False):
    result = ''
    data_jstree = '{"type": "%s"}' % node.get_type()
    if root or (expand is True) or (isinstance(expand, int) and node.depth < expand):
        css_class = 'jstree-open'
    else:
        css_class = ''
    result += "<li data-jstree='{}' class='{}'>".format(data_jstree, css_class)
    result += '<span>{}</span>'.format(node.get_text())
    children = node.get_children()
    if children:
        result += '<ul>'
        for c in children:
            result += tree_html_sublist(c, expand=expand)
        result += '</ul>'
    result += '</li>'
    return result


def tree_html(group, expand, level):

    result = ''

    # include CSS for jstree default theme
    css_url = '//cdnjs.cloudflare.com/ajax/libs/jstree/3.3.3/themes/default/style.min.css'
    result += '<link rel="stylesheet" href="{}"/>'.format(css_url)

    # construct the tree as HTML nested lists
    node_id = uuid.uuid4()
    result += '<div id="{}" class="zarr-tree">'.format(node_id)
    result += '<ul>'
    root = TreeNode(group, level=level)
    result += tree_html_sublist(root, root=True, expand=expand)
    result += '</ul>'
    result += '</div>'

    # construct javascript
    result += dedent("""
        <script>
            if (!require.defined('jquery')) {
                require.config({
                    paths: {
                        jquery: '//cdnjs.cloudflare.com/ajax/libs/jquery/1.12.1/jquery.min'
                    },
                });
            }
            if (!require.defined('jstree')) {
                require.config({
                    paths: {
                        jstree: '//cdnjs.cloudflare.com/ajax/libs/jstree/3.3.3/jstree.min'
                    },
                });
            }
            require(['jstree'], function() {
                $('#%s').jstree({
                    types: {
                        Group: {
                            icon: "%s"
                        },
                        Array: {
                            icon: "%s"
                        }
                    },
                    plugins: ["types"]
                });
            });
        </script>
    """ % (node_id, tree_group_icon, tree_array_icon))

    return result


tree_group_icon = 'fa fa-folder'
tree_array_icon = 'fa fa-table'
# alternatives...
# tree_group_icon: 'jstree-folder'
# tree_array_icon: 'jstree-file'


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
            UP_AND_RIGHT=u"\u2514",
            HORIZONTAL=u"\u2500",
            VERTICAL=u"\u2502",
            VERTICAL_AND_RIGHT=u"\u251C"
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
        if not PY2:
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
        if PY2:  # pragma: py3 no cover
            return self.__bytes__()
        else:  # pragma: py2 no cover
            return self.__unicode__()

    def _repr_html_(self):
        return tree_html(self.group, expand=self.expand, level=self.level)


def check_array_shape(param, array, shape):
    if not hasattr(array, 'shape'):
        raise TypeError('parameter {!r}: expected an array-like object, got {!r}'
                        .format(param, type(array)))
    if array.shape != shape:
        raise ValueError('parameter {!r}: expected array with shape {!r}, got {!r}'
                         .format(param, shape, array.shape))


def is_valid_python_name(name):
    if PY2:  # pragma: py3 no cover
        import ast
        # noinspection PyBroadException
        try:
            ast.parse('"".{};'.format(name))
        except Exception:
            return False
        else:
            return True
    else:  # pragma: py2 no cover
        from keyword import iskeyword
        return name.isidentifier() and not iskeyword(name)


def instance_dir(obj):  # pragma: py3 no cover
    """Vanilla implementation of built-in dir() for PY2 to help with overriding __dir__.
    Based on implementation of dir() in pypy."""
    d = dict()
    d.update(obj.__dict__)
    d.update(class_dir(obj.__class__))
    result = sorted(d.keys())
    return result


def class_dir(klass):  # pragma: py3 no cover
    d = dict()
    d.update(klass.__dict__)
    bases = klass.__bases__
    for base in bases:
        d.update(class_dir(base))
    return d


class NoLock(object):
    """A lock that doesn't lock."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


nolock = NoLock()
