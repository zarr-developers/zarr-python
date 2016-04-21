
import numpy as np
from zarr.store.base import ArrayStore
from zarr.util import normalize_cparams, normalize_shape, normalize_chunks, \
    normalize_resize_args

class DictStore(object):
    """ ArrayStore implemented from three MutableMappings

    Parameters
    ----------
    data: MutableMapping
        Holds bulk data
    attrs: MutableMapping
        Holds user defined attributes
    meta: MutableMapping
        Holds array metadata, like shape, chunks, dtype, etc..
    **extra_meta: Keyword arguments
        shape, chunks, dtype, etc..  See MemoryStore for options

    Examples
    --------
    >>> data = {}
    >>> attrs = {}
    >>> meta = {}
    >>> store = DictStore(data, attrs, meta, shape=(1000,), chunks=(100,))
    >>> meta['shape']
    (1000,)

    >>> import zarr
    >>> x = zarr.Array(store)
    >>> x[:] = 1
    >>> sorted(data.keys())
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    def __init__(self, data, attrs, meta, count_cbytes=False, **extra_meta):
        meta.update(extra_meta)

        # normalize arguments
        meta['cname'], meta['clevel'], meta['shuffle'] = \
                normalize_cparams(meta.get('cname', None),
                                  meta.get('clevel', None),
                                  meta.get('shuffle', None))
        meta['shape'] = normalize_shape(meta['shape'])
        meta['chunks'] = normalize_chunks(meta.get('chunks', None),
                                          meta['shape'])
        meta['dtype'] = np.dtype(meta.get('dtype', None))  # float64 default
        meta['fill_value'] = meta.get('fill_value', None)

        self.data = data
        self.attrs = attrs
        self.meta = meta
        self.count_cbytes = count_cbytes

    @property
    def cbytes(self):
        if self.count_cbytes:
            return sum(len(v) for v in itervalues(self.data))
        else:
            return -1

    @property
    def initialized(self):
        return len(self.data)

    def resize(self, *args):
        # normalize new shape argument
        old_shape = self.meta['shape']
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self.meta['chunks']
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for ckey in list(self.data):
            cidx = map(int, ckey.split('.'))
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                pass  # keep the chunk
            else:
                del self.data[ckey]

        # update metadata
        self.meta['shape'] = new_shape
