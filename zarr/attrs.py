# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
from collections import MutableMapping


from zarr.compat import text_type
from zarr.errors import PermissionError


class Attributes(MutableMapping):
    """Class providing access to user attributes on an array or group. Should not be
    instantiated directly, will be available via the `.attrs` property of an array or
    group.

    Parameters
    ----------
    store : MutableMapping
        The store in which to store the attributes.
    key : str, optional
        The key under which the attributes will be stored.
    read_only : bool, optional
        If True, attributes cannot be modified.
    cache : bool, optional
        If True (default), attributes will be cached locally.
    synchronizer : Synchronizer
        Only necessary if attributes may be modified from multiple threads or processes.

    """

    def __init__(self, store, key='.zattrs', read_only=False, cache=True,
                 synchronizer=None):
        self.store = store
        self.key = key
        self.read_only = read_only
        self.cache = cache
        self._cached_asdict = None
        self.synchronizer = synchronizer

    def _get_nosync(self):
        try:
            data = self.store[self.key]
        except KeyError:
            d = dict()
        else:
            d = json.loads(text_type(data, 'ascii'))
        return d

    def asdict(self):
        """Retrieve all attributes as a dictionary."""
        if self.cache and self._cached_asdict is not None:
            return self._cached_asdict
        d = self._get_nosync()
        if self.cache:
            self._cached_asdict = d
        return d

    def refresh(self):
        """Refresh cached attributes from the store."""
        if self.cache:
            self._cached_asdict = self._get_nosync()

    def __contains__(self, x):
        return x in self.asdict()

    def __getitem__(self, item):
        return self.asdict()[item]

    def _write_op(self, f, *args, **kwargs):

        # guard condition
        if self.read_only:
            raise PermissionError('attributes are read-only')

        # synchronization
        if self.synchronizer is None:
            return f(*args, **kwargs)
        else:
            with self.synchronizer[self.key]:
                return f(*args, **kwargs)

    def __setitem__(self, item, value):
        self._write_op(self._setitem_nosync, item, value)

    def _setitem_nosync(self, item, value):

        # load existing data
        d = self._get_nosync()

        # set key value
        d[item] = value

        # _put modified data
        self._put_nosync(d)

    def __delitem__(self, item):
        self._write_op(self._delitem_nosync, item)

    def _delitem_nosync(self, key):

        # load existing data
        d = self._get_nosync()

        # delete key value
        del d[key]

        # _put modified data
        self._put_nosync(d)

    def put(self, d):
        """Overwrite all attributes with the key/value pairs in the provided dictionary
        `d` in a single operation."""
        self._write_op(self._put_nosync, d)

    def _put_nosync(self, d):
        s = json.dumps(d, indent=4, sort_keys=True, ensure_ascii=True,
                       separators=(',', ': '))
        self.store[self.key] = s.encode('ascii')
        if self.cache:
            self._cached_asdict = d

    # noinspection PyMethodOverriding
    def update(self, *args, **kwargs):
        """Update the values of several attributes in a single operation."""
        self._write_op(self._update_nosync, *args, **kwargs)

    def _update_nosync(self, *args, **kwargs):

        # load existing data
        d = self._get_nosync()

        # update
        d.update(*args, **kwargs)

        # _put modified data
        self._put_nosync(d)

    def keys(self):
        return self.asdict().keys()

    def __iter__(self):
        return iter(self.asdict())

    def __len__(self):
        return len(self.asdict())

    def _ipython_key_completions_(self):
        return sorted(self)
