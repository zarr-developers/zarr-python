# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
from collections import MutableMapping


from zarr.compat import text_type
from zarr.errors import PermissionError


class Attributes(MutableMapping):

    def __init__(self, store, key='.zattrs', read_only=False,
                 synchronizer=None):
        self.store = store
        self.key = key
        self.read_only = read_only
        self.synchronizer = synchronizer

    def __contains__(self, x):
        return x in self.asdict()

    def __getitem__(self, item):
        return self.asdict()[item]

    def _put(self, d):
        s = json.dumps(d, indent=4, sort_keys=True, ensure_ascii=True)
        self.store[self.key] = s.encode('ascii')

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
        d = self.asdict()

        # set key value
        d[item] = value

        # _put modified data
        self._put(d)

    def __delitem__(self, item):
        self._write_op(self._delitem_nosync, item)

    def _delitem_nosync(self, key):

        # load existing data
        d = self.asdict()

        # delete key value
        del d[key]

        # _put modified data
        self._put(d)

    def asdict(self):
        if self.key in self.store:
            return json.loads(text_type(self.store[self.key], 'ascii'))
        else:
            return dict()

    def update(self, *args, **kwargs):
        # override to provide update in a single write
        self._write_op(self._update_nosync, *args, **kwargs)

    def _update_nosync(self, *args, **kwargs):

        # load existing data
        d = self.asdict()

        # update
        d.update(*args, **kwargs)

        # _put modified data
        self._put(d)

    def __iter__(self):
        return iter(self.asdict())

    def __len__(self):
        return len(self.asdict())
