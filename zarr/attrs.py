# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
from collections import MutableMapping


from zarr.compat import text_type
from zarr.errors import ReadOnlyError


class Attributes(MutableMapping):

    def __init__(self, store, key='attrs', readonly=False):
        if key not in store:
            store[key] = json.dumps(dict()).encode('ascii')
        self.store = store
        self.key = key
        self.readonly = readonly

    def __contains__(self, x):
        return x in self.asdict()

    def __getitem__(self, item):
        return self.asdict()[item]

    def put(self, d):

        # guard conditions
        if self.readonly:
            raise ReadOnlyError('attributes are read-only')

        s = json.dumps(d, indent=4, sort_keys=True, ensure_ascii=True)
        self.store[self.key] = s.encode('ascii')

    def __setitem__(self, key, value):

        # guard conditions
        if self.readonly:
            raise ReadOnlyError('attributes are read-only')

        # load existing data
        d = self.asdict()

        # set key value
        d[key] = value

        # put modified data
        self.put(d)

    def __delitem__(self, key):

        # guard conditions
        if self.readonly:
            raise ReadOnlyError('mapping is read-only')

        # load existing data
        d = self.asdict()

        # delete key value
        del d[key]

        # put modified data
        self.put(d)

    def asdict(self):
        return json.loads(text_type(self.store[self.key], 'ascii'))

    def update(self, *args, **kwargs):
        # override to provide update in a single write

        # guard conditions
        if self.readonly:
            raise ReadOnlyError('mapping is read-only')

        # load existing data
        d = self.asdict()

        # update
        d.update(*args, **kwargs)

        # put modified data
        self.put(d)

    def __iter__(self):
        return iter(self.asdict())

    def __len__(self):
        return len(self.asdict())

    def keys(self):
        return self.asdict().keys()

    def values(self):
        return self.asdict().values()

    def items(self):
        return self.asdict().items()
