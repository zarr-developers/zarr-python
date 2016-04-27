# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json


class DirectoryMap(MutableMapping):
    """Mutable Mapping interface to a directory. Keys must be strings,
    values must be bytes.

    Parameters
    ----------
    path : string

    """

    def __init__(self, path):

        # guard conditions
        if not os.path.exists(path):
            raise ValueError('path does not exist')
        elif not os.path.isdir(path):
            raise ValueError('path is not a directory')

        self.path = path

    def __getitem__(self, key):

        # guard conditions
        if key not in self:
            raise KeyError(key)

        with open(os.path.join(self.path, key), 'rb') as f:
            return f.read()

    def __setitem__(self, key, value):

        # guard conditions
        if not isinstance(value, bytes):
            raise ValueError('value must be of type bytes')

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=self.path,
                                         prefix=key + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        os.rename(temp_path, os.path.join(self.path, key))

    def __delitem__(self, key):

        # guard conditions
        if key not in self:
            raise KeyError(key)

        os.remove(os.path.join(self.path, key))

    def __contains__(self, key):
        return os.path.isfile(os.path.join(self.path, key))

    def keys(self):
        for key in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, key)):
                yield key

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    @property
    def size(self):
        """Total size of all values in number of bytes."""
        return sum(os.path.getsize(os.path.join(self.path, key))
                   for key in self.keys())


class JSONFileMap(MutableMapping):
    """Mutable Mapping interface to a JSON file. Keys must be strings,
    values must be JSON serializable.

    Parameters
    ----------
    path : string

    """

    def __init__(self, path, readonly=False):

        # guard conditions
        if not os.path.exists(path):
            if readonly:
                raise ValueError('path does not exist: %s' % path)
            else:
                with open(path, mode='w') as f:
                    json.dump(dict(), f, indent=4, sort_keys=True)

        # setup instance
        self.path = path
        self.readonly = readonly

    def __contains__(self, x):
        return x in self.asdict()

    def __getitem__(self, item):
        return self.asdict()[item]

    def __setitem__(self, key, value):

        # guard conditions
        if self.readonly:
            raise PermissionError('mapping is read-only')

        # load existing data
        d = self.asdict()

        # set key value
        d[key] = value

        # write modified data
        with open(self.path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def __delitem__(self, key):

        # guard conditions
        if self.readonly:
            raise PermissionError('mapping is read-only')

        # load existing data
        d = self.asdict()

        # delete key value
        del d[key]

        # write modified data
        with open(self.path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def asdict(self):
        with open(self.path, mode='r') as f:
            d = json.load(f)
        return d

    def update(self, *args, **kwargs):
        # override to provide update in a single write

        # guard conditions
        if self.readonly:
            raise PermissionError('mapping is read-only')

        # load existing data
        d = self.asdict()

        # update
        d.update(*args, **kwargs)

        # write modified data
        with open(self.path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

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
