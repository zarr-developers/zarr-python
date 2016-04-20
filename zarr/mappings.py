# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import Mapping, MutableMapping
import functools
import operator
import os
import tempfile
import json


# noinspection PyPep8Naming
class frozendict(Mapping):

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __contains__(self, x):
        return x in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def copy(self, **add_or_replace):
        return frozendict(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<frozendict %s>' % repr(self._dict)

    def __hash__(self):
        if self._hash is None:
            hashes = map(hash, self.items())
            self._hash = functools.reduce(operator.xor, hashes, 0)
        return self._hash


class Directory(MutableMapping):
    """Mutable Mapping interface to a directory. Keys must be strings,
    values must be bytes.

    Parameters
    ----------
    path : string
    suffix : string

    """

    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def __getitem__(self, key):
        try:
            with open(os.path.join(self._path, key), 'rb') as f:
                return f.read()
        except (IOError, OSError):
            raise KeyError(key)

    def __setitem__(self, key, value):
        if not isinstance(value, bytes):
            raise ValueError('value must be of type bytes')

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=self._path,
                                         prefix=key + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        os.rename(temp_path, os.path.join(self._path, key))

    def __delitem__(self, key):
        os.remove(os.path.join(self._path, key))

    def __contains__(self, key):
        return os.path.exists(os.path.join(self._path, key))

    def keys(self):
        return iter(os.listdir(self._path))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def size(self):
        return sum(os.path.getsize(os.path.join(self._path, key))
                   for key in self.keys())


class JSONFile(MutableMapping):
    """Mutable Mapping interface to a JSON file. Keys must be strings,
    values must be JSON serializable.

    Parameters
    ----------
    path : string

    """

    def __init__(self, path, read_only=False):
        self._path = path
        self._read_only = read_only

    def __contains__(self, x):
        return x in self.asdict()

    def __getitem__(self, item):

        if not os.path.exists(self._path):
            raise KeyError(item)

        with open(self._path, mode='r') as f:
            return json.load(f)[item]

    def __setitem__(self, key, value):

        # handle read-only state
        if self._read_only:
            raise ValueError('array is read-only')

        # load existing data
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)

        # set key value
        d[key] = value

        # write modified data
        with open(self._path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def __delitem__(self, key):

        # handle read-only state
        if self._read_only:
            raise ValueError('array is read-only')

        # load existing data
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)

        # delete key value
        del d[key]

        # write modified data
        with open(self._path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def asdict(self):
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)
        return d

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
