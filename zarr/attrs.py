from collections.abc import MutableMapping

from zarr._storage.store import Store, StoreV3
from zarr.util import json_dumps


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

        self._version = getattr(store, '_store_version', 2)
        assert key

        if self._version == 3 and '.z' in key:
            raise ValueError('invalid v3 key')

        _Store = Store if self._version == 2 else StoreV3
        self.store = _Store._ensure_store(store)
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
            if self._version > 2:
                d['attributes'] = {}
        else:
            d = self.store._metadata_class.parse_metadata(data)
        return d

    def asdict(self):
        """Retrieve all attributes as a dictionary."""
        if self.cache and self._cached_asdict is not None:
            return self._cached_asdict
        d = self._get_nosync()
        if self._version == 3:
            d = d['attributes']
        if self.cache:
            self._cached_asdict = d
        return d

    def refresh(self):
        """Refresh cached attributes from the store."""
        if self.cache:
            if self._version == 3:
                self._cached_asdict = self._get_nosync()['attributes']
            else:
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
        if self._version == 2:
            d[item] = value
        else:
            d['attributes'][item] = value

        # _put modified data
        self._put_nosync(d)

    def __delitem__(self, item):
        self._write_op(self._delitem_nosync, item)

    def _delitem_nosync(self, key):

        # load existing data
        d = self._get_nosync()

        # delete key value
        if self._version == 2:
            del d[key]
        else:
            del d['attributes'][key]

        # _put modified data
        self._put_nosync(d)

    def put(self, d):
        """Overwrite all attributes with the key/value pairs in the provided dictionary
        `d` in a single operation."""
        if self._version == 2:
            self._write_op(self._put_nosync, d)
        else:
            self._write_op(self._put_nosync, dict(attributes=d))

    def _put_nosync(self, d):
        if self._version == 2:
            self.store[self.key] = json_dumps(d)
            if self.cache:
                self._cached_asdict = d
        else:
            if self.key in self.store:
                # Cannot write the attributes directly to JSON, but have to
                # store it within the pre-existing attributes key of the v3
                # metadata.

                # Note: this changes the store.counter result in test_caching_on!

                meta = self.store._metadata_class.parse_metadata(self.store[self.key])
                if 'attributes' in meta and 'filters' in meta['attributes']:
                    # need to preserve any existing "filters" attribute
                    d['attributes']['filters'] = meta['attributes']['filters']
                meta['attributes'] = d['attributes']
            else:
                meta = d
            self.store[self.key] = json_dumps(meta)
            if self.cache:
                self._cached_asdict = d['attributes']

    # noinspection PyMethodOverriding
    def update(self, *args, **kwargs):
        """Update the values of several attributes in a single operation."""
        self._write_op(self._update_nosync, *args, **kwargs)

    def _update_nosync(self, *args, **kwargs):

        # load existing data
        d = self._get_nosync()

        # update
        if self._version == 2:
            d.update(*args, **kwargs)
        else:
            if 'attributes' not in d:
                d['attributes'] = {}
            d['attributes'].update(*args, **kwargs)

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
