"""
Zarr spec v3 draft implementation
"""

__version__ = "0.0.1"

import json
import os
import sys
from collections.abc import MutableMapping
import pathlib
from string import ascii_letters, digits
from numcodecs.compat import ensure_bytes

from .utils import syncify, nested_run

# flake8: noqa
from .comparer import StoreComparer

RENAMED_MAP = {
    "dtype": "data_type",
    "order": "chunk_memory_layout",
}


from typing import NewType

Key = NewType("Key", str)
Path = NewType("Path", str)


def _assert_valid_path(path: str):
    if sys.version_info > (3, 7):
        assert path.isascii()
    assert path.startswith("/")
    assert "\\" not in path


class BaseV3Store:
    """
    Base utility class to create a v3-complient store with extra checks and utilities.

    It provides a number of default method implementation adding extra checks in
    order to ensure the correctness fo the implmentation.
    """

    _store_version = 3
    _async = True

    @staticmethod
    def _valid_key(key: str) -> bool:
        """
        Verify that a key is confirm to the specification.

        A key us any string containing only character in the range a-z, A-Z,
        0-9, or in the set /.-_ it will return True if that's the case, false
        otherwise.

        In addition, in spec v3, keys can only start with the prefix meta/,
        data/ or be exactly zarr.json. This should not be exposed to the
        user, and is a store implmentation detail, so thie method will raise
        a ValueError in that case.
        """
        if sys.version_info > (3, 7):
            if not key.isascii():
                return False
        if set(key) - set(ascii_letters + digits + "/.-_"):
            return False

        if (
            not key.startswith("data/")
            and (not key.startswith("meta/"))
            and (not key == "zarr.json")
        ):
            raise ValueError("keys starts with unexpected value: `{}`".format(key))
        # todo likely more logics to add there.
        return True

    async def async_get(self, key: str):
        """
        default implementation of async_get/get that validate the key, a
        check that the return value by bytes. rely on `async def _get(key)`
        to be implmented.

        Will ensure that the following are correct:
            - return group metadata objects are json and contain a signel
            `attributes` keys.
        """
        assert self._valid_key(key), key
        result = await self._get(key)
        assert isinstance(result, bytes), "Expected bytes, got {}".format(result)
        if key == "zarr.json":
            v = json.loads(result.decode())
            assert set(v.keys()) == {
                "zarr_format",
                "metadata_encoding",
                "extensions",
            }, "v is {}".format(v)
        elif key.endswith("/.group"):
            v = json.loads(result.decode())
            assert set(v.keys()) == {"attributes"}, "got unexpected keys {}".format(
                v.keys()
            )
        return result

    async def async_set(self, key: str, value: bytes):
        """
        default implementation of async_set/set that validate the key, and
        check that the return value by bytes. rely on `async def _set(key, value)`
        to be implmented.

        Will ensure that the following are correct:
            - set group metadata objects are json and contain a signel `attributes` keys.
        """
        if key == "zarr.json":
            v = json.loads(value.decode())
            assert set(v.keys()) == {
                "zarr_format",
                "metadata_encoding",
                "extensions",
            }, "v is {}".format(v)
        elif key.endswith(".array"):
            v = json.loads(value.decode())
            expected = {
                "shape",
                "data_type",
                "chunk_grid",
                "chunk_memory_layout",
                "compressor",
                "fill_value",
                "extensions",
                "attributes",
            }
            current = set(v.keys())
            # ets do some conversions.
            assert current == expected, "{} extra, {} missing in {}".format(
                current - expected, expected - curent, v
            )

        assert isinstance(value, bytes)
        assert self._valid_key(key)
        await self._set(key, value)

    async def async_list_prefix(self, prefix):
        return [k for k in await self.async_list() if k.startswith(prefix)]

    async def async_delete(self, key):
        deln = await self._backend().delete(key)
        if deln == 0:
            raise KeyError(key)

    async def async_initialize(self):
        pass

    async def async_list_dir(self, prefix):
        """
        Note: carefully test this with trailing/leading slashes
        """
        assert prefix.endswith("/")

        def part1(key):
            if "/" not in key:
                return key
            else:
                return key.split("/", maxsplit=1)[0] + "/"

        all_keys = await self.async_list_prefix(prefix)
        len_prefix = len(prefix)
        trail = {part1(k[len_prefix:]) for k in all_keys}
        return [prefix + k for k in trail]

    async def async_contains(self, key):
        assert key.startswith(("meta/", "data/")), "Got {}".format(key)
        return key in await self.async_list()

    def __contains__(self, key):
        if hasattr(self, "contains"):
            return self.contains(key)
        else:
            with nested_run():
                import trio

                return trio.run(self.async_contains, key)


class AsyncV3DirectoryStore(BaseV3Store):
    log = []

    def __init__(self, key):
        self.log.append("init")
        self.root = pathlib.Path(key)

    async def _get(self, key: Key):
        self.log.append("get" + key)
        path = self.root / key
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise KeyError(path)

    async def _set(self, key, value):
        self.log.append("set {} {}".format(key, value))
        assert not key.endswith("root/.group")
        assert value
        path = self.root / key
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        return path.write_bytes(value)

    async def async_list(self):
        ll = []
        for it in os.walk(self.root):
            if os.path.sep != "/":
                prefix = "/".join(it[0].split(os.path.sep))
            else:
                prefix = it[0]
            for file in it[2]:
                str_key = "/".join([prefix, file])[len(str(self.root)) + 1 :]
                assert "\\" not in str_key, str_key
                ll.append(str_key)
        return ll

    async def async_delete(self, key):
        self.log.append("delete {}".format(key))
        path = self.root / key
        os.remove(path)


@syncify
class SyncV3DirectoryStore(AsyncV3DirectoryStore):
    _async = False

    def __getitem__(self, key):
        assert not key.endswith("root/.group")
        return self.get(key)


class AsyncV3RedisStore(BaseV3Store):
    def __init__(self, host=None, port=None):
        """initialisation is in _async initialize
        for early failure.
        """
        self.host = host
        self.port = port

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()
        from redio import Redis

        self._backend = Redis("redis://localhost/")

    async def async_initialize(self):
        from redio import Redis

        self._backend = Redis("redis://localhost/")
        b = self._backend()
        for k in await self._backend().keys():
            b.delete(k)
        await b
        return self

    async def _get(self, key):
        res = await self._backend().get(key)
        if res is None:
            raise KeyError
        return res

    async def _set(self, key, value):
        return await self._backend().set(key, value)

    async def async_list(self):
        return await self._backend().keys()


@syncify
class SyncV3RedisStore(AsyncV3RedisStore):
    _async = False

    def __setitem__(self, key, value):
        assert ".zgroup" not in key
        return self.set(key, value)


class AsyncV3MemoryStore(BaseV3Store):
    def __init__(self):
        self._backend = dict()

    async def _get(self, key):
        return self._backend[key]

    async def _set(self, key, value):
        self._backend[key] = value

    async def async_delete(self, key):
        del self._backend[key]

    async def async_list(self):
        return list(self._backend.keys())


@syncify
class SyncV3MemoryStore(AsyncV3MemoryStore):
    _async = False


class AsyncZarrProtocolV3:
    def __init__(self, store):
        if isinstance(store, type):
            self._store = store()
        else:
            self._store = store
        if hasattr(self, "init_hierarchy"):
            self.init_hierarchy()

    async def async_init_hierarchy(self):
        basic_info = {
            "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0",
            "metadata_encoding": "https://tools.ietf.org/html/rfc8259",
            "extensions": [],
        }
        try:
            await self._store.async_get("zarr.json")
        except KeyError:
            await self._store.async_set("zarr.json", json.dumps(basic_info).encode())

    def _g_meta_key(self, path):
        _assert_valid_path(path)
        return "meta" + path + ".group"

    async def async_create_group(self, group_path: str):
        """
        create a goup at `group_path`,
        we need to make sure none of the subpath of group_path are arrays.

        say  path is g1/g2/g3, we want to check

        /meta/g1.array
        /meta/g1/g2.array

        we could also assume that protocol implementation never do that.
        """
        _assert_valid_path(group_path)
        DEFAULT_GROUP = """{
        "attributes": {
            "spam": "ham",
            "eggs": 42,
        }  }
        """
        await self._store.async_set(
            self._g_meta_key(group_path), DEFAULT_GROUP.encode()
        )

    def _create_array_metadata(self, shape=(10,), dtype="<f64", chunk_shape=(1,)):
        return {
            "shape": shape,
            "data_type": dtype,
            "chunk_grid": {
                "type": "regular",
                "chunk_shape": chunk_shape,
                "separator": "/",
            },
            "chunk_memory_layout": "C",
            "compressor": {
                "codec": "https://none",
                "configuration": {},
                "fill_value": "NaN",
            },
            "extensions": [],
            "attributes": {},
        }

    async def create_array(self, array_path: str):
        """
        create a goup at `array_path`,
        we need to make sure none of the subpath of array_path are arrays.

        say  path is g1/g2/d3, we want to check

        /meta/g1.array
        /meta/g1/g2.array

        we could also assume that protocol implementation never do that.
        """
        _assert_valid_path(array_path)
        DEFAULT_ARRAY = """{
                "extensions": [],
        "attributes": {
            "spam": "ham",
            "eggs": 42,
        }  }
        """
        await self._store.set(self._g_meta_key(array_path), DEFAULT_ARRAY.encode())


@syncify
class ZarrProtocolV3(AsyncZarrProtocolV3):
    pass


class V2from3Adapter(MutableMapping):
    """
    class to wrap a 3 store and return a V2 interface
    """

    _store_version = 2

    def __init__(self, v3store):
        """

        Wrapper arround a v3store to give a v2 compatible interface.

        Still have some rough edges, and try to do the sensible things for
        most case mostly key converstions so far.

        Note that the V3 spec is still in flux, so this is simply a prototype
        to see the pain points in using the spec v3 and must not be used for
        production.


        This will try to do the followign conversions:
         - name of given keys `.zgroup` -> `.group` for example.
         - path of storage (prefix with root/ meta// when relevant and vice versa.)
         - try to ensure the stored objects are bytes before reachign the underlying store.
         - try to adapt v2/v2 nested/flat structures

        THere will ikley need to be _some_

        """
        self._v3store = v3store

    def __getitem__(self, key):
        """
        In v2  both metadata and data are mixed so we'll need to convert things
        that ends with .z to the metadata path.
        """
        assert isinstance(key, str), "expecting string got {key}".format(key=repr(key))
        v3key = self._convert_2_to_3_keys(key)
        if key.endswith(".zattrs"):
            try:
                res = self._v3store.get(v3key)
            except KeyError:
                v3key = v3key.replace(".array", ".group")
        res = self._v3store.get(v3key)

        assert isinstance(res, bytes)
        if key.endswith(".zattrs"):
            data = json.loads(res.decode())["attributes"]
            res = json.dumps(data, indent=4).encode()
        elif key.endswith(".zarray"):
            data = json.loads(res.decode())
            for target, source in RENAMED_MAP.items():
                tmp = data[source]
                del data[source]
                data[target] = tmp
            data["chunks"] = data["chunk_grid"]["chunk_shape"]
            del data["chunk_grid"]

            data["zarr_format"] = 2
            data["filters"] = None
            del data["extensions"]
            del data["attributes"]
            res = json.dumps(data, indent=4).encode()

        if v3key.endswith(".group") or v3key == "zarr.json":
            data = json.loads(res.decode())
            data["zarr_format"] = 2
            if data.get("attributes") is not None:
                del data["attributes"]
            res = json.dumps(data, indent=4).encode()
        assert isinstance(res, bytes)
        return res

    def __setitem__(self, key, value):
        """
        In v2  both metadata and data are mixed so we'll need to convert things
        that ends with .z to the metadata path.
        """
        # TODO convert to bytes if needed

        v3key = self._convert_2_to_3_keys(key)
        assert not key.endswith("root/.group")
        # convert chunk separator from ``.`` to ``/``

        if key.endswith(".zarray"):
            data = json.loads(value.decode())
            for source, target in RENAMED_MAP.items():
                tmp = data[source]
                del data[source]
                data[target] = tmp
            data["chunk_grid"] = {}
            data["chunk_grid"]["chunk_shape"] = data["chunks"]
            del data["chunks"]
            data["chunk_grid"]["type"] = "rectangular"
            data["chunk_grid"]["separator"] = "/"
            assert data["zarr_format"] == 2
            del data["zarr_format"]
            assert data["filters"] in ([], None), "found filters: {}".format(
                data["filters"]
            )
            del data["filters"]
            data["extensions"] = []
            try:
                attrs = json.loads(self._v3store.get(v3key).decode())["attributes"]
            except KeyError:
                attrs = []
            data["attributes"] = attrs
            data = json.dumps(data, indent=4).encode()
        elif key.endswith(".zattrs"):
            try:
                # try zarray first...
                data = json.loads(self._v3store.get(v3key).decode())
            except KeyError:
                try:
                    v3key = v3key.replace(".array", ".group")
                    data = json.loads(self._v3store.get(v3key).decode())
                except KeyError:
                    data = {}
            data["attributes"] = json.loads(value.decode())
            self._v3store.set(v3key, json.dumps(data, indent=4).encode())
            return
        # todo: we want to keep the .zattr which i sstored in the  group/array file.
        # so to set, we need to get from the store assign update.
        elif v3key == "meta/root.group":
            # todo: this is wrong, the top md document is zarr.json.
            data = json.loads(value.decode())
            data["zarr_format"] = "https://purl.org/zarr/spec/protocol/core/3.0"
            data = json.dumps(data, indent=4).encode()
        elif v3key.endswith("/.group"):
            data = json.loads(value.decode())
            del data["zarr_format"]
            if "attributes" not in data:
                data["attributes"] = {}
            data = json.dumps(data).encode()
        else:
            data = value
        assert not isinstance(data, dict)
        self._v3store.set(v3key, ensure_bytes(data))

    def __contains__(self, key):
        return self._convert_2_to_3_keys(key) in self._v3store.list()

    def _convert_3_to_2_keys(self, v3key: str) -> str:
        """
        todo:
         - handle special .attribute which is merged with .zarray/.zgroup
         - look at the grid separator
        """
        if v3key == "meta/root.group":
            return ".zgroup"
        if v3key == "meta/root.array":
            return ".zarray"
        suffix = v3key[10:]
        if suffix.endswith(".array"):
            return suffix[:-6] + ".zarray"
        if suffix.endswith(".group"):
            return suffix[:-6] + ".zgroup"
        return suffix

    def _convert_2_to_3_keys(self, v2key: str) -> str:
        """
        todo:
         - handle special .attribute which is merged with .zarray/.zgroup
         - look at the grid separator

        """
        # head of the hierachy is different:
        if v2key in (".zgroup", ".zattrs"):
            return "meta/root.group"
        if v2key == ".zarray":
            return "meta/root.array"
        assert not v2key.startswith(
            "/"
        ), "expect keys to not start with slash but does {}".format(repr(v2key))
        if v2key.endswith(".zarray") or v2key.endswith(".zattrs"):
            return "meta/root/" + v2key[:-7] + ".array"
        if v2key.endswith(".zgroup"):
            return "meta/root/" + v2key[:-7] + ".group"
        return "data/root/" + v2key

    def __len__(self):
        return len(self._v3store.list())

    def clear(self):
        keys = self._v3store.list()
        for k in keys:
            self._v3store.delete(k)

    def __delitem__(self, key):
        item3 = self._convert_2_to_3_keys(key)

        items = self._v3store.list_prefix(item3)
        if not items:
            raise KeyError(
                "{} not found in store (converted key to {}".format(key, item3)
            )
        for _item in self._v3store.list_prefix(item3):
            self._v3store.delete(_item)

    def keys(self):
        # TODO: not as stritforward. we need to actually poke internally at
        # .group/.array to potentially return '.zattrs' if attribute is set. it
        # also seem in soem case zattrs is set in arrays even if the rest of the
        # infomation is not set.
        key = self._v3store.list()
        fixed_paths = []
        for p in key:
            if p.endswith((".group", ".array")):
                res = self._v3store.get(p)
                if json.loads(res.decode()).get("attributes"):
                    fixed_paths.append(p[10:-6] + ".zattrs")
            fixed_paths.append(self._convert_3_to_2_keys(p))

        return list(set(fixed_paths))

    def listdir(self, path=""):
        """
        This_will be wrong as we also need to list meta/prefix, but need to
        be carefull and use list-prefix in that case with the right optiosn
        to convert the chunks separators.
        """
        v3path = self._convert_2_to_3_keys(path)
        if not v3path.endswith("/"):
            v3path = v3path + "/"
        # if not v3path.startswith("/"):
        #    v3path = '/'+v3path
        ps = [p for p in self._v3store.list_dir(v3path)]
        fixed_paths = []
        for p in ps:
            if p == ".group":
                res = self._v3store.get(path + "/.group")
                if json.loads(res.decode())["attributes"]:
                    fixed_paths.append(".zattrs")
            fixed_paths.append(self._convert_3_to_2_keys(p))

        res = [p.split("/")[-2] for p in fixed_paths]
        return res

    def __iter__(self):
        return iter(self.keys())
