from functools import reduce
from itertools import product
from typing import Any, Iterable, Iterator, Optional, Tuple

from zarr._storage.store import BaseStore, Store
from zarr.storage import StoreLike, array_meta_key, attrs_key, group_meta_key


def _cum_prod(x: Iterable[int]) -> Iterable[int]:
    prod = 1
    yield prod
    for i in x[:-1]:
        prod *= i
        yield prod


class ShardedStore(Store):
    """This class should not be used directly,
    but is added to an Array as a wrapper when needed automatically."""

    def __init__(
        self, store:
        StoreLike,
        shards: Tuple[int, ...],
        dimension_separator: str,
        chunk_has_constant_size: bool,
        fill_value: bytes,
        value_len: Optional[int],
    ) -> None:
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._shards = shards
        # This defines C/F-order
        self._shards_cumprod = tuple(_cum_prod(shards))
        self._num_chunks_per_shard = reduce(lambda x, y: x*y, shards, 1)
        self._dimension_separator = dimension_separator
        # TODO: add jumptable for compressed data
        assert not chunk_has_constant_size, "Currently only uncompressed data can be used."
        self._chunk_has_constant_size = chunk_has_constant_size
        if not chunk_has_constant_size:
            assert value_len is not None
            self._fill_chunk = fill_value * value_len
        else:
            self._fill_chunk = None

        # TODO: add warnings for ineffective reads/writes:
        # * warn if partial reads are not available
        # * optionally warn on unaligned writes if no partial writes are available
  
    def __key_to_sharded__(self, key: str) -> Tuple[str, int]:
        # TODO: allow to be in a group (aka only use last parts for dimensions)
        subkeys = map(int, key.split(self._dimension_separator))

        shard_tuple, index_tuple = zip(*((subkey // shard_i, subkey % shard_i) for subkey, shard_i in zip(subkeys, self._shards)))
        shard_key = self._dimension_separator.join(map(str, shard_tuple))
        index = sum(i * j for i, j in zip(index_tuple, self._shards_cumprod))
        return shard_key, index

    def __get_chunk_slice__(self, shard_key: str, shard_index: int) -> Tuple[int, int]:
        # TODO: here we would use the jumptable for compression
        start = shard_index * len(self._fill_chunk)
        return slice(start, start + len(self._fill_chunk))

    def __getitem__(self, key: str) -> bytes:
        shard_key, shard_index = self.__key_to_sharded__(key)
        chunk_slice = self.__get_chunk_slice__(shard_key, shard_index)
        # TODO use partial reads if available
        full_shard_value = self._store[shard_key]
        return full_shard_value[chunk_slice]

    def __setitem__(self, key: str, value: bytes) -> None:
        shard_key, shard_index = self.__key_to_sharded__(key)
        if shard_key in self._store:
            full_shard_value = bytearray(self._store[shard_key])
        else:
            full_shard_value = bytearray(self._fill_chunk * self._num_chunks_per_shard)
        chunk_slice = self.__get_chunk_slice__(shard_key, shard_index)
        # TODO use partial writes if available
        full_shard_value[chunk_slice] = value
        self._store[shard_key] = full_shard_value

    def __delitem__(self, key) -> None:
        # TODO not implemented yet
        # For uncompressed chunks, deleting the "last" chunk might need to be detected.
        raise NotImplementedError("Deletion is not yet implemented")

    def __iter__(self) -> Iterator[str]:
        for shard_key in self._store.__iter__():
            if any(shard_key.endswith(i) for i in (array_meta_key, group_meta_key, attrs_key)):
                yield shard_key
            else:
                # TODO: allow to be in a group (aka only use last parts for dimensions)
                subkeys = tuple(map(int, shard_key.split(self._dimension_separator)))
                for offset in product(*(range(i) for i in self._shards)):
                    original_key = (subkeys_i * shards_i + offset_i for subkeys_i, offset_i, shards_i in zip(subkeys, offset, self._shards))
                    yield self._dimension_separator.join(map(str, original_key))

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    # TODO: For efficient reads and writes, we need to implement
    # getitems, setitems & delitems
    # and combine writes/reads/deletions to the same shard.
