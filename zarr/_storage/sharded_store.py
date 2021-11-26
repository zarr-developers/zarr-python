from collections import defaultdict
from functools import reduce
import math
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np

from zarr._storage.store import BaseStore, Store
from zarr.storage import StoreLike, array_meta_key, attrs_key, group_meta_key


def _cum_prod(x: Iterable[int]) -> Iterable[int]:
    prod = 1
    yield prod
    for i in x[:-1]:
        prod *= i
        yield prod


class MortonOrderShardedStore(Store):
    """This class should not be used directly,
    but is added to an Array as a wrapper when needed automatically."""

    def __init__(
        self,
        store: StoreLike,
        shards: Tuple[int, ...],
        dimension_separator: str,
        are_chunks_compressed: bool,
        dtype: np.dtype,
        fill_value: Any,
        chunk_size: int,
    ) -> None:
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._shards = shards
        self._num_chunks_per_shard = reduce(lambda x, y: x*y, shards, 1)
        self._dimension_separator = dimension_separator

        chunk_has_constant_size = not are_chunks_compressed and not dtype == object
        assert chunk_has_constant_size, "Currently only uncompressed, fixed-length data can be used."
        self._chunk_has_constant_size = chunk_has_constant_size
        if chunk_has_constant_size:
            binary_fill_value = np.full(1, fill_value=fill_value or 0, dtype=dtype).tobytes()
            self._fill_chunk = binary_fill_value * chunk_size
            self._emtpy_meta = b"\x00" * math.ceil(self._num_chunks_per_shard / 8)

        # unused when using Morton order
        self._shard_strides = tuple(_cum_prod(shards))

        # TODO: add warnings for ineffective reads/writes:
        # * warn if partial reads are not available
        # * optionally warn on unaligned writes if no partial writes are available

    def __get_meta__(self, shard_content: Union[bytes, bytearray]) -> int:
        return int.from_bytes(shard_content[-len(self._emtpy_meta):], byteorder="big")

    def __set_meta__(self, shard_content: bytearray, meta: int) -> None:
        shard_content[-len(self._emtpy_meta):] = meta.to_bytes(len(self._emtpy_meta), byteorder="big")

    # The following two methods define the order of the chunks in a shard
    # TODO use morton order
    def __chunk_key_to_shard_key_and_index__(self, chunk_key: str) -> Tuple[str, int]:
        # TODO: allow to be in a group (aka only use last parts for dimensions)
        chunk_subkeys = map(int, chunk_key.split(self._dimension_separator))

        shard_tuple, index_tuple = zip(*((subkey // shard_i, subkey % shard_i) for subkey, shard_i in zip(chunk_subkeys, self._shards)))
        shard_key = self._dimension_separator.join(map(str, shard_tuple))
        index = sum(i * j for i, j in zip(index_tuple, self._shard_strides))
        return shard_key, index

    def __shard_key_and_index_to_chunk_key__(self, shard_key_tuple: Tuple[int, ...], shard_index: int) -> str:
        offset = tuple(shard_index % s2 // s1 for s1, s2 in zip(self._shard_strides, self._shard_strides[1:] + (self._num_chunks_per_shard,)))
        original_key = (shard_key_i * shards_i + offset_i for shard_key_i, offset_i, shards_i in zip(shard_key_tuple, offset, self._shards))
        return self._dimension_separator.join(map(str, original_key))

    def __keys_to_shard_groups__(self, keys: Iterable[str]) -> Dict[str, List[Tuple[str, str]]]:
        shard_indices_per_shard_key = defaultdict(list)
        for chunk_key in keys:
            shard_key, shard_index = self.__chunk_key_to_shard_key_and_index__(chunk_key)
            shard_indices_per_shard_key[shard_key].append((shard_index, chunk_key))
        return shard_indices_per_shard_key

    def __get_chunk_slice__(self, shard_index: int) -> Tuple[int, int]:
        start = shard_index * len(self._fill_chunk)
        return slice(start, start + len(self._fill_chunk))

    def __getitem__(self, key: str) -> bytes:
        return self.getitems([key])[key]

    def getitems(self, keys: Iterable[str], **kwargs) -> Dict[str, bytes]:
        result = {}
        for shard_key, chunks_in_shard in self.__keys_to_shard_groups__(keys).items():
            # TODO use partial reads if available
            full_shard_value = self._store[shard_key]
            # TODO omit items if they don't exist
            for shard_index, chunk_key in chunks_in_shard:
                result[chunk_key] = full_shard_value[self.__get_chunk_slice__(shard_index)]
        return result

    def __setitem__(self, key: str, value: bytes) -> None:
        self.setitems({key: value})

    def setitems(self, values: Dict[str, bytes]) -> None:
        for shard_key, chunks_in_shard in self.__keys_to_shard_groups__(values.keys()).items():
            if len(chunks_in_shard) == self._num_chunks_per_shard:
                # TODO shards at a non-dataset-size aligned surface are not captured here yet
                full_shard_value = b"".join(
                    values[chunk_key] for _, chunk_key in sorted(chunks_in_shard)
                ) + b"\xff" * len(self._emtpy_meta)
                self._store[shard_key] = full_shard_value
            else:
                # TODO use partial writes if available
                try:
                    full_shard_value = bytearray(self._store[shard_key])
                except KeyError:
                    full_shard_value = bytearray(self._fill_chunk * self._num_chunks_per_shard + self._emtpy_meta)
                chunk_mask = self.__get_meta__(full_shard_value)
                for shard_index, chunk_key in chunks_in_shard:
                    chunk_mask |= 1 << shard_index
                    full_shard_value[self.__get_chunk_slice__(shard_index)] = values[chunk_key]
                self.__set_meta__(full_shard_value, chunk_mask)
                self._store[shard_key] = full_shard_value

    def __delitem__(self, key) -> None:
        # TODO not implemented yet, also delitems
        # Deleting the "last" chunk in a shard needs to remove the whole shard
        raise NotImplementedError("Deletion is not yet implemented")

    def __iter__(self) -> Iterator[str]:
        for shard_key in self._store.__iter__():
            if any(shard_key.endswith(i) for i in (array_meta_key, group_meta_key, attrs_key)):
                # Special keys such as ".zarray" are passed on as-is
                yield shard_key
            else:
                # For each shard key in the wrapped store, all corresponding chunks are yielded.
                # TODO: allow to be in a group (aka only use last parts for dimensions)
                shard_key_tuple = tuple(map(int, shard_key.split(self._dimension_separator)))
                mask = self.__get_meta__(self._store[shard_key])
                for i in range(self._num_chunks_per_shard):
                    if mask == 0:
                        break
                    if mask & 1:
                        yield self.__shard_key_and_index_to_chunk_key__(shard_key_tuple, i)
                    mask >>= 1

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())


SHARDED_STORES = {
    "morton_order": MortonOrderShardedStore,
}
