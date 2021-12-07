from collections import defaultdict
from functools import reduce
from itertools import product
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from zarr._storage.store import BaseStore, Store
from zarr.storage import StoreLike, array_meta_key, attrs_key, group_meta_key


class _ShardIndex(NamedTuple):
    store: "IndexedShardedStore"
    offsets_and_lengths: np.ndarray  # dtype uint64, shape (shards_0, _shards_1, ..., 2)

    def __localize_chunk__(self, chunk: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(chunk_i % shard_i for chunk_i, shard_i in zip(chunk, self.store._shards))

    def get_chunk_slice(self, chunk: Tuple[int, ...]) -> Optional[slice]:
        localized_chunk = self.__localize_chunk__(chunk)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if chunk_len == 0:
            return None
        else:
            return slice(chunk_start, chunk_start + chunk_len)

    def set_chunk_slice(self, chunk: Tuple[int, ...], chunk_slice: Optional[slice]) -> None:
        localized_chunk = self.__localize_chunk__(chunk)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (0, 0)
        else:
            self.offsets_and_lengths[localized_chunk] = (chunk_slice.start, chunk_slice.stop - chunk_slice.start)

    def to_bytes(self) -> bytes:
        return self.offsets_and_lengths.tobytes(order='C')

    @classmethod
    def from_bytes(cls, buffer: Union[bytes, bytearray], store: "IndexedShardedStore") -> "_ShardIndex":
        return cls(
            store=store,
            offsets_and_lengths=np.frombuffer(bytearray(buffer), dtype=">u8").reshape(*store._shards, 2, order="C")
        )

    @classmethod
    def create_empty(cls, store: "IndexedShardedStore"):
        # reserving 2*64bit per chunk for offset and length:
        return cls.from_bytes(b"\x00" * (16 * store._num_chunks_per_shard), store=store)


class IndexedShardedStore(Store):
    """This class should not be used directly,
    but is added to an Array as a wrapper when needed automatically."""

    def __init__(
        self,
        store: StoreLike,
        shards: Tuple[int, ...],
        dimension_separator: str,
    ) -> None:
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._shards = shards
        self._num_chunks_per_shard = reduce(lambda x, y: x*y, shards, 1)
        self._dimension_separator = dimension_separator

        # TODO: add warnings for ineffective reads/writes:
        # * warn if partial reads are not available
        # * optionally warn on unaligned writes if no partial writes are available

    def __keys_to_shard_groups__(self, keys: Iterable[str]) -> Dict[str, List[Tuple[str, str]]]:
        shard_indices_per_shard_key = defaultdict(list)
        for chunk_key in keys:
            # TODO: allow to be in a group (aka only use last parts for dimensions)
            chunk_subkeys = tuple(map(int, chunk_key.split(self._dimension_separator)))
            shard_key_tuple = (subkey // shard_i for subkey, shard_i in zip(chunk_subkeys, self._shards))
            shard_key = self._dimension_separator.join(map(str, shard_key_tuple))
            shard_indices_per_shard_key[shard_key].append((chunk_key, chunk_subkeys))
        return shard_indices_per_shard_key

    def __get_index__(self, buffer: Union[bytes, bytearray]) -> _ShardIndex:
        # At the end of each shard 2*64bit per chunk for offset and length define the index:
        return _ShardIndex.from_bytes(buffer[-16 * self._num_chunks_per_shard:], self)

    def __get_chunks_in_shard(self, shard_key: str) -> Iterator[Tuple[int, ...]]:
        # TODO: allow to be in a group (aka only use last parts for dimensions)
        shard_key_tuple = tuple(map(int, shard_key.split(self._dimension_separator)))
        for chunk_offset in product(*(range(i) for i in self._shards)):
            yield tuple(
                shard_key_i * shards_i + offset_i
                for shard_key_i, offset_i, shards_i
                in zip(shard_key_tuple, chunk_offset, self._shards)
            )

    def __getitem__(self, key: str) -> bytes:
        return self.getitems([key])[key]

    def getitems(self, keys: Iterable[str], **kwargs) -> Dict[str, bytes]:
        result = {}
        for shard_key, chunks_in_shard in self.__keys_to_shard_groups__(keys).items():
            # TODO use partial read if available
            full_shard_value = self._store[shard_key]
            index = self.__get_index__(full_shard_value)
            for chunk_key, chunk_subkeys in chunks_in_shard:
                chunk_slice = index.get_chunk_slice(chunk_subkeys)
                if chunk_slice is not None:
                    result[chunk_key] = full_shard_value[chunk_slice]
        return result

    def __setitem__(self, key: str, value: bytes) -> None:
        self.setitems({key: value})

    def setitems(self, values: Dict[str, bytes]) -> None:
        for shard_key, chunks_in_shard in self.__keys_to_shard_groups__(values.keys()).items():
            all_chunks = set(self.__get_chunks_in_shard(shard_key))
            chunks_to_set = set(chunk_subkeys for _chunk_key, chunk_subkeys in chunks_in_shard)
            chunks_to_read = all_chunks - chunks_to_set
            new_content = {chunk_subkeys: values[chunk_key] for chunk_key, chunk_subkeys in chunks_in_shard}
            try:
                # TODO use partial read if available
                full_shard_value = self._store[shard_key]
            except KeyError:
                index = _ShardIndex.create_empty(self)
                for chunk_to_read in chunks_to_read:
                    new_content[chunk_to_read] = b""
            else:
                index = self.__get_index__(full_shard_value)
                for chunk_to_read in chunks_to_read:
                    chunk_slice = index.get_chunk_slice(chunk_to_read)
                    if chunk_slice is None:
                        new_content[chunk_to_read] = b""
                    else:
                        new_content[chunk_to_read] = full_shard_value[chunk_slice]

            # TODO use partial write if available and possible (e.g. at the end)
            shard_content = b""
            # TODO: order the chunks in the shard:
            for chunk_subkeys, chunk_content in new_content.items():
                chunk_slice = slice(len(shard_content), len(shard_content) + len(chunk_content))
                index.set_chunk_slice(chunk_subkeys, chunk_slice)
                shard_content += chunk_content
            # Appending the index at the end of the shard:
            shard_content += index.to_bytes()
            self._store[shard_key] = shard_content

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
                # TODO: use partial read if available:
                index = self.__get_index__(self._store[shard_key])
                for chunk_tuple in self.__get_chunks_in_shard(shard_key):
                    if index.get_chunk_slice(chunk_tuple) is not None:
                        # TODO: if shard is in a group, prepend group-prefix to chunk
                        yield self._dimension_separator.join(map(str, chunk_tuple))

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())


SHARDED_STORES = {
    "indexed": IndexedShardedStore,
}
