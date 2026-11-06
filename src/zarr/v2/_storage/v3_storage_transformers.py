import functools
import itertools
import os
from typing import NamedTuple, Tuple, Optional, Union, Iterator

from numcodecs.compat import ensure_bytes
import numpy as np

from zarr._storage.store import StorageTransformer, StoreV3, _rmdir_from_keys_v3
from zarr.util import normalize_storage_path
from zarr.types import DIMENSION_SEPARATOR


MAX_UINT_64 = 2**64 - 1


v3_sharding_available = os.environ.get("ZARR_V3_SHARDING", "0").lower() not in ["0", "false"]


def assert_zarr_v3_sharding_available():
    if not v3_sharding_available:
        raise NotImplementedError(
            "Using V3 sharding is experimental and not yet finalized! To enable support, set:\n"
            "ZARR_V3_SHARDING=1"
        )  # pragma: no cover


class _ShardIndex(NamedTuple):
    store: "ShardingStorageTransformer"
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

    def __localize_chunk__(self, chunk: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(
            chunk_i % shard_i for chunk_i, shard_i in zip(chunk, self.store.chunks_per_shard)
        )

    def is_all_empty(self) -> bool:
        return np.array_equiv(self.offsets_and_lengths, MAX_UINT_64)

    def get_chunk_slice(self, chunk: Tuple[int, ...]) -> Optional[slice]:
        localized_chunk = self.__localize_chunk__(chunk)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return slice(int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(self, chunk: Tuple[int, ...], chunk_slice: Optional[slice]) -> None:
        localized_chunk = self.__localize_chunk__(chunk)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start,
            )

    def to_bytes(self) -> bytes:
        return self.offsets_and_lengths.tobytes(order="C")

    @classmethod
    def from_bytes(
        cls, buffer: Union[bytes, bytearray], store: "ShardingStorageTransformer"
    ) -> "_ShardIndex":
        try:
            return cls(
                store=store,
                offsets_and_lengths=np.frombuffer(bytearray(buffer), dtype="<u8").reshape(
                    *store.chunks_per_shard, 2, order="C"
                ),
            )
        except ValueError as e:  # pragma: no cover
            raise RuntimeError from e

    @classmethod
    def create_empty(cls, store: "ShardingStorageTransformer"):
        # reserving 2*64bit per chunk for offset and length:
        return cls.from_bytes(
            MAX_UINT_64.to_bytes(8, byteorder="little") * (2 * store._num_chunks_per_shard),
            store=store,
        )


class ShardingStorageTransformer(StorageTransformer):  # lgtm[py/missing-equals]
    """Implements sharding as a storage transformer, as described in the spec:
    https://zarr-specs.readthedocs.io/en/latest/extensions/storage-transformers/sharding/v1.0.html
    https://purl.org/zarr/spec/storage_transformers/sharding/1.0
    """

    extension_uri = "https://purl.org/zarr/spec/storage_transformers/sharding/1.0"
    valid_types = ["indexed"]

    def __init__(self, _type, chunks_per_shard) -> None:
        assert_zarr_v3_sharding_available()
        super().__init__(_type)
        if isinstance(chunks_per_shard, int):
            chunks_per_shard = (chunks_per_shard,)
        else:
            chunks_per_shard = tuple(int(i) for i in chunks_per_shard)
            if chunks_per_shard == ():
                chunks_per_shard = (1,)
        self.chunks_per_shard = chunks_per_shard
        self._num_chunks_per_shard = functools.reduce(lambda x, y: x * y, chunks_per_shard, 1)
        self._dimension_separator = None
        self._data_key_prefix = None

    def _copy_for_array(self, array, inner_store):
        transformer_copy = super()._copy_for_array(array, inner_store)
        transformer_copy._dimension_separator = array._dimension_separator
        transformer_copy._data_key_prefix = array._data_key_prefix
        if len(array._shape) > len(self.chunks_per_shard):
            # The array shape might be longer when initialized with subdtypes.
            # subdtypes dimensions come last, therefore padding chunks_per_shard
            # with ones, effectively disabling sharding on the unlisted dimensions.
            transformer_copy.chunks_per_shard += (1,) * (
                len(array._shape) - len(self.chunks_per_shard)
            )
        return transformer_copy

    @property
    def dimension_separator(self) -> DIMENSION_SEPARATOR:
        assert (
            self._dimension_separator is not None
        ), "dimension_separator is not initialized, first get a copy via _copy_for_array."
        return self._dimension_separator

    def _is_data_key(self, key: str) -> bool:
        assert (
            self._data_key_prefix is not None
        ), "data_key_prefix is not initialized, first get a copy via _copy_for_array."
        return key.startswith(self._data_key_prefix)

    def _key_to_shard(self, chunk_key: str) -> Tuple[str, Tuple[int, ...]]:
        prefix, _, chunk_string = chunk_key.rpartition("c")
        chunk_subkeys = (
            tuple(map(int, chunk_string.split(self.dimension_separator))) if chunk_string else (0,)
        )
        shard_key_tuple = (
            subkey // shard_i for subkey, shard_i in zip(chunk_subkeys, self.chunks_per_shard)
        )
        shard_key = prefix + "c" + self.dimension_separator.join(map(str, shard_key_tuple))
        return shard_key, chunk_subkeys

    def _get_index_from_store(self, shard_key: str) -> _ShardIndex:
        # At the end of each shard 2*64bit per chunk for offset and length define the index:
        index_bytes = self.inner_store.get_partial_values(
            [(shard_key, (-16 * self._num_chunks_per_shard, None))]
        )[0]
        if index_bytes is None:
            raise KeyError(shard_key)
        return _ShardIndex.from_bytes(
            index_bytes,
            self,
        )

    def _get_index_from_buffer(self, buffer: Union[bytes, bytearray]) -> _ShardIndex:
        # At the end of each shard 2*64bit per chunk for offset and length define the index:
        return _ShardIndex.from_bytes(buffer[-16 * self._num_chunks_per_shard :], self)

    def _get_chunks_in_shard(self, shard_key: str) -> Iterator[Tuple[int, ...]]:
        _, _, chunk_string = shard_key.rpartition("c")
        shard_key_tuple = (
            tuple(map(int, chunk_string.split(self.dimension_separator))) if chunk_string else (0,)
        )
        for chunk_offset in itertools.product(*(range(i) for i in self.chunks_per_shard)):
            yield tuple(
                shard_key_i * shards_i + offset_i
                for shard_key_i, offset_i, shards_i in zip(
                    shard_key_tuple, chunk_offset, self.chunks_per_shard
                )
            )

    def __getitem__(self, key):
        if self._is_data_key(key):
            if self.supports_efficient_get_partial_values:
                # Use the partial implementation, which fetches the index separately
                value = self.get_partial_values([(key, (0, None))])[0]
                if value is None:
                    raise KeyError(key)
                else:
                    return value
            shard_key, chunk_subkey = self._key_to_shard(key)
            try:
                full_shard_value = self.inner_store[shard_key]
            except KeyError as e:
                raise KeyError(key) from e
            index = self._get_index_from_buffer(full_shard_value)
            chunk_slice = index.get_chunk_slice(chunk_subkey)
            if chunk_slice is not None:
                return full_shard_value[chunk_slice]
            else:
                raise KeyError(key)
        else:
            return self.inner_store.__getitem__(key)

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        if self._is_data_key(key):
            shard_key, chunk_subkey = self._key_to_shard(key)
            chunks_to_read = set(self._get_chunks_in_shard(shard_key))
            chunks_to_read.remove(chunk_subkey)
            new_content = {chunk_subkey: value}
            try:
                if self.supports_efficient_get_partial_values:
                    index = self._get_index_from_store(shard_key)
                    full_shard_value = None
                else:
                    full_shard_value = self.inner_store[shard_key]
                    index = self._get_index_from_buffer(full_shard_value)
            except KeyError:
                index = _ShardIndex.create_empty(self)
            else:
                chunk_slices = [
                    (chunk_to_read, index.get_chunk_slice(chunk_to_read))
                    for chunk_to_read in chunks_to_read
                ]
                valid_chunk_slices = [
                    (chunk_to_read, chunk_slice)
                    for chunk_to_read, chunk_slice in chunk_slices
                    if chunk_slice is not None
                ]
                # use get_partial_values if less than half of the available chunks must be read:
                # (This can be changed when set_partial_values can be used efficiently.)
                use_partial_get = (
                    self.supports_efficient_get_partial_values
                    and len(valid_chunk_slices) < len(chunk_slices) / 2
                )

                if use_partial_get:
                    chunk_values = self.inner_store.get_partial_values(
                        [
                            (
                                shard_key,
                                (
                                    chunk_slice.start,
                                    chunk_slice.stop - chunk_slice.start,
                                ),
                            )
                            for _, chunk_slice in valid_chunk_slices
                        ]
                    )
                    for chunk_value, (chunk_to_read, _) in zip(chunk_values, valid_chunk_slices):
                        new_content[chunk_to_read] = chunk_value
                else:
                    if full_shard_value is None:
                        full_shard_value = self.inner_store[shard_key]
                    for chunk_to_read, chunk_slice in valid_chunk_slices:
                        if chunk_slice is not None:
                            new_content[chunk_to_read] = full_shard_value[chunk_slice]

            shard_content = b""
            for chunk_subkey, chunk_content in new_content.items():
                chunk_slice = slice(len(shard_content), len(shard_content) + len(chunk_content))
                index.set_chunk_slice(chunk_subkey, chunk_slice)
                shard_content += chunk_content
            # Appending the index at the end of the shard:
            shard_content += index.to_bytes()
            self.inner_store[shard_key] = shard_content
        else:  # pragma: no cover
            self.inner_store[key] = value

    def __delitem__(self, key):
        if self._is_data_key(key):
            shard_key, chunk_subkey = self._key_to_shard(key)
            try:
                index = self._get_index_from_store(shard_key)
            except KeyError as e:
                raise KeyError(key) from e

            index.set_chunk_slice(chunk_subkey, None)

            if index.is_all_empty():
                del self.inner_store[shard_key]
            else:
                index_bytes = index.to_bytes()
                self.inner_store.set_partial_values([(shard_key, -len(index_bytes), index_bytes)])
        else:  # pragma: no cover
            del self.inner_store[key]

    def _shard_key_to_original_keys(self, key: str) -> Iterator[str]:
        if self._is_data_key(key):
            index = self._get_index_from_store(key)
            prefix, _, _ = key.rpartition("c")
            for chunk_tuple in self._get_chunks_in_shard(key):
                if index.get_chunk_slice(chunk_tuple) is not None:
                    yield prefix + "c" + self.dimension_separator.join(map(str, chunk_tuple))
        else:
            yield key

    def __iter__(self) -> Iterator[str]:
        for key in self.inner_store:
            yield from self._shard_key_to_original_keys(key)

    def __len__(self):
        return sum(1 for _ in self.keys())

    def get_partial_values(self, key_ranges):
        if self.supports_efficient_get_partial_values:
            transformed_key_ranges = []
            cached_indices = {}
            none_indices = []
            for i, (key, range_) in enumerate(key_ranges):
                if self._is_data_key(key):
                    shard_key, chunk_subkey = self._key_to_shard(key)
                    try:
                        index = cached_indices[shard_key]
                    except KeyError:
                        try:
                            index = self._get_index_from_store(shard_key)
                        except KeyError:
                            none_indices.append(i)
                            continue
                        cached_indices[shard_key] = index
                    chunk_slice = index.get_chunk_slice(chunk_subkey)
                    if chunk_slice is None:
                        none_indices.append(i)
                        continue
                    range_start, range_length = range_
                    if range_length is None:
                        range_length = chunk_slice.stop - chunk_slice.start
                    transformed_key_ranges.append(
                        (shard_key, (range_start + chunk_slice.start, range_length))
                    )
                else:  # pragma: no cover
                    transformed_key_ranges.append((key, range_))
            values = self.inner_store.get_partial_values(transformed_key_ranges)
            for i in none_indices:
                values.insert(i, None)
            return values
        else:
            return StoreV3.get_partial_values(self, key_ranges)

    def supports_efficient_set_partial_values(self):
        return False

    def set_partial_values(self, key_start_values):
        # This does not yet implement efficient set_partial_values
        StoreV3.set_partial_values(self, key_start_values)

    def rename(self, src_path: str, dst_path: str) -> None:
        StoreV3.rename(self, src_path, dst_path)  # type: ignore[arg-type]

    def list_prefix(self, prefix):
        return StoreV3.list_prefix(self, prefix)

    def erase_prefix(self, prefix):
        if self._is_data_key(prefix):
            StoreV3.erase_prefix(self, prefix)
        else:
            self.inner_store.erase_prefix(prefix)

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        _rmdir_from_keys_v3(self, path)

    def __contains__(self, key):
        if self._is_data_key(key):
            shard_key, chunk_subkeys = self._key_to_shard(key)
            try:
                index = self._get_index_from_store(shard_key)
            except KeyError:
                return False
            chunk_slice = index.get_chunk_slice(chunk_subkeys)
            return chunk_slice is not None
        else:
            return self._inner_store.__contains__(key)
