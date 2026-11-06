# Permissions in Zarr Python

Write/delete permissions in Zarr Python are confusing. This document aims to describe how they currently work.

# Data models

Zarr Python has two data models (`Array` and `Group`) and one storage model (`Store`). Only the store has a concept of write/delete permissions. Both `write` and `delete` permissions on `Store` instances are controlled by the `read_only` immutable property. Permissions on `Store` classes (i.e., implementations) are also influenced by the `supports_writes` and `supports_deletes` property, which should be the same for all instances of a class.

`Array` and `Group` do not have any permissions, instead they store a reference to a store that has permissions.

## Store properties related to permissions

### Instance properties

####  `read_only`

The `read_only` property indicates whether store *instances* should allow `set`, `delete` operations and their permutations. If `read_only` is `True`, then the store should reject any write or delete operations. If `read_only` is `False`, then the store should allow write and delete operations. The property is tested by `Store` methods by calling `self._check_writable()`, which raises a `ValueError` if the store's `read_only` property is true.

The `read_only` property is one of the most likely places to encounter a bug for a few reasons:

- `Store` implementations must remember to call `super.__init__(read_only=read_only)` in their `__init__` method to set the `read_only` property correctly.
- `Store` implementations must remember to call `self._check_writable()` in their `set` and `delete` methods to enforce the `read_only` property.
- `Array` and `Group` classes must remember to check alignment with the `read_only` property of the store with any `overwrite` arguments.
- Top level API functions must remember to check the `read_only` property of the store when creating new arrays or groups. This is complicated by the API functions using "mode" semantics like "w", "r", "a", etc., which are not directly related to the `read_only` property. Each function typically has its own logic for matching mode semntics to the `read_only` property of the store.

This is one of the most likely place to encounter a bug where a `read_only` property is not respected because implementations must remember to call `self._check_writable()` when implementing `store.set()`, which is not implemented in the `Store` abstract base class.

The Zarr spec does not seem to define how APIs should constrain write/delete permissions at the instance level.

### Class properties

The Zarr spec provides distinctions between readable, writeable, and listable stores, but does not define how to distinguish between these groups of store operations. The Zarr Python library has adopted the following properties to distinguish between these groups of operations at the *class* level, which are used by the `Store` abstract base class and the testing framework.

#### `supports_writes`

This is a property of the *class* that should indicate whether the store implements the following methods:

- `async def set(self, key: str, value: Buffer) -> None:`
- `async def set_if_not_exists(self, key: str, value: Buffer) -> None:`

`supports_writes` is primarily used by tests to determine the expected result of write operations. It is not used by the library to enforce permissions.

#### `supports_partial_writes`

The purpose of this property of the *class* is currently ambiguous.

One interpretation is that it indicates whether the store implements the following methods:

- `async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, BytesLike]]) -> None:

But the `FsspecStore` class does not implement this method, but it does have `supports_partial_writes = True`.

Another interpretation is that it indicates whether the store supports a `byte_range` argument in the `set` method.

#### `supports_deletes`

This is a property of the *class* that should indicate whether the store implements the following methods:

- `async def delete(self, key: str) -> None:`

The `supports_deletes` property is used by the `Store` abstract base class to determine whether it should delete all keys under a given prefix using `store.delete_dir(prefix)`.

The `supports_deletes` property is used by the `Array` and `Group` classes before calling `store.delete_dir(prefix)` to determine whether the store supports deleting keys when those data classes are opened with `overwrite=True`. If a store does not support deletes, the `Array` and `Group` classes check if an array or group is identified in that location via a `.zarray`, `.zgroup`, or `zarr.json` key. If such a key exists, the `Array` or `Group` will raise an error without trying to delete it. If a store does support deletes, the `Array` and `Group` classes will attempt to recursively delete the keys in the store using `store.delete_dir(prefix)`.

The `supports_deletes` property is also used by the testing framework to determine the expected result of delete operations.

!!! note
    Store implementations are agnostic to the Zarr data model. They will delete everything under the given prefix, regardless of whether it is an array, group, or unrelated to the Zarr store.

#### `supports_listing`

This is a property of the *class* that should indicate whether the store implements the following method:

- `async def list(self, prefix: str = '', delimiter: str = '') -> List[str]:`

This used to determine whether the `Store` abstract base classes `is_empty`, `clear`, and `delete_dir`methods should raise a `NotImplementedError`.

The `supports_listing` property is also used by the testing framework to determine the expected result of list operations.
