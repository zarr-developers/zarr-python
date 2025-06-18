# Permissions in Zarr Python

Write/delete permissions in Zarr Python are confusing. This document aims to describe how they currently work.

# Data models

Zarr Python has two data models (`Array` and `Group`) and one storage model (`Store`). Only the store has a concept of write/delete permissions. Both `write` and `delete` permissions on `Store` instances are controlled by the `read_only` immutable property. Permissions on `Store` classes (i.e., implementations) are also influenced by the `supports_writes` and `supports_deletes` property, which should be the same for all instances of a class.

`Array` and `Group` do not have any permissions, instead they store a reference to a store that has permissions.

## Store properties related to permissions

### `supports_writes`

This is a property of the *class* that should indicate whether the store implements the following methods:

- `async def set(self, key: str, value: Buffer) -> None:`
- `async def set_if_not_exists(self, key: str, value: Buffer) -> None:`

`supports_writes` is primarily used by tests to determine the expected result of write operations. It is not used by the library to enforce permissions.

### `supports_deletes`

This is a property of the *class* that should indicate whether the store implements the following methods:

- `async def delete(self, key: str) -> None:`

The `supports_deletes` property is used by the `Store` abstract base class to determine whether it should delete all keys under a given prefix using `store.delete_dir(prefix)`.

The `supports_deletes` property is used by the `Array` and `Group` classes before calling `store.delete_dir(prefix)` to determine whether the store supports deleting keys when those data classes are opened with `overwrite=True`. If a store does not support deletes, the `Array` and `Group` classes check if an array or group is identified in that location via a `.zarray`, `.zgroup`, or `zarr.json` key. If such a key exists, the `Array` or `Group` will raise an error without trying to delete it. If a store does support deletes, the `Array` and `Group` classes will attempt to recursively delete the keys in the store using `store.delete_dir(prefix)`.

The `supports_deletes` property is also used by the testing framework to determine the expected result of delete operations.

!!! note
    Store implementations are agnostic to the Zarr data model. They will delete everything under the given prefix, regardless of whether it is an array, group, or unrelated to the Zarr store.

### `supports_listing`

This is a property of the *class* that should indicate whether the store implements the following method:

- `async def list(self, prefix: str = '', delimiter: str = '') -> List[str]:`