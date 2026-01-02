# Working with groups

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the [`zarr.group`][] function:

```python exec="true" session="groups" source="above" result="ansi"
import zarr
store = zarr.storage.MemoryStore()
root = zarr.create_group(store=store)
print(root)
```

Groups have a similar API to the Group class from [h5py](https://www.h5py.org/).  For example, groups can contain other groups:

```python exec="true" session="groups" source="above"
foo = root.create_group('foo')
bar = foo.create_group('bar')
```

Groups can also contain arrays, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
z1 = bar.create_array(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z1)
```

Members of a group can be accessed via the suffix notation, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo'])
```

The '/' character can be used to access multiple levels of the hierarchy in one
call, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo/bar'])
```

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo/bar/baz'])
```

The [`zarr.Group.tree`][] method can be used to print a tree
representation of the hierarchy, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root.tree())
```

The [`zarr.open_group`][] function provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups stored in
sub-directories, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
root = zarr.open_group('data/group.zarr', mode='w')
print(root)
```

```python exec="true" session="groups" source="above" result="ansi"
z = root.create_array(name='foo/bar/baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z)
```

For more information on groups see the [`zarr.Group` API docs](../api/zarr/group.md).

## Batch Group Creation

You can also create multiple groups concurrently with a single function call. [`zarr.create_hierarchy`][] takes
a [`zarr Storage instance`](../api/zarr/storage.md) instance and a dict of `key : metadata` pairs, parses that dict, and
writes metadata documents to storage:

```python exec="true" session="groups" source="above" result="ansi"
from zarr import create_hierarchy
from zarr.core.group import GroupMetadata
from zarr.storage import LocalStore

from pprint import pprint
import io

node_spec = {'a/b/c': GroupMetadata()}
nodes_created = dict(create_hierarchy(store=LocalStore(root='data'), nodes=node_spec))
# Report nodes (pprint is used for cleaner rendering in the docs)
output = io.StringIO()
pprint(nodes_created, stream=output, width=60)
print(output.getvalue())
```

Note that we only specified a single group named `a/b/c`, but 4 groups were created. These additional groups
were created to ensure that the desired node `a/b/c` is connected to the root group `''` by a sequence
of intermediate groups. [`zarr.create_hierarchy`][] normalizes the `nodes` keyword argument to
ensure that the resulting hierarchy is complete, i.e. all groups or arrays are connected to the root
of the hierarchy via intermediate groups.

Because [`zarr.create_hierarchy`][] concurrently creates metadata documents, it's more efficient
than repeated calls to [`create_group`][zarr.create_group] or [`create_array`][zarr.create_array], provided you can statically define
the metadata for the groups and arrays you want to create.

## Array and group diagnostics

Diagnostic information about arrays and groups is available via the `info`
property. E.g.:

```python exec="true" session="groups" source="above" result="ansi"
store = zarr.storage.MemoryStore()
root = zarr.group(store=store)
foo = root.create_group('foo')
bar = foo.create_array(name='bar', shape=1000000, chunks=100000, dtype='int64')
bar[:] = 42
baz = foo.create_array(name='baz', shape=(1000, 1000), chunks=(100, 100), dtype='float32')
baz[:] = 4.2
print(root.info)
```

```python exec="true" session="groups" source="above" result="ansi"
print(foo.info)
```

```python exec="true" session="groups" source="above" result="ansi"
print(bar.info_complete())
```

```python exec="true" session="groups" source="above" result="ansi"
print(baz.info)
```

Groups also have the [`zarr.Group.tree`][] method, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root.tree())
```

!!! note
    [`zarr.Group.tree`][] requires the optional [rich](https://rich.readthedocs.io/en/stable/) dependency. It can be installed with the `[tree]` extra.
