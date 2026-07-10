# Working with attributes

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example:

```python exec="true" session="attributes" source="above" result="ansi"
import zarr
root = zarr.create_group(store="memory://attributes-demo")
root.attrs['foo'] = 'bar'
z = root.create_array(name='zzz', shape=(10000, 10000), dtype='int32')
z.attrs['baz'] = 42
z.attrs['qux'] = [1, 4, 7, 12]
print(sorted(root.attrs))
```

```python exec="true" session="attributes" source="above" result="ansi"
print('foo' in root.attrs)
```

```python exec="true" session="attributes" source="above" result="ansi"
print(root.attrs['foo'])
```
```python exec="true" session="attributes" source="above" result="ansi"
print(sorted(z.attrs))
```

```python exec="true" session="attributes" source="above" result="ansi"
print(z.attrs['baz'])
```

```python exec="true" session="attributes" source="above" result="ansi"
print(z.attrs['qux'])
```

Attributes can be deleted with the `del` operator:

```python exec="true" session="attributes" source="above" result="ansi"
del z.attrs['baz']
print(sorted(z.attrs))
```

Note that each attribute assignment or deletion writes the node's metadata
document back to the store. To change several attributes in a single write,
use [`zarr.Array.update_attributes`][] (or [`zarr.Group.update_attributes`][]
for groups), which merges the given dict into the existing attributes and
returns the updated array or group:

```python exec="true" session="attributes" source="above" result="ansi"
z = z.update_attributes({'baz': 43, 'quux': True})
print(sorted(z.attrs))
```

Internally Zarr uses JSON to store array and group attributes, so attribute
values must be JSON serializable.

When working with hierarchies that contain many arrays and groups, reading the
attributes of each node separately can be slow. See
[Consolidated metadata](consolidated_metadata.md) for a way to store the
metadata (including attributes) of all nodes in a hierarchy in a single
document.
