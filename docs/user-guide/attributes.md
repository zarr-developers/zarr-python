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

Two kinds of value that Python's `json` module accepts are not valid JSON, and
the config controls how Zarr writes them. Each option takes `"allow"` (write
without comment), `"warn"` (write, and emit a `ZarrFutureWarning`) or `"raise"`
(refuse to write):

- Non-string keys, such as `{1: "a"}`, are written as strings, so they read
  back as different keys. `attributes.non_string_keys` defaults to `"warn"`,
  and this will become an error in a future version of Zarr.
- `NaN` and infinite floats are written as non-standard `NaN` / `Infinity`
  literals that other JSON parsers may reject.
  `attributes.non_finite_floats` defaults to `"allow"`. Set it to `"raise"` to
  make sure the documents Zarr writes are valid JSON.

```python exec="true" session="attributes" source="above" result="ansi"
arr = zarr.create_array(store="memory://attributes-nan-demo", shape=(1,), dtype="f8")
with zarr.config.set({"attributes.non_finite_floats": "raise"}):
    try:
        arr.attrs["scale"] = float("nan")
    except ValueError as e:
        print(e)
```

When copying a Zarr array with [`zarr.from_array`][], its attributes are
deep-copied by default so nested dictionaries and lists are independent of the
source. Deeply nested attributes can raise `RecursionError` during this copy,
even when the source array can be stored and reopened successfully. The threshold
depends on Python's recursion limit and the current call stack; it is not a fixed
Zarr nesting limit. Pass `attributes={}` if the copy should omit attributes, or
provide an explicit attribute dictionary to replace the inherited attributes.

When working with hierarchies that contain many arrays and groups, reading the
attributes of each node separately can be slow. See
[Consolidated metadata](consolidated_metadata.md) for a way to store the
metadata (including attributes) of all nodes in a hierarchy in a single
document.
