# Working with attributes

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example:

```python exec="true" session="arrays" source="above" result="ansi"
import zarr
store = zarr.storage.MemoryStore()
root = zarr.create_group(store=store)
root.attrs['foo'] = 'bar'
z = root.create_array(name='zzz', shape=(10000, 10000), dtype='int32')
z.attrs['baz'] = 42
z.attrs['qux'] = [1, 4, 7, 12]
print(sorted(root.attrs))
```

```python exec="true" session="arrays" source="above" result="ansi"
print('foo' in root.attrs)
```

```python exec="true" session="arrays" source="above" result="ansi"
print(root.attrs['foo'])
```
```python exec="true" session="arrays" source="above" result="ansi"
print(sorted(z.attrs))
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.attrs['baz'])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.attrs['qux'])
```

Internally Zarr uses JSON to store array attributes, so attribute values must be
JSON serializable.
