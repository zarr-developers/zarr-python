# Working with attributes

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example:

```python
import zarr
store = zarr.storage.MemoryStore()
root = zarr.create_group(store=store)
root.attrs['foo'] = 'bar'
z = root.create_array(name='zzz', shape=(10000, 10000), dtype='int32')
z.attrs['baz'] = 42
z.attrs['qux'] = [1, 4, 7, 12]
sorted(root.attrs)
# ['foo']
'foo' in root.attrs
# True
root.attrs['foo']
# 'bar'
sorted(z.attrs)
# ['baz', 'qux']
z.attrs['baz']
# 42
z.attrs['qux']
# [1, 4, 7, 12]
```

Internally Zarr uses JSON to store array attributes, so attribute values must be
JSON serializable.
