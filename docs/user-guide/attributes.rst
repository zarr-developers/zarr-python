.. _user-guide-attrs:

Working with attributes
=======================

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example::

   >>> import zarr
   >>> # TODO: replace with create_group after #2463
   >>> root = zarr.group()
   >>> root.attrs['foo'] = 'bar'
   >>> z = root.zeros(name='zzz', shape=(10000, 10000))
   >>> z.attrs['baz'] = 42
   >>> z.attrs['qux'] = [1, 4, 7, 12]
   >>> sorted(root.attrs)
   ['foo']
   >>> 'foo' in root.attrs
   True
   >>> root.attrs['foo']
   'bar'
   >>> sorted(z.attrs)
   ['baz', 'qux']
   >>> z.attrs['baz']
   42
   >>> z.attrs['qux']
   [1, 4, 7, 12]

Internally Zarr uses JSON to store array attributes, so attribute values must be
JSON serializable.
