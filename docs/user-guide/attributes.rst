.. _tutorial_attrs:

Working with attributes
=======================

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example:

.. ipython:: python

   root = zarr.group()

   root.attrs['foo'] = 'bar'

   z = root.zeros(name='zzz', shape=(10000, 10000))

   z.attrs['baz'] = 42

   z.attrs['qux'] = [1, 4, 7, 12]

   sorted(root.attrs)

   'foo' in root.attrs

   root.attrs['foo']

   sorted(z.attrs)

   z.attrs['baz']

   z.attrs['qux']

Internally Zarr uses JSON to store array attributes, so attribute values must be
JSON serializable.
