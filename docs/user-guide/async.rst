
Asynchronous API
==========

Zarr-Python 3 added a new asynchronous API for reading and writing data using `asyncio`.

The usage of the async API mirrors the synchronous API.

.. ipython:: python

   import asyncio
   import numpy as np
   import zarr.api.asynchronous as async_zarr

   # create a new group using the asynchronous API
   root = await async_zarr.create_group(attributes={'foo': 'bar'})
   root
   # create an array using the AsyncGroup
   z = await root.create_array(name='foo', shape=(100, ), chunks=(5, ), dtype=np.uint32)
   z
   # set and get items
   await z.setitem((slice(None), ), np.arange(0, 100, dtype=z.dtype))
   # set a single item
   await z.setitem((7,), -99)
   # set a range to a constant value using a slice
   await z.setitem(slice(0, 3), -1)
   # get a slice of the array
   await z.getitem((slice(0, 5), ))
   # gather multiple items concurrently
   await asyncio.gather(z.getitem(slice(0, 5)), z.getitem(slice(6, 10)))

See the API docs for more details:

* :mod:`zarr.api.asynchronous`
* :class:`zarr.AsyncArray`
* :class:`zarr.AsyncGroup`
