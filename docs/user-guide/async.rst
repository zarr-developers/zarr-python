
Async Zarr
==========

Zarr-Python 3 added a new asynchronous API for reading and writing data using `asyncio`.

The usage of the async API mirrors the synchronous API.

.. ipython:: python

   import zarr.api.asynchronous as async_zarr

   z = await async_zarr.open(mode='w', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
   z

   await z.setitem((0, 0), 42)

   await z.getitem((slice(0, 4), slice(0, 1)))

See the API docs for more details:

* :mod:`zarr.api.asynchronous`
* :class:`zarr.AsyncArray`
* :class:`zarr.AsyncGroup`
