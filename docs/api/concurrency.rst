Concurrency
===========

Zarr supports concurrent reads and writes to distinct blocks through the use of an `Executor <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor>`__ object.
The read and write routines of a :class:`zarr.core.Array` accept an optional ``executor`` keyword argument which controls how or if concurrent execution should be performed.
By default, or if ``executor=None``, all blocks will be read and written serially.

.. warning::

   Not all executors can be used with all stores safely.
   For example, a ``ThreadPoolExecutor`` may only be used if the underlying store is in fact thread safe.

For stores where the data is already in memory or can be read very quickly, serial execution will likely be the fastest type of execution.
A concurrent executor is particularly useful when there is a high IO cost to retrieving a block, for example, with a store that reads data from some cloud object storage like Amazon S3.
In the case of some cloud object storage, a concurrent executor allows the Zarr to submit all of the web requests at once, instead of executing many web requests serially.
