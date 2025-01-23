.. _user-guide-gpu:

Using GPUs with Zarr
====================

Zarr can be used along with GPUs to accelerate your workload. Currently,
Zarr supports reading data into GPU memory. In the future, Zarr will
support GPU-accelerated codecs and file IO.

Reading data into device memory
-------------------------------

.. code-block:: python

   >>> import zarr
   >>> import cupy as cp
   >>> zarr.config.enable_cuda()
   >>> store = zarr.storage.MemoryStore()
   >>> z = zarr.create_array(store=store, shape=(100, 100), chunks=(10, 10), dtype="float32")
   >>> type(z[:10, :10])
   cupy.ndarray

:meth:`zarr.config.enable_cuda` updates the Zarr configuration to use device
memory for all data buffers used by Zarr. This means that any reads from a Zarr
store will return a CuPy ndarray rather than a NumPy ndarray. Any buffers used
for metadata will be on the host.