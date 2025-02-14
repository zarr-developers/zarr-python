.. _user-guide-gpu:

Using GPUs with Zarr
====================

Zarr can use GPUs to accelerate your workload by running
:meth:`zarr.config.enable_gpu`.

.. note::

   `zarr-python` currently supports reading the ndarray data into device (GPU)
   memory as the final stage of the codec pipeline. Data will still be read into
   or copied to host (CPU) memory for encoding and decoding.

   In the future, codecs will be available compressing and decompressing data on
   the GPU, avoiding the need to move data between the host and device for
   compression and decompression.

Reading data into device memory
-------------------------------

:meth:`zarr.config.enable_gpu` configures Zarr to use GPU memory for the data
buffers used internally by Zarr.

.. code-block:: python

   >>> import zarr
   >>> import cupy as cp  # doctest: +SKIP
   >>> zarr.config.enable_gpu()  # doctest: +SKIP
   >>> store = zarr.storage.MemoryStore()  # doctest: +SKIP
   >>> z = zarr.create_array(  # doctest: +SKIP
   ...     store=store, shape=(100, 100), chunks=(10, 10), dtype="float32",
   ... )
   >>> type(z[:10, :10])  # doctest: +SKIP
   cupy.ndarray

Note that the output type is a ``cupy.ndarray`` rather than a NumPy array.
