.. _user-guide-gpu:

Using GPUs with Zarr
====================

Zarr can use GPUs to accelerate your workload by running
:meth:`zarr.config.enable_gpu`.

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

For Zstd compressed buffers, zarr will use the `nvcomp <https://docs.nvidia.com/cuda/nvcomp/samples/python_samples.html>`_
library to compress and decompress data on the GPU.