# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by running `zarr.Config.enable_gpu`.

## Reading data into device memory

[`zarr.config`][] configures Zarr to use GPU memory for the data
buffers used internally by Zarr via `enable_gpu()`.

```python
import zarr
import cupy as cp
zarr.config.enable_gpu()
store = zarr.storage.MemoryStore()
z = zarr.create_array(
    store=store, shape=(100, 100), chunks=(10, 10), dtype="float32",
)
type(z[:10, :10])
# cupy.ndarray
```

Note that the output type is a `cupy.ndarray` rather than a NumPy array.

For supported codecs, data will be decoded using the GPU via the [nvcomp] library.
See [`user-guide-config`][] for more. Isseus and feature requestsfor NVIDIA nvCOMP can be reported in the nvcomp [issue tracker].

[nvcomp]: https://docs.nvidia.com/cuda/nvcomp/samples/python_samples.html
[issue tradcker]: https://github.com/NVIDIA/CUDALibrarySamples/issues