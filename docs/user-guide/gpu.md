# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by running `zarr.Config.enable_gpu`.

!!! note
    `zarr-python` currently supports reading the ndarray data into device (GPU)
    memory as the final stage of the codec pipeline. Data will still be read into
    or copied to host (CPU) memory for encoding and decoding.

    In the future, codecs will be available compressing and decompressing data on
    the GPU, avoiding the need to move data between the host and device for
    compression and decompression.

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
