# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by running `zarr.config.enable_gpu()`.

!!! note
    `zarr-python` currently supports reading the ndarray data into device (GPU)
    memory as the final stage of the codec pipeline. Data will still be read into
    or copied to host (CPU) memory for encoding and decoding.

    In the future, codecs will be available for compressing and decompressing data on
    the GPU, avoiding the need to move data between the host and device for
    compression and decompression.

## Installation

Zarr's GPU support requires [CuPy](https://cupy.dev), which in turn requires a
CUDA-compatible NVIDIA GPU. CuPy can be installed alongside Zarr with the `gpu`
extra (see [Installation](installation.md) for the other optional dependency groups):

```console
pip install "zarr[gpu]"
```

This installs the `cupy-cuda12x` package. If you need a CuPy build for a different
CUDA version, see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
and install the appropriate package yourself.

## Reading data into device memory

Calling `zarr.config.enable_gpu()` configures Zarr to use GPU memory for the data
buffers used internally by Zarr:

```python test="true" session="gpu-demo" markers="gpu" source="above"
import zarr
import cupy as cp

zarr.config.enable_gpu()
z = zarr.create_array(
    store="memory://gpu-demo", shape=(100, 100), chunks=(10, 10), dtype="float32",
)
assert isinstance(z[:10, :10], cp.ndarray)
```

Note that the arrays returned by reads are of type `cupy.ndarray` rather than
NumPy arrays.

`zarr.config.enable_gpu()` returns a [donfig](https://donfig.readthedocs.io/en/latest/)
`ConfigSet`, which can be used as a context manager to enable GPU support for a
limited scope:

```python test="true" session="gpu-demo" markers="gpu" source="above"
with zarr.config.enable_gpu():
    data = z[:10, :10]
assert isinstance(data, cp.ndarray)
```

Under the hood, `enable_gpu()` selects the GPU-backed buffer classes
`zarr.buffer.gpu.Buffer` and `zarr.buffer.gpu.NDBuffer` via the `buffer` and
`ndbuffer` configuration keys. See [Custom array buffers](extending.md#custom-array-buffers)
for more on Zarr's buffer classes, including how to implement your own.
