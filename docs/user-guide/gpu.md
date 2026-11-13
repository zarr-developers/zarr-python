# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by running `zarr.config.enable_gpu()`.

## Installation

Zarr's GPU support requires [CuPy](https://cupy.dev), which in turn requires a
CUDA-compatible NVIDIA GPU. CuPy and nvCOMP can be installed alongside Zarr with the `cuda`
extra (see [Installation](installation.md) for the other optional dependency groups):

```console
pip install "zarr[cuda]"
```

This installs the `cupy-cuda12x` and `nvidia-nvcomp-cu12` packages. If you need a CuPy build for a different
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

For supported codecs, data will be decoded using the GPU via the [nvcomp] library.
See [runtime-configuration][config-runtime-configuration] for more. Issues and feature requests for NVIDIA nvCOMP can be reported on the nvcomp [issue tracker].

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

[nvcomp]: https://docs.nvidia.com/cuda/nvcomp/samples/python_samples.html
[issue tracker]: https://github.com/NVIDIA/CUDALibrarySamples/issues
