# zarr

A minimal implementation of chunked, compressed, N-dimensional arrays for 
Python.

## Installation

Install from GitHub (requires NumPy and Cython pre-installed):

```
$ pip install -U git+https://github.com/alimanfoo/zarr.git@master
```

## Status

Highly experimental, pre-alpha. Bug reports and pull requests very welcome.

## Usage

```python
>>> import numpy as np
>>> import zarr
```

Create a 1-dimensional array:

```python
>>> z = zarr.empty(10000000, dtype='i4', chunks=100000)
>>> z
zarr.ext.Array((10000000,), int32, chunks=(100000,), nbytes=38.1M, cbytes=0, cname=blosclz, clevel=5, shuffle=1)
```

TODO
