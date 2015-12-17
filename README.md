# zarr

A minimal implementation of chunked, compressed, N-dimensional arrays for 
Python.

## Installation

Install from GitHub (requires NumPy and Cython pre-installed):

```bash
$ pip install -U git+https://github.com/alimanfoo/zarr.git@master
```

## Status

Highly experimental, pre-alpha. Bug reports and pull requests very welcome.

## Goals

* Chunking in multiple dimensions
* Concurrent reads
* Concurrent writes

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

Fill it with some data:

```python
>>> z[:] = np.arange(10000000, dtype='i4')
>>> z
zarr.ext.Array((10000000,), int32, chunks=(100000,), nbytes=38.1M, cbytes=718.5K, cratio=54.4, cname=blosclz, clevel=5, shuffle=1)

```

Obtain a NumPy array:

```python
>>> z[:]
array([      0,       1,       2, ..., 9999997, 9999998, 9999999], dtype=int32)

```

Create a 2-dimensional array:

```python
>>> z = zarr.empty((10000, 1000), dtype='i4', chunks=(1000, 100))
>>> z
zarr.ext.Array((10000, 1000), int32, chunks=(1000, 100), nbytes=38.1M, cbytes=0, cname=blosclz, clevel=5, shuffle=1)

```

Fill it with some data:

```python
>>> z[:] = np.arange(10000000, dtype='i4').reshape((10000, 1000))
>>> z
zarr.ext.Array((10000, 1000), int32, chunks=(1000, 100), nbytes=38.1M, cbytes=2.0M, cratio=19.3, cname=blosclz, clevel=5, shuffle=1)

```

Obtain a NumPy array:

```python
>>> z[:]
array([[      0,       1,       2, ...,     997,     998,     999],
       [   1000,    1001,    1002, ...,    1997,    1998,    1999],
       [   2000,    2001,    2002, ...,    2997,    2998,    2999],
       ..., 
       [9997000, 9997001, 9997002, ..., 9997997, 9997998, 9997999],
       [9998000, 9998001, 9998002, ..., 9998997, 9998998, 9998999],
       [9999000, 9999001, 9999002, ..., 9999997, 9999998, 9999999]], dtype=int32)

```

## Acknowledgments

``zarr`` borrows code heavily from [bcolz](http://bcolz.blosc.org/).
