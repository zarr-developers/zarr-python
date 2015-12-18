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

## Design goals

* Chunking in multiple dimensions
* Resize any dimension
* Concurrent reads
* Concurrent writes
* Release the GIL during compression and decompression

## Usage

```python
>>> import numpy as np
>>> import zarr

```

Create an array:

```python
>>> z = zarr.empty((10000, 1000), dtype='i4', chunks=(1000, 100))
>>> z
zarr.ext.Array((10000, 1000), int32, chunks=(1000, 100), nbytes=38.1M, cbytes=0, cname=blosclz, clevel=5, shuffle=1)

```

Fill it with some data:

```python
>>> z[:] = np.arange(10000000, dtype='i4').reshape(10000, 1000)
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

Resize the array and add more data:

```python
>>> z.resize(20000, 1000)
>>> z
zarr.ext.Array((20000, 1000), int32, chunks=(1000, 100), nbytes=76.3M, cbytes=2.0M, cratio=38.5, cname=blosclz, clevel=5, shuffle=1)
>>> z[10000:, :] = np.arange(10000000, dtype='i4').reshape((10000, 1000))
>>> z
zarr.ext.Array((20000, 1000), int32, chunks=(1000, 100), nbytes=76.3M, cbytes=4.0M, cratio=19.3, cname=blosclz, clevel=5, shuffle=1)

```

For convenience, an `append` method is available, which can be used to 
append data to any axis:

```python
>>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
>>> z = zarr.array(a, chunks=(1000, 100))
>>> z
zarr.ext.Array((10000, 1000), int32, chunks=(1000, 100), nbytes=38.1M, cbytes=2.0M, cratio=19.3, cname=blosclz, clevel=5, shuffle=1)
>>> z.append(a+a)
>>> z
zarr.ext.Array((20000, 1000), int32, chunks=(1000, 100), nbytes=76.3M, cbytes=3.6M, cratio=21.2, cname=blosclz, clevel=5, shuffle=1)
>>> z.append(np.vstack([a, a]), axis=1)
>>> z
zarr.ext.Array((20000, 2000), int32, chunks=(1000, 100), nbytes=152.6M, cbytes=7.6M, cratio=20.2, cname=blosclz, clevel=5, shuffle=1)

```

## Tuning

``zarr`` is designed for use in parallel computations working chunk-wise 
over data. Try it with [dask.array](http://dask.pydata.org/en/latest/array.html).

``zarr`` is optimised for accessing and storing data in contiguous slices, 
of the same size or larger than chunks. It is not and will never be 
optimised for single item access. 

Chunks sizes >= 1M are generally good. Optimal chunk shape will depend on 
the correlation structure in your data.

## Acknowledgments

``zarr`` uses [c-blosc](https://github.com/Blosc/c-blosc) internally for 
compression and decompression and borrows code heavily from 
[bcolz](http://bcolz.blosc.org/).
