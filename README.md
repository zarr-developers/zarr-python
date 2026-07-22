<div align="center">
  <img src="https://raw.githubusercontent.com/zarr-developers/community/main/logos/logo2.png"><br>
</div>

# Zarr

[![Latest Release](https://badge.fury.io/py/zarr.svg)](https://pypi.org/project/zarr/)
[![CondaForge](https://anaconda.org/conda-forge/zarr/badges/version.svg)](https://anaconda.org/anaconda/zarr/)
[![Package Status](https://img.shields.io/pypi/status/zarr.svg)](https://pypi.org/project/zarr/)
[![License](https://img.shields.io/pypi/l/zarr.svg)](https://github.com/zarr-developers/zarr-python/blob/main/LICENSE.txt)
[![Coverage](https://codecov.io/gh/zarr-developers/zarr-python/branch/main/graph/badge.svg)](https://codecov.io/gh/zarr-developers/zarr-python)
[![Downloads](https://pepy.tech/badge/zarr)](https://zarr.readthedocs.io)
[![Developer Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr-Python)
[![Citation](https://zenodo.org/badge/DOI/10.5281/zenodo.3773450.svg)](https://doi.org/10.5281/zenodo.3773450)

## What is it?

Zarr is a Python package providing an implementation of compressed, chunked, N-dimensional arrays, designed for use in parallel computing. See the [documentation](https://zarr.readthedocs.io) for more information.

## Main Features

- [**Create**](https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#creating-an-array) N-dimensional arrays with any NumPy `dtype`.
- [**Chunk arrays**](https://zarr.readthedocs.io/en/stable/user-guide/performance.html#chunk-optimizations) along any dimension.
- [**Compress**](https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#compressors) and/or filter chunks using any NumCodecs codec.
- [**Store arrays**](https://zarr.readthedocs.io/en/stable/user-guide/storage.html) in memory, on disk, inside a zip file, on S3, etc...
- [**Read**](https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#reading-and-writing-data) an array [**concurrently**](https://zarr.readthedocs.io/en/stable/user-guide/performance.html#parallel-computing-and-synchronization) from multiple threads or processes.
- [**Write**](https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#reading-and-writing-data) to an array concurrently from multiple threads or processes.
- Organize arrays into hierarchies via [**groups**](https://zarr.readthedocs.io/en/stable/quickstart.html#hierarchical-groups).

## Where to get it

Zarr can be installed from PyPI using `pip`:

```bash
pip install zarr
```

or via `conda`:

```bash
conda install -c conda-forge zarr
```

For more details, including how to install from source, see the [installation documentation](https://zarr.readthedocs.io/en/stable/index.html#installation).
