<div align="center">
  <img src="https://raw.githubusercontent.com/zarr-developers/community/main/logos/logo2.png"><br>
</div>

# Zarr

[![Latest Release](https://badge.fury.io/py/zarr.svg)](https://pypi.org/project/zarr/)
[![CondaForge](https://anaconda.org/conda-forge/zarr/badges/version.svg)](https://anaconda.org/anaconda/zarr/)
[![Package Status](https://img.shields.io/pypi/status/zarr.svg)](https://pypi.org/project/zarr/)
[![License](https://img.shields.io/pypi/l/zarr.svg)](https://github.com/zarr-developers/zarr-python/blob/main/LICENSE.txt)
[![Coverage](https://codecov.io/gh/zarr-developers/zarr-python/branch/main/graph/badge.svg)](https://app.codecov.io/gh/zarr-developers/zarr-python)
[![Downloads](https://static.pepy.tech/badge/zarr)](https://zarr.readthedocs.io/en/stable/)
[![Developer Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr-Python)
[![Citation](https://zenodo.org/badge/DOI/10.5281/zenodo.3773450.svg)](https://doi.org/10.5281/zenodo.3773450)

## What is it?

The `zarr` package is a Python implementation of the [Zarr storage format](https://zarr.dev/). `zarr` delivers compressed, chunked, N-dimensional arrays that work well for parallel computing and object storage. See the [documentation](https://zarr.readthedocs.io/en/stable/) for more information.

## Main Features

- [**Create**](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#creating-an-array) N-dimensional arrays with NumPy-compatible `dtype`s.
- [**Chunk arrays**](https://zarr.readthedocs.io/en/stable/user-guide/performance/#chunk-optimizations) along any dimension.
- [**Encode**](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#compressors) chunks using a variety of useful encodings (e.g., compression).
- [**Store arrays**](https://zarr.readthedocs.io/en/stable/user-guide/storage/) in memory, on disk, inside a zip file, on S3, etc...
- [**Read**](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#reading-and-writing-data) an array [**concurrently**](https://zarr.readthedocs.io/en/stable/user-guide/performance/#parallel-computing-and-synchronization) from multiple threads or processes.
- [**Write**](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#reading-and-writing-data) to an array concurrently from multiple threads or processes.
- Organize arrays into hierarchies via [**groups**](https://zarr.readthedocs.io/en/stable/quick-start/#hierarchical-groups).

## Where to get it

Zarr can be installed from PyPI using `pip`:

```bash
pip install zarr
```

or via `conda`:

```bash
conda install -c conda-forge zarr
```

For more details, including how to install from source, see the [installation documentation](https://zarr.readthedocs.io/en/stable/#installation).

## Ecosystem

This repository contains several packages:

- [`zarr`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata): The Python implementation of the Zarr storage format.
- [`zarr-metadata`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata): Tools for Zarr metadata.
- [`zarr-indexing`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing): Tools for lazily indexing chunked arrays.
- [`zarr-http-server`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-http-server): An HTTP server implementation targeting Zarr data.
