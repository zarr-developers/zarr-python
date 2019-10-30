<div align="center">
  <img src="https://raw.githubusercontent.com/zarr-developers/community/master/logos/logo2.png"><br>
</div>

# Zarr

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/zarr/">
    <img src="https://badge.fury.io/py/zarr.svg" alt="latest release" />
    </a>
  </td>
</tr>
  <td></td>
  <td>
    <a href="https://anaconda.org/anaconda/zarr/">
    <img src="https://anaconda.org/conda-forge/zarr/badges/version.svg" alt="latest release" />
    </a>
</td>
</tr>
<tr>
  <td>Package Status</td>
  <td>
		<a href="https://pypi.org/project/zarr/">
		<img src="https://img.shields.io/pypi/status/zarr.svg" alt="status" />
		</a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/zarr-developers/zarr-python/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/zarr.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/zarr-developers/zarr-python">
    <img src="https://travis-ci.org/zarr-developers/zarr-python.svg?branch=master" alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td>
    <a href="https://coveralls.io/github/zarr-developers/zarr-python?branch=master">
    <img src="https://coveralls.io/repos/github/zarr-developers/zarr-python/badge.svg?branch=master" alt="coverage" />
    </a>
  </td>
</tr>
<tr>
  <td>Downloads</td>
  <td>
    <a href="https://pandas.pydata.org">
    <img src="https://pepy.tech/badge/zarr" alt="pypi downloads" />
    </a>
  </td>
</tr>
<tr>
	<td>Gitter</td>
	<td>
		<a href="https://gitter.im/zarr-developers/community">
		<img src="https://badges.gitter.im/zarr-developers/community.svg" />
		</a>
	</td>
</tr>
</table>

## What is it?

Zarr is a Python package providing an implementation of compressed, chunked, N-dimensional arrays, designed for use in parallel computing. See the [documentation](https://zarr.readthedocs.io) for more information.

## Main Features

- [**Create**](https://zarr.readthedocs.io/en/stable/tutorial.html#creating-an-array) N-dimensional arrays with any NumPy `dtype`.
- [**Chunk arrays**](https://zarr.readthedocs.io/en/stable/tutorial.html#chunk-optimizations) along any dimension.
- [**Compress**](https://zarr.readthedocs.io/en/stable/tutorial.html#compressors) and/or filter chunks using any NumCodecs codec.
- [**Store arrays**](https://zarr.readthedocs.io/en/stable/tutorial.html#tutorial-storage) in memory, on disk, inside a zip file, on S3, etc...
- [**Read**](https://zarr.readthedocs.io/en/stable/tutorial.html#reading-and-writing-data) an array [**concurrently**](https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization) from multiple threads or processes.
- Write to an array concurrently from multiple threads or processes.
- Organize arrays into hierarchies via [**groups**](https://zarr.readthedocs.io/en/stable/tutorial.html#groups).

## Where to get it

Zarr depends on NumPy. It is generally best to [install NumPy](https://docs.scipy.org/doc/numpy/user/install.html) first using whatever method is most appropriate for you operating system and Python distribution. Other dependencies should be installed automatically if using one of the installation methods below.

Install Zarr from PyPI:

```bash
pip install zarr
```

Alternatively, install Zarr via `conda`:

```bash
conda install -c conda-forge zarr
```

## Installation from sources

To install the latest development version of Zarr, you can use pip with the latest GitHub master:

```bash
pip install git+https://github.com/zarr-developers/zarr.git
```

or for installing in [development mode](https://pip.pypa.io/en/latest/reference/pip_install.html#editable-installs):

```bash
git clone --recursive https://github.com/zarr-developers/zarr.git
cd zarr
python setup.py install
```
