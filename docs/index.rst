.. zarr documentation master file, created by
   sphinx-quickstart on Mon May  2 21:40:09 2016.

Zarr
====

Zarr is a Python package providing a minimal implementation of
chunked, compressed, N-dimensional arrays.

* Source: https://github.com/alimanfoo/zarr
* Documentation: http://zarr.readthedocs.io/
* Download: https://pypi.python.org/pypi/zarr
* Release notes: https://github.com/alimanfoo/zarr/releases

Highlights
----------

* Create N-dimensional arrays with any NumPy dtype.
* Chunk arrays along any dimension.
* Compress chunks using the fast Blosc_ meta-compressor or alternatively using zlib, BZ2 or LZMA.
* Store arrays in memory, on disk, inside a Zip file, on S3, ... pretty much anywhere you like.
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
    
Status
------

Zarr is still in an early, experimental phase of development. Feedback and bug
reports are very welcome, please get in touch via the `GitHub issue
tracker <https://github.com/alimanfoo/zarr/issues>`_.

Installation
------------

N.B., installation of Zarr requires that Numpy is already installed, e.g.::

    $ pip install numpy

Install Zarr from PyPI::

    $ pip install zarr

Install Zarr from GitHub::

    $ git clone --recursive https://github.com/alimanfoo/zarr.git
    $ cd zarr
    $ python setup.py install

N.B., on posix systems Zarr will attempt to build the Blosc C extension.
Zarr will fall back to a pure Python installation on other platforms or if
the C extension fails to build for any reason, which means that 'blosc'
compression will not be available.

Contents
--------

.. toctree::
    :maxdepth: 2

    tutorial
    api
    spec

Acknowledgments
---------------

Zarr bundles the `c-blosc <https://github.com/Blosc/c-blosc>`_
library and uses it as the default compressor.

Zarr is inspired by `HDF5 <https://www.hdfgroup.org/HDF5/>`_, `h5py
<http://www.h5py.org/>`_ and `bcolz <http://bcolz.blosc.org/>`_.

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Blosc: http://www.blosc.org/
