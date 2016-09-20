.. zarr documentation master file, created by
   sphinx-quickstart on Mon May  2 21:40:09 2016.

Zarr
====

Zarr is a Python package providing an implementation of chunked,
compressed, N-dimensional arrays.

Highlights
----------

* Create N-dimensional arrays with any NumPy dtype.
* Chunk arrays along any dimension.
* Compress chunks using the fast Blosc_ meta-compressor or alternatively using zlib, BZ2 or LZMA.
* Store arrays in memory, on disk, inside a Zip file, on S3, ...
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
* Organize arrays into hierarchies via groups.
* Use filters to preprocess data and improve compression.

Status
------

Zarr is still in an early phase of development. Feedback and bug
reports are very welcome, please get in touch via the `GitHub issue
tracker <https://github.com/alimanfoo/zarr/issues>`_.

Installation
------------

Zarr depends on NumPy. It is generally best to `install NumPy
<http://docs.scipy.org/doc/numpy/user/install.html>`_ first using
whatever method is most appropriate for you operating system and
Python distribution.

Install Zarr from PyPI::

    $ pip install zarr

Alternatively, install Zarr via conda::

    $ conda install -c conda-forge zarr
  
Zarr includes a C extension providing integration with the Blosc_
library. Installing via conda will install a pre-compiled binary distribution. 
However, if you have a newer CPU that supports the AVX2 instruction set (e.g., 
Intel Haswell, Broadwell or Skylake) then installing via pip is preferable, 
because this will compile the Blosc library from source with optimisations 
for AVX2.
  
To work with Zarr source code in development, install from GitHub::

    $ git clone --recursive https://github.com/alimanfoo/zarr.git
    $ cd zarr
    $ python setup.py install

To verify that Zarr has been fully installed (including the Blosc
extension) run the test suite::

    $ pip install nose
    $ python -m nose -v zarr
  
Contents
--------

.. toctree::
    :maxdepth: 2

    tutorial
    api
    spec
    release

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
