.. zarr documentation master file, created by
   sphinx-quickstart on Mon May  2 21:40:09 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Zarr
====

Zarr is a Python package providing a minimal implementation of
chunked, compressed, N-dimensional arrays.

* Source: https://github.com/alimanfoo/zarr
* Documentation: http://zarr.readthedocs.io/
* Download: https://pypi.python.org/pypi/zarr
* Release notes: https://github.com/alimanfoo/zarr/releases

Motivation
----------

Zarr is motivated by the desire to work interactively with
multi-dimensional scientific datasets too large to fit into memory on
commodity desktop or laptop computers. Interactive data analysis
requires fast array storage, because an interactive session may
involve creation and manipulation of many intermediate data
structures. Faster storage provides more freedom to explore a rich and
complex dataset in a variety of different ways. The Blosc compression
library provides extremely fast multi-threaded compression and
decompression, and so a primary motivation for Zarr was to bring
together Blosc with multi-dimensional arrays in a convenient way.

A second motivation is to provide array storage that is convenient and
well-suited to use in parallel computations. This means supporting
concurrent data access from multiple threads or processes, without
unnecessary locking or exclusion, to maximise the possibility for work
to be carried out in parallel.

Status
------

Zarr is still in an early, experimental phase of development. Feedback and bug
reports are very welcome, please get in touch via the `GitHub issue
tracker <https://github.com/alimanfoo/zarr/issues>`_.

Installation
------------

N.B., currently Zarr requires that Numpy and Cython are both pre-installed.

Install from PyPI::

    $ pip install zarr

Install from GitHub::

    $ git clone --recursive https://github.com/alimanfoo/zarr.git
    $ cd zarr
    $ python setup.py install

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

Zarr is inspired by and borrows code from `bcolz <http://bcolz.blosc.org/>`_.

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
