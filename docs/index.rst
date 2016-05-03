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


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

