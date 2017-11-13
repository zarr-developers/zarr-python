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
* Compress and/or filter chunks using any numcodecs_ codec.
* Store arrays in memory, on disk, inside a Zip file, on S3, ...
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
* Organize arrays into hierarchies via groups.

Status
------

Zarr is still a young project. Feedback and bug reports are very welcome, please get in touch via
the `GitHub issue tracker <https://github.com/alimanfoo/zarr/issues>`_.

Installation
------------

Zarr depends on NumPy. It is generally best to `install NumPy
<http://docs.scipy.org/doc/numpy/user/install.html>`_ first using whatever method is most
appropriate for you operating system and Python distribution. Other dependencies should be
installed automatically if using one of the installation methods below.

Install Zarr from PyPI::

    $ pip install zarr

Alternatively, install Zarr via conda::

    $ conda install -c conda-forge zarr

To work with Zarr source code in development, install from GitHub::

    $ git clone --recursive https://github.com/alimanfoo/zarr.git
    $ cd zarr
    $ python setup.py install

To verify that Zarr has been fully installed, run the test suite::

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

The following people have contributed to the development of Zarr, by contributing code and/or
providing ideas, feedback and advice:

* `Francesc Alted <https://github.com/FrancescAlted>`_
* `Stephan Hoyer <https://github.com/shoyer>`_
* `John Kirkham <https://github.com/jakirkham>`_
* `Alistair Miles <https://github.com/alimanfoo>`_
* `Matthew Rocklin <https://github.com/mrocklin>`_
* `Vincent Schut <https://github.com/vincentschut>`_

Zarr is inspired by `HDF5 <https://www.hdfgroup.org/HDF5/>`_, `h5py
<http://www.h5py.org/>`_ and `bcolz <http://bcolz.blosc.org/>`_.

Development of Zarr is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _numcodecs: http://numcodecs.readthedocs.io/
