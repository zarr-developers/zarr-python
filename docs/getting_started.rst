Getting Started
===============

Zarr is a format for the storage of chunked, compressed, N-dimensional arrays
inspired by `HDF5 <https://www.hdfgroup.org/HDF5/>`_, `h5py
<https://www.h5py.org/>`_ and `bcolz <https://bcolz.readthedocs.io/>`_.

The project is fiscally sponsored by `NumFOCUS <https://numfocus.org/>`_, a US
501(c)(3) public charity, and development is supported by the
`MRC Centre for Genomics and Global Health <https://www.cggh.org>`_
and the `Chan Zuckerberg Initiative <https://chanzuckerberg.com/>`_.

These documents describe the Zarr Python implementation. More information
about the Zarr format can be found on the `main website <https://zarr.dev>`_.

Highlights
----------

* Create N-dimensional arrays with any NumPy dtype.
* Chunk arrays along any dimension.
* Compress and/or filter chunks using any NumCodecs_ codec.
* Store arrays in memory, on disk, inside a Zip file, on S3, ...
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
* Organize arrays into hierarchies via groups.

Contributing
------------

Feedback and bug reports are very welcome, please get in touch via
the `GitHub issue tracker <https://github.com/zarr-developers/zarr-python/issues>`_. See
:doc:`contributing` for further information about contributing to Zarr.

Projects using Zarr
-------------------

If you are using Zarr, we would `love to hear about it
<https://github.com/zarr-developers/community/issues/19>`_.

.. toctree::
    :caption: Getting Started
    :hidden:

    installation

.. _NumCodecs: https://numcodecs.readthedocs.io/
