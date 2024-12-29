.. _zarr_docs_mainpage:

***********
Zarr-Python
***********

.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart
    about
    user-guide/index
    api/index
    developers/index
    developers/release

**Version**: |version|

Zarr is a storage format for chunked, compressed, N-dimensional arrays based on an open-source specification. Highlights include:

* Create N-dimensional arrays with any NumPy dtype.
* Chunk arrays along any dimension.
* Compress and/or filter chunks using any NumCodecs_ codec.
* Store arrays in memory, on disk, inside a Zip file, on S3, ...
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
* Organize arrays into hierarchies via groups.

.. grid:: 2

    .. grid-item-card::
        :img-top: _static/index_getting_started.svg

        Quick Start
        ^^^^^^^^^^^

        New to Zarr? Check out the quick start guide. It contains a brief
        introduction to Zarr's main concepts and links to additional tutorials.

        +++

        .. button-ref:: quickstart
            :expand:
            :color: dark
            :click-parent:

            To the Quick Start

    .. grid-item-card::
        :img-top: _static/index_user_guide.svg

        Guide
        ^^^^^

        The user guide provides a detailed guide for how to use Zarr-Python.

        +++

        .. button-ref:: user-guide
            :ref-type: ref
            :expand:
            :color: dark
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: _static/index_api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in Zarr. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api/index
            :expand:
            :color: dark
            :click-parent:

            To the API reference guide

    .. grid-item-card::
        :img-top: _static/index_contribute.svg

        Developer's Guide
        ^^^^^^^^^^^^^^^^^

        Want to contribute to Zarr? We welcome contributions in the form of bug reports,
        bug fixes, documentation, enhancement proposals and more. The contributing
        guidelines will guide you through the process of improving Zarr.

        +++

        .. button-ref:: developers/index
            :expand:
            :color: dark
            :click-parent:

            To the developers's guide

**Useful links**:
`Source Repository <https://github.com/zarr-developers/zarr-python>`_ |
`Issue Tracker <https://github.com/zarr-developers/zarr-python/issues>`_ |
`Zulip Chat <https://ossci.zulipchat.com/>`_ |
`Zarr specifications <https://zarr-specs.readthedocs.io>`_

**Download documentation**: `PDF/Zipped HTML <https://readthedocs.org/projects/zarr/downloads/>`_

.. _NumCodecs: https://numcodecs.readthedocs.io
