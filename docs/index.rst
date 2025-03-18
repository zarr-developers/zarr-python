.. _zarr_docs_mainpage:

***********
Zarr-Python
***********

.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart
    user-guide/index
    API reference <api/zarr/index>
    release-notes
    developers/index
    about

**Version**: |version|

**Useful links**:
`Source Repository <https://github.com/zarr-developers/zarr-python>`_ |
`Issue Tracker <https://github.com/zarr-developers/zarr-python/issues>`_ |
`Developer Chat <https://ossci.zulipchat.com/>`_ |
`Zarr specifications <https://zarr-specs.readthedocs.io>`_

Zarr-Python is a Python library for reading and writing Zarr groups and arrays. Highlights include:

* Specification support for both Zarr format 2 and 3.
* Create and read from N-dimensional arrays using NumPy-like semantics.
* Flexible storage enables reading and writing from local, cloud and in-memory stores.
* High performance: Enables fast I/O with support for asynchronous I/O and multi-threading.
* Extensible: Customizable with user-defined codecs and stores.

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

        A detailed guide for how to use Zarr-Python.

        +++

        .. button-ref:: user-guide/index
            :expand:
            :color: dark
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: _static/index_api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in Zarr. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api/zarr/index
            :expand:
            :color: dark
            :click-parent:

            To the API reference

    .. grid-item-card::
        :img-top: _static/index_contribute.svg

        Contributor's Guide
        ^^^^^^^^^^^^^^^^^^^

        Want to contribute to Zarr? We welcome contributions in the form of bug reports,
        bug fixes, documentation, enhancement proposals and more. The contributing guidelines
        will guide you through the process of improving Zarr.

        +++

        .. button-ref:: developers/contributing
            :expand:
            :color: dark
            :click-parent:

            To the contributor's guide


**Download documentation**: `PDF/Zipped HTML <https://readthedocs.org/projects/zarr/downloads/>`_

.. _NumCodecs: https://numcodecs.readthedocs.io
