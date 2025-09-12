Installation
============

Required dependencies
---------------------

Required dependencies include:

- `Python <https://docs.python.org/3/>`_ (3.11 or later)
- `packaging <https://packaging.pypa.io>`_ (22.0 or later)
- `numpy <https://numpy.org>`_ (1.26 or later)
- `numcodecs[crc32c] <https://numcodecs.readthedocs.io>`_ (0.14 or later)
- `typing_extensions <https://typing-extensions.readthedocs.io>`_ (4.9 or later)
- `donfig <https://donfig.readthedocs.io>`_ (0.8 or later)

pip
---

Zarr is available on `PyPI <https://pypi.org/project/zarr/>`_. Install it using ``pip``:

.. code-block:: console

    $ pip install zarr

There are a number of optional dependency groups you can install for extra functionality.
These can be installed using ``pip install "zarr[<extra>]"``, e.g. ``pip install "zarr[gpu]"``

- ``gpu``: support for GPUs
- ``remote``: support for reading/writing to remote data stores

Additional optional dependencies include ``rich``, ``universal_pathlib``. These must be installed separately.

conda
-----

Zarr is also published to `conda-forge <https://conda-forge.org>`_. Install it using ``conda``:

.. code-block:: console

    $ conda install -c conda-forge zarr

Conda does not support optional dependencies, so you will have to manually install any packages
needed to enable extra functionality.

Nightly wheels
--------------

Development wheels are built nightly and published to the `scientific-python-nightly-wheels <https://anaconda.org/scientific-python-nightly-wheels>`_ index. To install the latest nightly build:

.. code-block:: console

    $ pip install --pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple zarr

Note that nightly wheels may be unstable and are intended for testing purposes.

Dependency support
------------------
Zarr has endorsed `Scientific-Python SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ and now follows the version support window as outlined below:

- Python: 36 months after initial release
- Core package dependencies (e.g. NumPy): 24 months after initial release

Development
-----------
To install the latest development version of Zarr, see the :ref:`contributing guide <dev-guide-contributing>`.
