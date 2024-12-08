Installation
============

pip
---

.. code-block:: console

    $ pip install zarr

There are a number of optional dependency groups you can install for extra functionality.
These can be installed using ``pip install "zarr[<extra>]"``, e.g. ``pip install "zarr[gpu]"``

- ``gpu``: support for GPUs
- ``fsspec``: support for reading/writing to remote data stores
- ``tree``: support for pretty printing of directory trees

conda
-----

.. code-block:: console

    $ conda install -c conda-forge zarr

Conda does not support optional dependencies, so you will have to manually install any packages
needed to enable extra functionality.

Dependency support
------------------
Zarr has endorsed `Scientific-Python SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ and now follows the version support window as outlined below:

- Python: 36 months after initial release
- Core package dependencies (e.g. NumPy): 24 months after initial release

Development
-----------
To install the latest development version of Zarr, see `the contributing guide <contributing.html>`_.
