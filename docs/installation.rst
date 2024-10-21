Installation
============

Zarr depends on NumPy. It is generally best to `install NumPy
<https://numpy.org/doc/stable/user/install.html>`_ first using whatever method is most
appropriate for your operating system and Python distribution. Other dependencies should be
installed automatically if using one of the installation methods below.

Note: Zarr has endorsed `Scientific-Python SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ and now follows the version support window as outlined below: 

- Python: 36 months after initial release
- Core package dependencies (e.g. NumPy): 24 months after initial release

Install Zarr from PyPI::

    $ pip install zarr

Alternatively, install Zarr via conda::

    $ conda install -c conda-forge zarr

To install the latest development version of Zarr, you can use pip with the
latest GitHub main::

    $ pip install git+https://github.com/zarr-developers/zarr-python.git

To work with Zarr source code in development, see `Contributing <contributing.html>`_.