# Zarr-Python

**Useful links**:
[Source Repository](https://github.com/zarr-developers/zarr-python) |
[Issue Tracker](https://github.com/zarr-developers/zarr-python/issues) |
[Developer Chat](https://ossci.zulipchat.com/) |
[Zarr specifications](https://zarr-specs.readthedocs.io)


Zarr is a powerful library for storage of n-dimensional arrays, supporting chunking,
compression, and various backends, making it a versatile choice for scientific and
large-scale data.

Zarr-Python is a Python library for reading and writing Zarr groups and arrays. Highlights include:

* Specification support for both Zarr format 2 and 3.
* Create and read from N-dimensional arrays using NumPy-like semantics.
* Flexible storage enables reading and writing from local, cloud and in-memory stores.
* High performance: Enables fast I/O with support for asynchronous I/O and multi-threading.
* Extensible: Customizable with user-defined codecs and stores.

## Installation

Zarr requires Python 3.12 or higher. You can install it via `pip`:

```bash
pip install zarr
```

or `conda`:

```bash
conda install --channel conda-forge zarr
```

## Navigating the documentation

<div class="grid cards" markdown>

-   [:material-clock-fast:{ .lg .middle } __Quick start__](quick-start.md)

    ---

    New to Zarr? Check out the quick start guide. It contains a brief
    introduction to Zarr's main concepts and links to additional tutorials.


-   [:material-book-open:{ .lg .middle } __User guide__](user-guide/installation.md)

    ---

    A detailed guide for how to use Zarr-Python.


-   [:material-api:{ .lg .middle } __API Reference__](api/zarr/open.md)

    ---

    The reference guide contains a detailed description of the functions, modules,
    and objects included in Zarr. The reference describes how the methods work and
    which parameters can be used. It assumes that you have an understanding of the
    key concepts.


-   [:material-account-group:{ .lg .middle } __Contributor's Guide__](contributing.md)

    ---

    Want to contribute to Zarr? We welcome contributions in the form of bug reports,
    bug fixes, documentation, enhancement proposals and more. The contributing guidelines
    will guide you through the process of improving Zarr.

</div>


## Project Status

More information about the Zarr format can be found on the [main website](https://zarr.dev).

If you are using Zarr-Python, we would [love to hear about it](https://github.com/zarr-developers/community/issues/19).

### Funding and Support
The project is fiscally sponsored by [NumFOCUS](https://numfocus.org/), a US
501(c)(3) public charity, and development has been supported by the
[MRC Centre for Genomics and Global Health](https://github.com/cggh/)
and the [Chan Zuckerberg Initiative](https://chanzuckerberg.com/).

[Donate to Zarr](https://numfocus.org/donate-to-zarr) to support the project!
