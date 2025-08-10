# Installation

## Required dependencies

Required dependencies include:

- [Python](https://docs.python.org/3/) (3.11 or later)
- [packaging](https://packaging.pypa.io) (22.0 or later)
- [numpy](https://numpy.org) (1.26 or later)
- [numcodecs[crc32c]](https://numcodecs.readthedocs.io) (0.14 or later)
- [typing_extensions](https://typing-extensions.readthedocs.io) (4.9 or later)
- [donfig](https://donfig.readthedocs.io) (0.8 or later)

## pip

Zarr is available on [PyPI](https://pypi.org/project/zarr/). Install it using `pip`:

```console
$ pip install zarr
```

There are a number of optional dependency groups you can install for extra functionality.
These can be installed using `pip install "zarr[<extra>]"`, e.g. `pip install "zarr[gpu]"`

- `gpu`: support for GPUs
- `remote`: support for reading/writing to remote data stores

Additional optional dependencies include `rich`, `universal_pathlib`. These must be installed separately.

## conda

Zarr is also published to [conda-forge](https://conda-forge.org). Install it using `conda`:

```console
$ conda install -c conda-forge zarr
```

Conda does not support optional dependencies, so you will have to manually install any packages
needed to enable extra functionality.

## Dependency support

Zarr has endorsed [Scientific-Python SPEC 0](https://scientific-python.org/specs/spec-0000/) and now follows the version support window as outlined below:

- Python: 36 months after initial release
- Core package dependencies (e.g. NumPy): 24 months after initial release

## Development

To install the latest development version of Zarr, see the contributing guide.
