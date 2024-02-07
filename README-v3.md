# V3 Contributor Guide

A bare-bones guide to contributing to V3.

Developed for the Feb. 2024 Zarr Sprint.

## Clone V3 branch

[Fork](https://github.com/zarr-developers/zarr-python/fork) zarr-python and clone it locally.

```
git clone {your remote}
git remote add upstream https://github.com/zarr-developers/zarr-python
git fetch upstream
git checkout --track upstream/v3
```
## Set the environment variables for V3

There are a couple environment variables required for enabling V3:

```
export ZARR_V3_EXPERIMENTAL_API=1
export ZARR_V3_SHARDING=1
```

## Set up your environment

Zarr uses [hatch](https://hatch.pypa.io/) for its build system.

```
mamba install hatch
```

or

```
pip install hatch
```

Then

```
hatch env create test
```

## Run the Tests

```
hatch run test:run 
```

or

```
hatch -e test shell
pytest -v
```