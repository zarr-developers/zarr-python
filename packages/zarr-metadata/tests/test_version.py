"""Verify that `zarr_metadata.__version__` matches the installed
distribution metadata, which in turn comes from `pyproject.toml`.

This catches the easy mistake of bumping the version in one place and
forgetting the other.
"""

from __future__ import annotations

from importlib.metadata import version

import zarr_metadata


def test_version_matches_distribution_metadata() -> None:
    assert zarr_metadata.__version__ == version("zarr-metadata")
