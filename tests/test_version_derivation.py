"""Sanity check that ``zarr.__version__`` looks like a v3-or-newer release.

Background: zarr-python derives its version from ``git describe`` via
hatch-vcs. The repo also publishes a separate ``zarr-metadata`` subpackage
that uses ``zarr_metadata-v*`` tags. Without the ``--match v*`` filter in
``[tool.hatch] version.raw-options.git_describe_command``, ``git describe``
walks back to those subpackage tags and reports a version like ``0.2.0`` for
a from-source build of zarr-python itself — see
https://github.com/zarr-developers/zarr-python/pull/3994.

This test catches that class of regression: anything that makes zarr-python
report a version lower than the v3 release line. When 4.0 is released,
bump the floor; that's a deliberate, one-line edit at a planned boundary.
"""

from __future__ import annotations

from packaging.version import Version

import zarr


def test_version_is_v3_or_newer() -> None:
    # Use packaging.Version so we transparently handle hatch-vcs dev suffixes
    # like "3.2.2.dev30+gdc5e1825" that appear on any source build past the
    # latest v* tag — Version.major returns 3 for that string.
    parsed = Version(zarr.__version__)
    assert parsed.major >= 3, (
        f"zarr.__version__={zarr.__version__!r} is not on the v3 (or newer) "
        f"release line. If this fires on a from-source build, check that "
        f"[tool.hatch] version.raw-options.git_describe_command in "
        f"pyproject.toml still includes ``--match v*`` so the "
        f"``zarr_metadata-v*`` subpackage tags are excluded. See PR #3994."
    )
