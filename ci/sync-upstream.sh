#!/usr/bin/env bash
# Bleeding-edge environment — the former hatch `tool.hatch.envs.upstream`.
# Latest test deps + nightly numpy + git mains of the core stack. Intentionally
# unlocked (no `--locked`): this job exists to catch upstream breakage early.
set -euo pipefail

uv sync --no-default-groups --group test --group remote-tests --extra remote
uv pip install --prerelease=allow \
  --index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  numpy \
  "packaging @ git+https://github.com/pypa/packaging" \
  "numcodecs @ git+https://github.com/zarr-developers/numcodecs" \
  "s3fs @ git+https://github.com/fsspec/s3fs" \
  "universal_pathlib @ git+https://github.com/fsspec/universal_pathlib" \
  "typing_extensions @ git+https://github.com/python/typing_extensions" \
  "donfig @ git+https://github.com/pytroll/donfig" \
  "obstore @ git+https://github.com/developmentseed/obstore@main#subdirectory=obstore"
