---
name: Zarr-Python release checklist
about: Checklist for a new Zarr-Python release. [For project maintainers only!]
title: Release Zarr-Python vX.Y.Z
labels: release-checklist
assignees: ''

---

**Release**: [v3.x.x](https://github.com/zarr-developers/zarr-python/milestones/?)
**Scheduled Date**: 20YY/MM/DD

**Priority PRs/issues to complete prior to release**

- [ ] Priority pull request #X

**Before release**:

- [ ] Make sure the release branch (e.g., `3.1.x`) is up to date with any backports.
- [ ] Make sure that all pull requests which will be included in the release have been properly documented as changelog files in the [`changes/` directory](https://github.com/zarr-developers/zarr-python/tree/main/changes).
- [ ] Run ``towncrier build --version x.y.z`` to create the changelog, and commit the result to the release branch.
- [ ] Check [SPEC 0](https://scientific-python.org/specs/spec-0000/#support-window) to see if the minimum supported version of Python or NumPy needs bumping.
- [ ] Check to ensure that:
  - [ ] Deprecated workarounds/codes/tests are removed. Run `grep "# TODO" **/*.py` to find all potential TODOs.
  - [ ] All tests pass in the ["Tests" workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/test.yml).
  - [ ] All tests pass in the ["GPU Tests" workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/gpu_test.yml).
  - [ ] All tests pass in the ["Hypothesis" workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/hypothesis.yaml).
  - [ ] Check that downstream libraries work well (maintainers can make executive decisions about whether all checks are required for this release).
    - [ ] numcodecs
    - [ ] Xarray (@jhamman @dcherian @TomNicholas)
        - Zarr's upstream compatibility is tested via the [Upstream Dev CI worklow](https://github.com/pydata/xarray/actions/workflows/upstream-dev-ci.yaml).
        - Click on the most recent workflow and check that the `upstream-dev` job has run and passed. `upstream-dev` is not run on all all workflow runs.
        - Check that the expected version of Zarr-Python was tested using the `Version Info` step of the `upstream-dev` job.
        - If testing on a branch other than `main` is needed, open a PR modifying https://github.com/pydata/xarray/blob/90ee30943aedba66a37856b2332a41264e288c20/ci/install-upstream-wheels.sh#L56 and add the `run-upstream` label.
    - [ ] Titiler.Xarray (@maxrjones)
        - [Modify dependencies](https://github.com/developmentseed/titiler/blob/main/src/titiler/xarray/pyproject.toml) for titiler.xarray.
        - Modify triggers for running [the test workflow](https://github.com/developmentseed/titiler/blob/61549f2de07b20cca8fb991cfcdc89b23e18ad05/.github/workflows/ci.yml#L5-L7).
        - Push the branch to the repository and check for the actions for any failures.

**Release**:

- [ ] Go to https://github.com/zarr-developers/zarr-python/releases.
  - [ ] Click "Draft a new release".
  - [ ] Choose a version number prefixed with a `v` (e.g. `v0.0.0`). For pre-releases, include the appropriate suffix (e.g. `v0.0.0a1` or `v0.0.0rc2`).
  - [ ] Set the target branch to the release branch (e.g., `3.1.x`)
  - [ ] Set the description of the release to: `See release notes https://zarr.readthedocs.io/en/stable/release-notes.html#release-0-0-0`, replacing the correct version numbers. For pre-release versions, the URL should omit the pre-release suffix, e.g. "a1" or "rc1".
  - [ ] Click on "Generate release notes" to auto-fill the description.
  - [ ] Make a release by clicking the 'Publish Release' button, this will automatically create a tag too.
- [ ] Verify that release workflows succeeded.
  - [ ] The latest version is correct on [PyPI](https://pypi.org/project/zarr/).
  - [ ] The stable version is correct on [ReadTheDocs](https://zarr.readthedocs.io/en/stable/).

**After release**:

- [ ] Review and merge the pull request on the conda-forge [zarr-feedstock](https://github.com/conda-forge/zarr-feedstock) that will be automatically generated.

---

- [ ] Party :tada:
