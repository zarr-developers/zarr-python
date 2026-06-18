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

- [ ] Check [SPEC 0](https://scientific-python.org/specs/spec-0000/#support-window) to see if the minimum supported version of Python or NumPy needs bumping.
- [ ] Verify that the latest CI workflows on `main` are passing: [Tests](https://github.com/zarr-developers/zarr-python/actions/workflows/test.yml), [GPU Tests](https://github.com/zarr-developers/zarr-python/actions/workflows/gpu_test.yml), [Hypothesis](https://github.com/zarr-developers/zarr-python/actions/workflows/hypothesis.yaml), [Docs](https://github.com/zarr-developers/zarr-python/actions/workflows/docs.yml), [Lint](https://github.com/zarr-developers/zarr-python/actions/workflows/lint.yml), [Wheels](https://github.com/zarr-developers/zarr-python/actions/workflows/releases.yml).
- [ ] Run the ["Prepare release" workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/prepare_release.yml) with the target version. This will build the changelog and open a release PR with the `run-downstream` label.
- [ ] Verify that the [downstream tests](https://github.com/zarr-developers/zarr-python/actions/workflows/downstream.yml) (triggered automatically by the `run-downstream` label) pass on the release PR.
- [ ] Review the release PR and verify the changelog in `docs/release-notes.md` looks correct.
- [ ] Merge the release PR.

**Release**:

- [ ] [Draft a new GitHub Release](https://github.com/zarr-developers/zarr-python/releases/new) with tag `vX.Y.Z` targeting `main`. Use "Generate release notes" for the description.
- [ ] Verify the release is published on [PyPI](https://pypi.org/project/zarr/) and [ReadTheDocs](https://zarr.readthedocs.io/en/stable/).

**After release**:

- [ ] Review and merge the pull request on the conda-forge [zarr-feedstock](https://github.com/conda-forge/zarr-feedstock) that will be automatically generated.

---

- [ ] Party :tada:

---

<details>
<summary><strong>Releasing from a branch other than main</strong></summary>

In rare cases (e.g. patch releases for an older minor version), you may need to release from a dedicated release branch (e.g. `3.1.x`):

- Create the release branch from the appropriate tag if it doesn't already exist.
- Cherry-pick or backport the necessary commits onto the branch.
- Run `towncrier build --version x.y.z` and commit the result to the release branch instead of `main`.
- When drafting the GitHub Release, set the target to the release branch instead of `main`.
- After the release, ensure any relevant changelog updates are also reflected on `main`.

</details>
