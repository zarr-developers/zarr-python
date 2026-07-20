# Provenance of the ndsel conformance corpus

The JSON fixtures in this directory (`point.json`, `box.json`, `slice.json`,
`points.json`, `transform.json`, `errors.json`) and `README.md` are **vendored,
unmodified**, from the ndsel reference repository.

- **Source:** <https://github.com/d-v-b/ndsel>
- **Branch:** `fix/slice-origin-trunc`
- **Commit:** `c132b4c1caa3205830ce35a42502363171f650a7`
- **Path in source:** `conformance/`

**Do not edit these files.** They are vendored as-is so that
`zarr_transforms`' ndsel message layer can be checked against the same
language-agnostic corpus every other ndsel implementation runs. To update the
corpus, re-vendor from a newer ndsel commit and update the commit SHA above.

The `fix/slice-origin-trunc` branch (ndsel PR #1) corrects the `slice`
desugaring origin from `floor(a/s)` to `trunc(a/s)` (rounding toward zero),
which matches `zarr_transforms`' existing `_trunc_div` semantics.
