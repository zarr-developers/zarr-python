# Provenance of the ndsel conformance corpus

The JSON fixtures in this directory (`point.json`, `box.json`, `slice.json`,
`points.json`, `transform.json`, `errors.json`) and `README.md` are **vendored,
unmodified**, from the ndsel reference repository.

- **Source:** <https://github.com/zarr-developers/ndsel>
- **Branch:** `main` (merge of d-v-b/ndsel#2, negative `step`)
- **Commit:** `92d6a32df0cd1ac47d548f14f42909a95997cf19` (previously vendored:
  `c59bc556c`, itself byte-identical to `c132b4c1caa3205830ce35a42502363171f650a7`)
- **Path in source:** `conformance/`

**Do not edit these files.** They are vendored as-is so that
`zarr_indexing`' ndsel message layer can be checked against the same
language-agnostic corpus every other ndsel implementation runs. To update the
corpus, re-vendor from a newer ndsel commit and update the commit SHA above.

ndsel PR #1 (merged) corrected the `slice` desugaring origin from
`floor(a/s)` to `trunc(a/s)` (rounding toward zero), which matches
`zarr_indexing`' existing `_trunc_div` semantics.

ndsel PR #2 (merged) specified negative `step`, which changed two fixtures:

- `slice.json` gained the negative-step cases (full reverse, `|s| > 1`
  non-divisible, negative-coordinate intervals, empty-at-any-coordinate).
- `errors.json` retired `error/negative-step` — the reason code
  `negative_step_unsupported` is retired with it — and replaced it with three
  `bounds_out_of_order` fixtures pinning that a reversed interval is an error
  for either sign of the step, rather than being clamped to empty.

Re-vendoring those two files and teaching `zarr_indexing.messages` the new
desugaring are one change: the corpus is the definition of correct here, so it
lands in the same commit as the code that satisfies it.
