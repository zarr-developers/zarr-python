Writing a changelog entry for `zarr-indexing`
-----------------------------------------------

Fragments in **this** directory are release notes for the `zarr-indexing`
package only — kept separate from the parent zarr-python `changes/`
directory so a PR touching only `packages/zarr-indexing/` produces a
release note for this package only.

Please put a new file in this directory named `xxxx.<type>.md`, where

- `xxxx` is the pull request number associated with this entry
- `<type>` is one of:
  - feature
  - bugfix
  - doc
  - removal
  - misc

Inside the file, please write a short description of what you have
changed, and how it impacts users of `zarr-indexing`.

A `zarr-indexing` release runs `towncrier build` in `packages/zarr-indexing/`,
which consumes the fragments here and updates `CHANGELOG.md`. Fragments
that describe parent zarr-python changes (not the transforms package)
belong in the top-level `changes/` directory, not here.
