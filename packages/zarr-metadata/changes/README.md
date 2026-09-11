Writing a changelog entry for `zarr-metadata`
---------------------------------------------

Fragments in **this** directory are released notes for the `zarr-metadata`
package only — kept separate from the parent zarr-python `changes/`
directory so a PR touching only `packages/zarr-metadata/` produces a
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
changed, and how it impacts users of `zarr-metadata`.

A `zarr-metadata` release runs `towncrier build` in `packages/zarr-metadata/`,
which consumes the fragments here and updates `CHANGELOG.md`. Fragments
that describe parent zarr-python changes (not the metadata package)
belong in the top-level `changes/` directory, not here.
