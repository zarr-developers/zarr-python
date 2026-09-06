# Contributing to zarr-indexing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, same invocation as CI
just docs-check  # strict build of the docs site
just check       # all of the above
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from anywhere in the repository as
`just packages/zarr-indexing/<recipe>`.

The test recipe runs against the workspace-root environment, because the
chunk-resolution tests exercise this package against `zarr`'s chunk grids and
`zarr` is deliberately not a dependency of this package.

## License

MIT

The package lives at `packages/zarr-indexing` inside the
[zarr-python](https://github.com/zarr-developers/zarr-python) repository.
