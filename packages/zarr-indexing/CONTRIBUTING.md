# Contributing to zarr-indexing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, same invocation as CI
just docs-check  # strict build of the docs site
just check       # the checks above plus TensorStore parity
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from the repository root as
`just packages/zarr-indexing/<recipe>`.

The test recipe layers this package into the repository-root environment.
Chunk-resolution tests use this package’s own grids; the Dask example also
uses `zarr`, which is not a dependency of the base indexing package.

## License

MIT

The package lives at `packages/zarr-indexing` inside the
[zarr-python](https://github.com/zarr-developers/zarr-python) repository.
