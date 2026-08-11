# Related Projects

Alongside `zarr` itself, the
[zarr-python repository](https://github.com/zarr-developers/zarr-python) hosts a
small number of companion packages. Each one is developed in the same repository
but versioned, released, and documented independently, so you can depend on it
without taking on `zarr` as a dependency.

<div class="grid cards" markdown>

- [:material-code-json:{ .lg .middle } __zarr-metadata__](https://zarr.readthedocs.io/projects/zarr-metadata/)

    ---

    Spec-defined metadata types, models, and validators for Zarr v2 and v3, with
    minimal dependencies. Useful if your software reads or writes Zarr metadata
    documents but does not need a full Zarr implementation.

    ```bash
    pip install zarr-metadata
    ```

- [:material-vector-polyline:{ .lg .middle } __zarr-indexing__](https://zarr.readthedocs.io/projects/zarr-indexing/)

    ---

    Composable, lazy coordinate transforms for Zarr array indexing. Makes the
    mapping from requested coordinates to stored coordinates a first-class,
    composable value, and resolves which chunks a selection touches.

    ```bash
    pip install zarr-indexing
    ```

</div>

## Elsewhere in the Zarr ecosystem

- [Zarr specifications](https://zarr-specs.readthedocs.io) — the format
  specifications that `zarr-python` implements.
- [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/) — the
  registry of specified extensions to the Zarr v3 core format.
- [zarr.dev](https://zarr.dev) — the Zarr project website, including
  implementations in other languages.
