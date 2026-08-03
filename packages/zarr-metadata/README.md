# zarr-metadata

Python types, models, and validators for Zarr v2 and v3 metadata.

Documentation: <https://zarr-metadata.readthedocs.io/en/stable/>

## What this is

Two layers and an optional integration:

- **Typed JSON shapes**: `TypedDict` definitions and `Literal` aliases for the
  JSON documents specified by the [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
  and [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  specifications, plus types for [`zarr-extensions`](https://github.com/zarr-developers/zarr-extensions/)
  and a few widely-used-but-unspecified entities (e.g. consolidated metadata).
- **Document models** (`zarr_metadata.model`): canonical frozen-dataclass
  models of whole metadata documents, with structural validators, loc-aware
  parsers, and store-key (de)serialization. A document produced by `to_json`
  shares no mutable state with the model that produced it.
- **Optional Pydantic integration** (`zarr_metadata.pydantic`, requires
  Pydantic 2.13 or newer): each model as a Pydantic field type that validates
  raw documents through the same strict parser.

## What this is for

The public `TypedDict` definitions describe the static JSON shape of Zarr
metadata. For strict, loc-aware validation of JSON loaded from disk, use the
model parser:

```python
import json
from zarr_metadata.model import ZarrV3ArrayMetadata

with open("zarr.json", "rb") as f:
    raw = json.load(f)

metadata = ZarrV3ArrayMetadata.from_json(raw)
```

The optional Pydantic integration delegates raw input to the same strict
parser and returns the same normalized model class:

```python
from pydantic import TypeAdapter
import zarr_metadata.pydantic as zmp

metadata = TypeAdapter(zmp.ZarrV3ArrayMetadata).validate_python(raw)
encoded = metadata.to_key_value()["zarr.json"]
```

A bare `TypeAdapter` over a public document `TypedDict` is a coercive shape
adapter, not a Zarr conformance validator; it may coerce values or discard
members that the strict model parser rejects.

## Validation boundary

The model validators enforce the declared document structure and a small set
of context-free consistency rules, including fixed format literals, finite
JSON numbers, non-negative dimensions, non-empty v3 codec pipelines, and one
`dimension_names` entry per array dimension. They do not interpret extension
names or configurations, resolve codec pipelines, or decide whether a data
type, chunk grid, codec, or storage transformer is supported. Those decisions
belong to consumer implementations.

The Pydantic integration's generated JSON Schemas express independently
checkable document structure and field constraints, but they are not a
replacement for runtime model validation. Standard JSON Schema treats a
mathematically integral number such as `1.0` as an integer, while the runtime
boundary requires Python `int` values, and it cannot express arbitrary
same-length relations such as `dimension_names` versus `shape` or v2 `chunks`
versus `shape`. Consumers should run the model parser after schema validation.

## Scope

At minimum, this library supports what Zarr-Python needs: the complete
Zarr v2 and v3 specs, consolidated metadata, and a subset of the metadata
defined in `zarr-extensions`. We are generally open to contributions that
add types, models, or structural validation for Zarr metadata with a
published spec.

Runtime array behavior is out of scope: nothing here encodes or decodes
chunks, resolves codec or data type names to implementations, or performs
store I/O. The models begin and end at the metadata documents themselves —
`from_key_value` / `to_key_value` map documents to store keys and bytes,
and everything past that belongs to consumer libraries.

## Developing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, pinned to the version CI uses
just docs-check  # strict build of the docs site
just check       # all of the above
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from anywhere in the repository as
`just packages/zarr-metadata/<recipe>`.

## Releasing

The package version is derived from git tags by `hatch-vcs`. Tags must
match the pattern `zarr_metadata-v<version>` (e.g. `zarr_metadata-v0.2.0`)
so they do not collide with the main `zarr-python` release tags.

To cut a release:

1. Create and push a tag of the form `zarr_metadata-v<version>` on the
   commit you want to publish, e.g.:
   ```
   git tag zarr_metadata-v0.2.0 <commit>
   git push origin zarr_metadata-v0.2.0
   ```
2. Pushing the tag fires the `zarr-metadata release` workflow, which
   builds the wheel/sdist (version resolved from the tag), runs an
   install smoke test, and publishes to PyPI via OIDC trusted publishing.

We intentionally do *not* create a GitHub Release for `zarr-metadata`
versions — GitHub Releases live at the repo level, and a zarr-metadata
release would surface in the zarr-python repo's Releases UI as if it
were a zarr-python release.

To dry-run a build against TestPyPI, dispatch the workflow manually
(`Actions` → `zarr-metadata release` → `Run workflow`). Manual dispatches
build from the current commit; with no recent tag the version will look
like `0.1.devN`, which is fine for TestPyPI.

## License

[MIT](./LICENSE.txt)
