# zarr-metadata

Python type definitions for Zarr v2 and v3 metadata.

## What this is

A typed-data package: `TypedDict` definitions and `Literal` aliases for the
JSON shapes specified by the [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
and [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
specifications, plus types for [`zarr-extensions`](https://github.com/zarr-developers/zarr-extensions/)
and a few widely-used-but-unspecified entities (e.g. consolidated metadata).
It also provides canonical frozen-dataclass models, structural validators,
parsers, store-key serialization, and optional Pydantic field integrations.
The optional integration requires Pydantic 2.13 or newer.

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
add types for Zarr metadata with a published spec.

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
