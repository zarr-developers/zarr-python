# zarr-metadata

Python type definitions for Zarr v2 and v3 metadata.

## What this is

A typed-data package: `TypedDict` definitions and `Literal` aliases for the
JSON shapes specified by the [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
and [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
specifications, plus types for [`zarr-extensions`](https://github.com/zarr-developers/zarr-extensions/)
and a few widely-used-but-unspecified entities (e.g. consolidated metadata).

## What this is for

These types describe the JSON shape of Zarr metadata. They are
intended for libraries that **read, write, validate, or transform**
Zarr metadata. Pair them with a runtime validator like
[pydantic](https://docs.pydantic.dev/) to check JSON loaded from disk:

```python
import json
from pydantic import TypeAdapter
from zarr_metadata.v3.array import ArrayMetadataV3

with open("zarr.json", "rb") as f:
    raw = json.load(f)

metadata = TypeAdapter(ArrayMetadataV3).validate_python(raw)
```

## What this is *not*

- Not a parser or builder. There are no `make_array_metadata(...)` factories —
  that surface belongs to consumer libraries.
- Not a runtime validator on its own. Pair with `pydantic`, `msgspec`, or
  similar to enforce shapes at decode time.

Even with a runtime validator, these types only describe **structural**
shape — they will not flag *semantically* invalid metadata, like a 3D v3
array whose `dimension_names` has 4 entries instead of 3. That's a job
for downstream validator routines.

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
