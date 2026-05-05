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

## License

[MIT](./LICENSE.txt)
