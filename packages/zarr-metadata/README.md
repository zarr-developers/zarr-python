# zarr-metadata

Documentation: <https://zarr-metadata.readthedocs.io/>

Types, models and validators for Zarr metadata documents, with minimal
dependencies.

`zarr-metadata` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
and released independently of `zarr`. Install it with:

```
pip install zarr-metadata
```

## What this is

A collection of data structures and routines that closely model the
metadata rules in the Zarr v2 and v3 specifications and in zarr-extensions:

- `TypedDict` shapes for every document and configuration.
- Immutable document models. Their readers report every problem at its
  location.
- **Contexts**, which say exactly which codecs, data types and other
  extensions a program reads.

This is *not* a full Zarr implementation. Nothing here reads or writes
chunks, encodes or decodes data, or performs store I/O.

```python test="true" session="readme"
import json
from zarr_metadata import ZarrV3ArrayMetadata

with open("zarr.json", "rb") as f:
    raw = json.load(f)

metadata = ZarrV3ArrayMetadata.from_json(raw)
```

`from_json` returns the model, or raises `MetadataValidationError` with
every problem it found. The
[documentation](https://zarr-metadata.readthedocs.io/) has a guide for
each kind of use, and the API reference.

## License

[MIT](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-metadata/LICENSE.txt)
