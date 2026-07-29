---
title: zarr-metadata
---

# `zarr-metadata`

Spec-defined metadata types, models, and validators for Zarr v2 and v3.

`zarr-metadata` is a standalone package developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
and released separately from `zarr` itself. Install it with:

```
pip install zarr-metadata
```

The package has two layers and an optional integration:

- **Typed JSON shapes** ([`zarr_metadata.v2`](v2.md) and
  [`zarr_metadata.v3`](v3/index.md)): `TypedDict` definitions and `Literal`
  aliases for the JSON documents specified by the
  [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html) and
  [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  specifications, plus types for
  [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/) and a
  few widely-used-but-unspecified entities (e.g. consolidated metadata).
- **Document models** ([`zarr_metadata.model`](model.md)): canonical
  frozen-dataclass models of whole metadata documents, with structural
  validators, loc-aware parsers, and store-key (de)serialization.
- **Optional Pydantic integration** ([`zarr_metadata.pydantic`](pydantic.md),
  requires Pydantic 2.13 or newer): each model as a Pydantic field type that
  validates raw documents through the same strict parser.

Every public name is also re-exported at the top level, so
`from zarr_metadata import ZarrV3ArrayMetadataJSON` and
`from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON` are equivalent.

The `TypedDict` definitions describe the static JSON shape of Zarr metadata
for type checkers. For strict, loc-aware runtime validation of JSON loaded
from a store, use the model parsers:

```python exec="false" reason="zarr-metadata is not installed in the docs build environment"
import json
from zarr_metadata.model import ZarrV3ArrayMetadata

with open("zarr.json", "rb") as f:
    raw = json.load(f)

metadata = ZarrV3ArrayMetadata.from_json(raw)
```

## Common types

A few cross-cutting aliases are exported only from the top-level
`zarr_metadata` namespace:

::: zarr_metadata.JSONValue

::: zarr_metadata.ZarrV3NamedConfigJSON
