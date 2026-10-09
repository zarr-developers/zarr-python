---
title: metadata
---

::: zarr.metadata
    options:
      members: false

The members listed for each class below are its public API. Other attributes and
methods, and the class constructors, may change without a deprecation cycle; build
these objects with `from_dict`.

::: zarr.metadata.ArrayMetadata

::: zarr.metadata.ArrayV3Metadata
    options:
      members:
        - shape
        - data_type
        - dtype
        - ndim
        - chunks
        - shards
        - fill_value
        - codecs
        - attributes
        - dimension_names
        - storage_transformers
        - zarr_format
        - node_type
        - from_dict
        - to_dict

::: zarr.metadata.ArrayV2Metadata
    options:
      members:
        - shape
        - chunks
        - shards
        - dtype
        - ndim
        - fill_value
        - order
        - filters
        - compressor
        - dimension_separator
        - attributes
        - zarr_format
        - from_dict
        - to_dict

::: zarr.metadata.GroupMetadata
    options:
      members:
        - attributes
        - zarr_format
        - node_type
        - consolidated_metadata
        - from_dict
        - to_dict

::: zarr.metadata.ConsolidatedMetadata
    options:
      members:
        - metadata
        - flattened_metadata
        - kind
        - must_understand
        - from_dict
        - to_dict

::: zarr.metadata.migrate_v3
