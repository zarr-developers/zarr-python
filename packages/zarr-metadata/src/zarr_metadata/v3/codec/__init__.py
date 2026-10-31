"""Zarr v3 codec spec types.

Each codec defined by the spec or by zarr-extensions has its own submodule
(`blosc`, `bytes`, `cast_value`, `crc32c`, `gzip`, `scale_offset`,
`sharding_indexed`, `transpose`, `vlen_bytes`, `vlen_utf8`, `zstd`). Import a codec's types and its
definition from its submodule: `<X>CodecMetadata`, the canonical type for
the codec's permitted JSON shapes (object form plus, where the spec allows,
a bare-string short-hand form), `<X>CodecObject`, `<X>_CODEC`, and, for a
codec that has a configuration, `<X>CodecConfiguration`.

For the JSON of any codec entry (one of the entries in array metadata's
`codecs`), import `ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`. A
configuration member that holds a codec, such as sharding's inner
pipelines, is annotated `ZarrV3CodecField` from
`zarr_metadata.v3.definition` instead, so that the reader reads it as a
codec.

Each codec's pipeline position (`array -> array`, `array -> bytes`,
`bytes -> bytes`) is its definition's `kind`, in
`zarr_metadata.v3.definition`.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

__all__: list[str] = []
