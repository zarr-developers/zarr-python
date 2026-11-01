Rename the package's vocabulary to match zarrs and the v3 spec: a name/value pair in a JSON metadata object is now a *field* (it was a *member*), and a value at an extension point that a scope resolves is now a *declaration* (it was a *field*). The public names change with no aliases:

- `AcceptedField`, `UnclaimedField`, `RefusedField` and `ResolvedField` are now `AcceptedDeclaration`, `UnclaimedDeclaration`, `RefusedDeclaration` and `ResolvedDeclaration`.
- `is_field`, `fields_of`, `field_key` and `field_kind` are now `is_declaration`, `declarations_of`, `declaration_key` and `declaration_kind`.
- `ZarrV3DataTypeField`, `ZarrV3ChunkGridField`, `ZarrV3ChunkKeyEncodingField`, `ZarrV3CodecField`, `ZarrV3StaticCodecField`, `ZarrV3StorageTransformerField` and `ZarrV2DataTypeField` are now the same names ending in `Declaration`, and a kind's `field_aliases` is now `declaration_aliases`.
- `ZarrV3MetadataFieldJSON` is now `ZarrV3DeclarationJSON`, and `field_json_schema_v3` is now `declaration_json_schema_v3`.
- A reading's `fields()` is now `declarations()`, and a model's `unclaimed_fields` is now `unclaimed_declarations`.
- The first parameter of `with_problems` is now `declarations` (it was `fields`), and the first parameter of `ZarrV3ConsolidatedMetadata` is now `value` (it was `member`).

Messages that said *member* now say *field*. `extra_fields`, `ExtraFieldDefinition`, `ZarrV3ExtensionField`, `StructField` and the pydantic `...MetadataField` types keep their names.
