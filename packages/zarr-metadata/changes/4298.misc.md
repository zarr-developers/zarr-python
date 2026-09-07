The chunk key encoding TypedDicts (`DefaultChunkKeyEncodingConfiguration`,
`DefaultChunkKeyEncodingObject`, `V2ChunkKeyEncodingConfiguration`,
`V2ChunkKeyEncodingObject`) now declare `extra_items=JSONValue` (PEP 728),
making them assignable to `Mapping[str, JSONValue]` under type checkers that
support closed/extra-items TypedDicts. No runtime behavior change.
