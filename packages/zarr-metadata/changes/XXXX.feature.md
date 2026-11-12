**Breaking:** a v3 codec is defined with the class of its kind,
`ArrayArrayCodecDefinition`, `ArrayBytesCodecDefinition` or
`BytesBytesCodecDefinition`, instead of `CodecDefinition(kind=...)`. The kind
is now the class's `kind` property, and `CodecDefinition` is their abstract
base, which a scope and `resolve` still use to name the kind. A codec that
receives an array, an `ArrayCodecDefinition`, has `chunk_rules` and
`pipelines`, an array -> array codec also has `transition`, and a bytes ->
bytes codec, which receives bytes, has none of them, so building one with any
of them is a `TypeError` from the class itself.
