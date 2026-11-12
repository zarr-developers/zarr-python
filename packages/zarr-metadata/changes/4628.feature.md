**Breaking:** a v3 codec is defined with the class of its kind,
`ArrayArrayCodecDefinition`, `ArrayBytesCodecDefinition` or
`BytesBytesCodecDefinition`, instead of `CodecDefinition(kind=...)`. The kind
is now the class's `kind` property, and `CodecDefinition` is their abstract
base, which a scope and `resolve` still use to name the kind. A codec that
receives an array, an `ArrayCodecDefinition`, has `chunk_rules` and
`pipelines`, an array -> array codec also has `transition`, and a bytes ->
bytes codec, which receives bytes, has none of them, so building one with any
of them is a `TypeError` from the class itself.

A v3 data type definition declares what kind of numbers its values are with
`number_class`, a `NumberClass`: `"integral"`, `"floating_point"`,
`"complex"` or `"non_numeric"`. `cast_value` and `scale_offset` read it to
decide which data types they accept, where each kept a list of data type
names, so a data type defined elsewhere is judged by what it declares: both
codecs refuse one that declares complex or non-numeric values, and
`cast_value` refuses to `wrap` to one that declares floating-point values. A
data type that declares no number class is accepted by both, as before. A data
type defined elsewhere under one of the float names the `cast_value` spec
lists, such as `bfloat16`, was refused as a `wrap` target by its name, and is
now refused once it declares `"floating_point"`.
