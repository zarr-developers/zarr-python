"""The extension layer: what an entity is, and what is in scope.

Every Zarr v3 extension point -- codecs, data types, chunk grids, chunk
key encodings, storage transformers -- is modelled as a class that
answers for itself. This module is the public door to that layer, for two
kinds of caller.

**Reading metadata.** `ArrayDocumentV3.from_json` is the fail-fast front
door: one call, and either every extension point is read or a single
`MetadataValidationError` carries every reason it is not. The entities it
yields know things the document does not spell out -- what a data type's
scalars are, which position a codec occupies, what a grid divides an
array into. A name the scope does not model is not a failure: it arrives
as an `Opaque` marked `out_of_scope`, for the reader to resolve
elsewhere.

    from zarr_metadata.v3.entity import ArrayDocumentV3, CodecEntity

    array = ArrayDocumentV3.from_json(json.loads(raw))   # or raises
    array.parts.grid.rank
    for codec in array.codecs:
        if isinstance(codec, CodecEntity):
            codec.kind                  # 'array_bytes'
        else:
            codec.json, codec.reason    # 'out_of_scope': resolve it yourself

**Writing an extension.** Subclass `CodecEntity`, `DataTypeEntity`,
`ChunkGridEntity` or `MetadataEntity`, declare the fields, and add it to
a scope:

    @dataclass(frozen=True)
    class AcmeLz4Codec(CodecEntity):
        # A required member has no default; an optional one defaults to
        # UNSET, so absence stays distinct from a JSON null.
        acceleration: int | UNSET = UNSET

        identifier: ClassVar[str] = "acme.lz4"
        kind: ClassVar[CodecKind] = "bytes_bytes"

    SCOPE = CORE_AND_EXTENSIONS.extended_with(
        codecs={AcmeLz4Codec.identifier: AcmeLz4Codec},
    )

    validate_array_metadata_v3(document, context=SCOPE)

The fields are the only place the shape is written. Which members exist,
which may be left out (the type admits `UNSET`), and how each one is
type-checked are all read off the annotations -- an `int`, a `Literal`
of names, an array, a nested entity type -- and `member_types` is for
the exception, an annotation the compiler does not read. A bound on a
value is written on the field too, in the `annotated_types` vocabulary:

    acceleration: Annotated[int, Interval(ge=1, le=65537)] | UNSET = UNSET

A rule about one member that is not a bound is a `@validates` rule: a
staticmethod taking the member's value, run only when the member is
present and has the type it declared, reporting relative to the member:

    @staticmethod
    @validates("order")
    def _order_permutes_itself(order: tuple[int, ...]) -> tuple[ValidationProblem, ...]:
        if sorted(order) != list(range(len(order))):
            return problem((), f"expected a permutation, got {order!r}", "invalid_value")
        return ()

A rule that reads two members together goes in a `value_problems`
staticmethod; annotate it with a TypedDict of the members so its body is
checked:

    class AcmeLz4Configuration(TypedDict, closed=True):
        acceleration: NotRequired[int]

    @staticmethod
    def value_problems(
        **members: Unpack[AcmeLz4Configuration],
    ) -> tuple[ValidationProblem, ...]:
        if "acceleration" not in members:
            return ()
        ...

A name in no scope is not rejected -- that is what extension openness
means -- so registering yours is how you get it judged rather than waved
through.

One known friction, under mypy only. An entity's `to_json` returns its own
object TypedDict, and mypy does not accept that where a
`ZarrV3MetadataFieldJSON` is wanted: it reads every TypedDict as
`Mapping[str, object]`, never as the `Mapping[str, JSONValue]` the
envelope declares. So putting `to_json()` output straight into a `codecs`
list needs a `cast` under mypy. Pyright accepts it.

That conversion is sound here, which is why pyright is the one that is
right. The rule mypy is applying exists because an ordinary TypedDict may
carry extra items of types it never declared, so the union of the
declared value types does not bound what is in the mapping. Every
TypedDict in this package is `closed` (PEP 728), which forbids exactly
that, and pyright implements PEP 728. Mypy does not yet -- see
python/mypy#8994 and python/mypy#18439.

The type therefore stays as it is. Widening `configuration` to
`Mapping[str, object]` or `Mapping[str, Any]` would satisfy mypy by
making the annotation say something false: a configuration's values are
JSON, and that is worth more than one checker's `cast`.
"""

from __future__ import annotations

from zarr_metadata.v3._chain import chain_problems, order_problems
from zarr_metadata.v3._document import ArrayDocumentV3, array_problems_v3, read_array_v3
from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    FROM_NAME,
    STORAGE_TRANSFORMERS,
    CheckCompiler,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Coerced,
    DataTypeEntity,
    ExtensionPointField,
    Ge,
    Gt,
    Interval,
    Le,
    Loc,
    Lt,
    MemberTypes,
    MetadataEntity,
    Opaque,
    StorageClass,
    TypeCheck,
    ValueRoutine,
    coerce_members,
    is_bool,
    is_int,
    is_integer,
    is_json_value,
    is_metadata_field,
    is_str,
    named_configuration,
    one_of,
    problem,
    register_check,
    sequence_of,
    validates,
    within,
)
from zarr_metadata.v3._parts import UNKNOWN_GRID, ArrayParts, ChunkGrid, Extents, shard_index_grid
from zarr_metadata.v3._registry import (
    CORE,
    CORE_AND_EXTENSIONS,
    Context,
    EntityTables,
    PartialEntityTables,
)
from zarr_metadata.v3.data_type._families import (
    FLOAT_SPECIALS,
    ComplexDataType,
    FloatDataType,
    IntegerDataType,
    NumpyTimeDataType,
    as_sequence,
    byte_values,
)

__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "CORE",
    "CORE_AND_EXTENSIONS",
    "DATA_TYPE",
    "FLOAT_SPECIALS",
    "FROM_NAME",
    "STORAGE_TRANSFORMERS",
    "UNKNOWN_GRID",
    "ArrayDocumentV3",
    "ArrayParts",
    "CheckCompiler",
    "ChunkGrid",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "ComplexDataType",
    "Context",
    "DataTypeEntity",
    "EntityTables",
    "ExtensionPointField",
    "Extents",
    "FloatDataType",
    "Ge",
    "Gt",
    "IntegerDataType",
    "Interval",
    "Le",
    "Loc",
    "Lt",
    "MemberTypes",
    "MetadataEntity",
    "NumpyTimeDataType",
    "Opaque",
    "PartialEntityTables",
    "StorageClass",
    "TypeCheck",
    "ValueRoutine",
    "array_problems_v3",
    "as_sequence",
    "byte_values",
    "chain_problems",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_metadata_field",
    "is_str",
    "named_configuration",
    "one_of",
    "order_problems",
    "problem",
    "read_array_v3",
    "register_check",
    "sequence_of",
    "shard_index_grid",
    "validates",
    "within",
]
