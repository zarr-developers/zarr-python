"""The kinds of metadata in a Zarr v3 document, and what reading one needs that is specific to v3.

The kinds at the spec's five extension points are `DataTypeDefinition`,
`ChunkGridDefinition`, `ChunkKeyEncodingDefinition`, `CodecDefinition`
and `StorageTransformerDefinition`. `ArrayExtraFieldDefinition` and
`GroupExtraFieldDefinition` read members the spec does not define. This module also holds the chunk that a
codec receives, how a data type's values are stored, the raw bits name
`r*`, and the JSON Schema of a v3 field. The reading itself is in
`zarr_metadata.definition`, which both formats share.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    TypeGuard,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeAliasType, is_typeddict

from zarr_metadata._json import (
    ValidationProblem,
    is_object,
    is_tuple,
    refine_json,
    refine_user_data,
    with_input,
)
from zarr_metadata._typed_json import (
    Loc,
    alias_value,
    is_alias,
    is_union,
    problem,
    strip_annotation,
    typeddict_keys,
)
from zarr_metadata.definition._definition import (
    AcceptedField,
    C,
    Definition,
    Nested,
    Problems,
    ResolvedField,
    WithFillValue,
    asked,
    is_field,
    no_rules,
    read_only,
    ruled,
)
from zarr_metadata.v3._common import (
    ZarrV3ChunkGridField,
    ZarrV3ChunkKeyEncodingField,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
    ZarrV3StaticCodecField,
    ZarrV3StorageTransformerField,
)
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue


def unknown_lengths(configuration: object, nested: object, shape: tuple[int, ...]) -> Lengths:
    """The chunk lengths of a grid that does not declare them: unknown along each axis of the array."""
    return (None,) * len(shape)


def unknown_chunk(*_: object) -> Chunk:
    """The output of an array -> array codec that does not declare its output: a chunk with nothing known about it."""
    return Chunk()


def no_pipelines(*_: object) -> Mapping[str, Chunk]:
    """The pipelines of a codec that contains none: an empty mapping."""
    return {}


StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How a data type's values are stored: in single bytes, in several bytes at a time, or each value in as many bytes as it needs.

A multi-byte number is stored in a byte order, which the `bytes` codec's
`endian` sets. A value made of single bytes (a `uint8`, or a struct of
`int8` fields) has no byte order. A value whose size varies needs its
own codec.
"""


STORAGE_CLASSES: Final[tuple[StorageClass, ...]] = get_args(StorageClass)
"""The storage classes. `storage_of` checks a definition's result against these."""


def single_byte(*_: object) -> StorageClass:
    """The storage of a data type made of single bytes, where byte order does not apply, such as `uint8`."""
    return "single_byte"


def multi_byte(*_: object) -> StorageClass:
    """The storage of a data type with multi-byte numbers, where byte order applies, such as `int16`."""
    return "multi_byte"


def variable_length(*_: object) -> StorageClass:
    """The storage of a data type whose values vary in size, such as `string`."""
    return "variable_length"


def unknown_storage(*_: object) -> None:
    """The storage of a data type that does not declare it: unknown."""


RAW_BYTES_NAME: Final = "r*"
"""The name raw bits are filed under: `r*`, as written in the specification's table of data types.

A document writes `r` followed by the size in bits (`r8`, `r16`), so raw
bits are the only data type whose name encodes its configuration.
`filed_under` reads such a name as `r*` with the size as its
configuration. `r*` itself is the table's notation, and no document uses
it as a name
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47).
"""


RAW_BYTES_NAME_PATTERN: Final = re.compile(r"r([0-9]{1,100})")
"""A raw bits name: `r` followed by a size in bits of up to a hundred digits, matched against the whole name, whether or not the size is allowed.

Only ASCII digits match. `\\d` would also match every other Unicode
decimal digit, so `r\uff11\uff16` would be read as sixteen bits, and a
third-party name spelled that way would be mistaken for raw bits. A size
the spec does not allow (`r0`, `r12`) is still raw bits, so it is
reported as a bad size instead of being treated as an unknown extension.
A hundred digits is more than any real size. `r` followed by more digits
than that is an ordinary name, which the spec allows, and which nothing
in scope claims.
"""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DataTypeDefinition(WithFillValue[C], kind=True, format=3):
    """A data type, and the fill value of an array with that data type.

    `fill_value` is the JSON shape of a fill value, such as
    `Int8FillValue`. It is an annotation that the checker reads the same
    way it reads a configuration's members, including any range.
    `fill_value_rules` yields the spec violations in a fill value of that
    shape that the type cannot express, such as a hex string of the wrong
    width.
    The rules receive the configuration, the nested fields as the scope
    read them (a struct's field types), and the typed fill value. A data
    type that does not declare its fill value accepts any JSON.

    `fill_value_canonical` writes a fill value without problems (well
    typed and allowed by the rules) in the single canonical spelling of
    its value. So two fill values are the same value of the type exactly
    when their canonical spellings are written the same way: `"NaN"` and
    `"0x7fc00000"` are the same `float32`, but `0.0` and `-0.0` are two
    different values. It receives the same arguments as the rules. A data
    type that does not declare it has one spelling per value: the value as
    written.

    `storage` declares how the values are stored: in single bytes, in
    several bytes at a time, or each value in as many bytes as it needs.
    The `bytes` codec uses this to decide whether the data type it
    receives needs an `endian` (it does for multi-byte numbers). A
    struct's storage depends on its fields, so `storage` receives the
    nested fields as the scope read them. A data type that does not
    declare it leaves its storage unknown.

    A data type named the way a document writes raw bits of one size (such
    as `r16`) is rejected. That name is read as `r*`, so this definition
    would never be used.
    """

    label: ClassVar[str] = "data type"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3DataTypeField,)

    @classmethod
    def filed_under(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
        if name == RAW_BYTES_NAME:
            return None, None
        written = RAW_BYTES_NAME_PATTERN.fullmatch(name)
        if written is not None:
            return RAW_BYTES_NAME, {"bits": int(written.group(1))}
        return name, None

    def carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        if self.name != RAW_BYTES_NAME:
            return None
        return f"r{configuration['bits']}"

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        # A data type with nothing to configure is written as its bare name,
        # which is how core data types have been written since Zarr v3.0.
        if len(configuration) != 0:
            return {"name": name, "configuration": configuration}
        return name

    storage: Callable[[C, Nested], StorageClass | None] = unknown_storage
    """How the values are stored, given the configuration and the nested fields; None when unknown."""

    def _refusal(self) -> str | None:
        if RAW_BYTES_NAME_PATTERN.fullmatch(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes raw bits of one size, which read as "
                f"{RAW_BYTES_NAME!r}; to read raw bits your own way, define {RAW_BYTES_NAME!r}"
            )
        # Use the explicit form, not bare `super()`, because a dataclass
        # with slots is rebuilt.
        return super(DataTypeDefinition, self)._refusal()


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ChunkGridDefinition(Definition[C], kind=True, format=3):
    """A chunk grid, and the array shapes it fits.

    `shape_rules` yields the spec violations in a grid with this
    configuration over an array of a given shape, such as a dimension with
    no chunk length, or chunks that do not cover the dimension. It receives
    the configuration, the nested fields as the scope read them, and the
    shape, and it locates its problems in the configuration. A grid that
    does not declare shape rules fits every shape.

    `chunk_lengths` determines what the first codec of the array's
    pipeline receives: the lengths of the grid's chunks along each axis of
    an array whose shape the grid fits. A regular grid has one length per
    axis, and a rectilinear grid has every length it lists. It is called
    only for a grid that its shape rules accept. A grid that does not
    declare it leaves the lengths along every axis unknown.
    """

    label: ClassVar[str] = "chunk grid"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3ChunkGridField,)

    shape_rules: Callable[[C, Nested, tuple[int, ...]], Iterable[ValidationProblem]] = no_rules
    """The spec violations in this grid over an array of a given shape, located in the configuration."""
    chunk_lengths: Callable[[C, Nested, tuple[int, ...]], Lengths] = unknown_lengths
    """The lengths of the chunks along each axis of an array whose shape the grid fits; None where unknown."""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ChunkKeyEncodingDefinition(Definition[C], kind=True, format=3):
    """A chunk key encoding."""

    label: ClassVar[str] = "chunk key encoding"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3ChunkKeyEncodingField,)


CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""What a codec converts its input to: the three positions a pipeline puts in order."""


CodecSize = Literal["static", "dynamic"]
"""Whether the size of a codec's output is determined by the size of its input.

`static`: it is. For example, `bytes` writes each element at its width,
and `crc32c` adds four bytes. `dynamic`: the size depends on the values,
as for every compressor.
"""


_UNASKED: Final[Mapping[CodecKind, tuple[str, ...]]] = {
    "array_array": (),
    "array_bytes": ("transition",),
    "bytes_bytes": ("chunk_rules", "transition", "pipelines"),
}
"""The functions that are never called for a codec of each kind. A bytes -> bytes codec receives bytes, and only an array -> array codec passes on a chunk."""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class CodecDefinition(Definition[C], kind=True, format=3):
    """A codec: what it converts its input to, and whether the size of its output is static.

    A codec that receives an array (array -> array, array -> bytes)
    declares the spec violations for a given input `Chunk` with
    `chunk_rules`. It receives the configuration, the nested fields as the
    scope read them, and the chunk, and it locates its problems in the
    configuration. For example, a `bytes` codec without `endian` that
    receives a multi-byte data type is a problem. An array -> array codec
    also declares its output with `transition`: given its input chunk, it
    returns the chunk the next codec receives. For example, `transpose`
    permutes the axes, and `cast_value` changes the data type. These two
    functions correspond to the spec: a codec computes its output from the
    shape and data type of its input, and "If the
    decoded_representation_type is not supported, this algorithm must
    fail with an error"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L987-L994).
    The transition is called for every input chunk, whatever the chunk
    rules found, so it must return only what is true in either case. For
    example, a `transpose` whose `order` has the wrong number of axes
    passes on unknown lengths. A codec that does not declare its output
    passes the next codec a chunk with nothing known about it.

    A codec that contains its own pipelines declares what each one
    receives with `pipelines`. Given the codec's input chunk, it returns
    the chunk that the first codec of each pipeline receives, keyed by the
    configuration member that holds the pipeline. For example, a shard's
    inner codecs receive its inner chunks, and its index codecs receive
    the shard index. Like the transition, it is called whatever the chunk
    rules found, and must return only what is true in either case. A
    function that is never called for a codec of its kind is rejected,
    such as the chunk rules of a bytes -> bytes codec, which receives
    bytes.
    """

    label: ClassVar[str] = "codec"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3CodecField, ZarrV3StaticCodecField)

    kind: CodecKind
    size: CodecSize
    chunk_rules: Callable[[C, Nested, Chunk], Iterable[ValidationProblem]] = no_rules
    """The spec violations for this codec with a given input chunk, located in the configuration."""
    transition: Callable[[C, Nested, Chunk], Chunk] = unknown_chunk
    """The chunk the next codec receives, given the chunk this array -> array codec receives."""
    pipelines: Callable[[C, Nested, Chunk], Mapping[str, Chunk]] = no_pipelines
    """The chunk each contained pipeline receives, keyed by the configuration member that holds the pipeline."""

    def _refusal(self) -> str | None:
        kind: object = self.kind
        if kind not in get_args(CodecKind):
            return f"{self.name!r}: kind is one of {get_args(CodecKind)!r}, got {kind!r}"
        size: object = self.size
        if size not in get_args(CodecSize):
            return f"{self.name!r}: size is one of {get_args(CodecSize)!r}, got {size!r}"
        defaults = {member.name: member.default for member in dataclasses.fields(CodecDefinition)}
        for member in _UNASKED[self.kind]:
            if getattr(self, member) is not defaults[member]:
                return f"{self.name!r}: {member}, which no codec of kind {kind!r} is asked"
        return None

    @classmethod
    def alias_problems(
        cls, alias: TypeAliasType, definition: Definition[Any], loc: Loc
    ) -> Problems:
        """A codec of dynamic size in a member annotated with `ZarrV3StaticCodecField`, which admits only codecs of static size, is a problem at the field."""
        if alias not in STATIC_SIZE_ALIASES:
            return ()
        if not isinstance(definition, CodecDefinition) or definition.size == "static":
            return ()
        return problem(
            loc,
            f"{definition.name!r} is a codec of dynamic size, and only codecs of static size "
            "may be used here",
            "invalid_value",
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StorageTransformerDefinition(Definition[C], kind=True, format=3):
    """A storage transformer."""

    label: ClassVar[str] = "storage transformer"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3StorageTransformerField,)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ExtraFieldDefinition(Definition[C]):
    """A member of a v3 document that the spec does not define, an extra field: its name is the member's key, and its configuration is the member's value.

    This is not a kind. `ArrayExtraFieldDefinition` and
    `GroupExtraFieldDefinition` are, because one key may mean different
    things in an array document and in a group document. An extra field is
    not one of the spec's extension points, and no registry names it, so
    any key may name one. As a field it is written as the member alone,
    `{key: value}`, and it is located at the member's key. A definition
    reads a member whose value is an object. A member that no definition
    in scope claims is not read: a reader keeps it as it is written, and
    lists it in `must_understand_fields` unless it says
    `"must_understand": false`.

    `finite` says whether every number in the member must be finite, as
    RFC 8259 JSON requires. A member that holds user data sets it to
    False: consolidated metadata holds node documents, whose attributes
    may hold `NaN`, `Infinity` and `-Infinity`. A member that nothing
    claims is not read, so like attributes it may hold any number.

    The spec lets every extension member carry `"must_understand": false`,
    which tells a reader that does not understand the member that it may
    ignore it. So the configuration declares `must_understand`, and a
    definition whose configuration does not is refused. A member the spec
    defines, such as an array's `shape`, is never an extra field, and a
    definition named after one is refused too.
    """

    finite: bool = True
    """Whether every number in the member must be finite."""

    spec_members: ClassVar[frozenset[str]] = frozenset()
    """The members the spec defines in a document this kind reads extra fields of, which are never extra fields."""

    def _finite_numbers(self) -> bool:
        return self.finite

    def _refusal(self) -> str | None:
        finite = cast("object", self.finite)
        if not isinstance(finite, bool):
            return f"{self.name!r}: finite is a boolean, got {finite!r}"
        if is_typeddict(self.configuration) and not _takes_the_waiver(self.configuration):
            return (
                f"{self.name!r}: expected a configuration that takes must_understand as a "
                "boolean that may be false, which the spec lets every extension member carry; "
                f"{self.configuration.__name__} does not"
            )
        return None

    @classmethod
    def name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        # No registry names extra fields, so any key the spec does not
        # define may name one.
        if name in cls.spec_members:
            return ValidationProblem(
                at, f"{name!r} is a member the spec defines, not an extra field", "invalid_value"
            )
        return None

    @classmethod
    def named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        # A member whose value is not an object has no configuration. Only
        # a definition that claims the member reads it, and `refined`
        # refuses such a value then.
        if not is_object(value) or len(value) != 1:
            return None, None, ()
        ((name, member),) = value.items()
        if not isinstance(name, str):
            return None, None, ()
        if not is_object(member):
            return name, None, ()
        return name, {key: item for key, item in member.items() if isinstance(key, str)}, ()

    @classmethod
    def envelope_problems(cls, value: object) -> Problems:
        name = cls.named_configuration(value)[0]
        if name is None:
            return (
                ValidationProblem(
                    (), "expected a member, as an object with one string key", "invalid_type"
                ),
            )
        bad = cls.name_problem(name, ())
        return () if bad is None else (bad,)

    @classmethod
    def refined(
        cls, data: object, loc: Loc, definition: Definition[Any] | None
    ) -> tuple[JSONValue | None, Problems]:
        # The member's value is at the member's location, `loc`. A member
        # nothing claims is not read, so like attributes it is user data.
        # A claimed member is user data when its definition says so, and
        # its value must be an object.
        name = cls.named_configuration(data)[0]
        if name is None or not is_object(data):
            return refine_json(data, loc)
        if definition is not None and not is_object(data[name]):
            return None, (ValidationProblem(loc, "expected an object", "invalid_type"),)
        refine = (
            refine_json
            if isinstance(definition, ExtraFieldDefinition) and definition.finite
            else refine_user_data
        )
        value, problems = refine(data[name], loc)
        return (None if len(problems) != 0 else {name: value}), problems

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        return {name: configuration}

    @classmethod
    def unclaimed_json(
        cls, json: JSONValue, name: str, configuration: Mapping[str, JSONValue]
    ) -> JSONValue:
        # A member no definition claims is written as it is, whatever its
        # value; only a definition reads it as an object.
        return json

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        return loc

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        return loc


def _takes_the_waiver(configuration: type) -> bool:
    """Whether `configuration`, a TypedDict, takes the spec's `"must_understand": false`: the type it gives `must_understand`, declared or as another key, admits `false` and only booleans."""
    keys = typeddict_keys(configuration)
    declared = keys.members.get("must_understand")
    booleans = _booleans(keys.extra_items if declared is None else declared[0], frozenset())
    return booleans is not None and False in booleans


def _booleans(annotation: object, seen: frozenset[object]) -> frozenset[bool] | None:
    """The booleans `annotation` admits when it admits only booleans, read through its aliases, unions and `Literal`s; None when it admits anything else."""
    inner = strip_annotation(annotation)[0]
    if is_alias(inner):
        # An alias that refers to itself adds nothing it has not already.
        return frozenset() if inner in seen else _booleans(alias_value(inner), seen | {inner})
    if inner is bool:
        return frozenset({True, False})
    if get_origin(inner) is Literal:
        values = get_args(inner)
        return frozenset(values) if all(isinstance(value, bool) for value in values) else None
    if not is_union(inner):
        return None
    admitted: set[bool] = set()
    for branch in get_args(inner):
        found = _booleans(branch, seen)
        if found is None:
            return None
        admitted |= found
    return frozenset(admitted)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ArrayExtraFieldDefinition(ExtraFieldDefinition[C], kind=True, format=3):
    """An extra field of a v3 array document: a member the spec does not define, read by its key."""

    label: ClassVar[str] = "array extra field"
    spec_members: ClassVar[frozenset[str]] = frozenset(
        typeddict_keys(ZarrV3ArrayMetadataJSON).members
    )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GroupExtraFieldDefinition(ExtraFieldDefinition[C], kind=True, format=3):
    """An extra field of a v3 group document: a member the spec does not define, read by its key. `consolidated_metadata` is one."""

    label: ClassVar[str] = "group extra field"
    spec_members: ClassVar[frozenset[str]] = frozenset(
        typeddict_keys(ZarrV3GroupMetadataJSON).members
    )


STATIC_SIZE_ALIASES: Final[frozenset[TypeAliasType]] = frozenset({ZarrV3StaticCodecField})
"""The field aliases whose codec must be of static size."""


Lengths: TypeAlias = tuple[frozenset[int] | None, ...]
"""For each axis, every chunk length along that axis (a set, because a rectilinear grid's lengths differ), or None where unknown."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """The input a codec receives: chunks with given lengths along each axis, and a data type.

    Unknown parts are None. This includes the lengths along an axis that
    the grid does not declare, and every part of the chunk output by a
    codec that does not declare its output. A data type field that the
    scope did not read is kept as written, and it also says nothing about
    the values. A codec's chunk rules check only what is known, so a chunk
    with nothing known about it, `Chunk()`, never causes a problem.
    """

    lengths: Lengths | None = None
    """For each axis, the chunk lengths along it; None when even the number of axes is unknown."""
    data_type: ResolvedField[DataTypeDefinition[Any]] | None = None
    """The data type field of the values, as a scope read it. None when no field declares the data type: a document with no data type (its reading holds `UNSET`) passes the pipeline a chunk of unknown type."""

    def __post_init__(self) -> None:
        lengths = cast("object", self.lengths)
        if lengths is not None and not _is_lengths(lengths):
            msg = f"a chunk's lengths are a frozenset of integers or None per axis, got {lengths!r}"
            raise TypeError(msg)
        data_type = cast("object", self.data_type)
        if data_type is not None and not _is_data_type_field(data_type):
            msg = f"a chunk's data type is a data type field a scope read, got {data_type!r}"
            raise TypeError(msg)

    @property
    def rank(self) -> int | None:
        """The number of axes; None when unknown."""
        return None if self.lengths is None else len(self.lengths)


def _is_data_type_field(value: object) -> bool:
    """Whether `value` is a field that a scope read as a data type."""
    return is_field(value) and value.read_as is DataTypeDefinition


def _is_lengths(value: object) -> TypeGuard[Lengths]:
    """Whether `value` is valid chunk lengths: for each axis, a frozenset of integers or None."""
    if not is_tuple(value):
        return False
    for axis in value:
        if axis is None:
            continue
        if not isinstance(axis, frozenset) or not all(
            isinstance(length, int) and not isinstance(length, bool)
            for length in cast("frozenset[object]", axis)
        ):
            return False
    return True


def storage_of(data_type: ResolvedField[DataTypeDefinition[Any]]) -> StorageClass | None:
    """How the values of `data_type` (a data type field a scope read) are stored; None when unknown.

    The storage is unknown when the scope did not read the data type, or
    its definition does not declare storage. The definition's `storage` is
    the extension author's code. Its result is checked to be a storage
    class. If it raises an error, a note is added that says which data
    type's storage raised it.
    """
    if not isinstance(data_type, AcceptedField):
        return None
    definition, configuration = data_type.definition, data_type.configuration
    found = asked(
        definition,
        "storage",
        lambda: cast("object", definition.storage(read_only(configuration), data_type.nested)),
    )
    if found is not None and found not in STORAGE_CLASSES:
        msg = (
            f"{definition.name!r}: its storage gives one of {STORAGE_CLASSES!r} or None, "
            f"got {found!r}"
        )
        raise TypeError(msg)
    return found


def chunk_grid_lengths(
    chunk_grid: ResolvedField[ChunkGridDefinition[Any]], shape: tuple[int, ...], loc: Loc = ()
) -> tuple[Lengths, Problems]:
    """The chunk lengths of `chunk_grid` (a chunk grid field a scope read) along each axis of an array of `shape`, and the problems with the grid for that shape.

    A chunk has an extent "for each dimension of the array"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L277-L281),
    so the lengths have an entry for each dimension of `shape`, with None
    where they are unknown. The grid's shape rules check it first, and
    locate their problems in the configuration, under `loc` (the location
    of the field). For example, a regular grid with chunk lengths for two
    dimensions over an array with three dimensions is a problem. Only a
    grid that fits the shape returns its lengths. A grid that the scope
    did not read (it is out of scope or invalid) is not checked and
    returns no lengths. The result of a grid's `chunk_lengths` is checked:
    lengths of the wrong type raise `TypeError`, and lengths with a
    different number of axes than `shape` raise `ValueError`. Both
    indicate a bug in the definition, not in the field.
    """
    unknown: Lengths = (None,) * len(shape)
    if not isinstance(chunk_grid, AcceptedField):
        return unknown, ()
    definition, configuration = chunk_grid.definition, chunk_grid.configuration
    at = (*loc, "configuration")
    problems = ruled(
        definition,
        lambda: definition.shape_rules(read_only(configuration), chunk_grid.nested, shape),
        at,
    )
    if len(problems) != 0:
        return unknown, with_input(problems, chunk_grid.json, loc)
    lengths = asked(
        definition,
        "chunk lengths",
        lambda: cast(
            "object", definition.chunk_lengths(read_only(configuration), chunk_grid.nested, shape)
        ),
        at,
    )
    if not _is_lengths(lengths):
        msg = (
            f"{definition.name!r}: its chunk_lengths give a frozenset of integers or None "
            f"per axis, got {lengths!r}"
        )
        raise TypeError(msg)
    if len(lengths) != len(shape):
        msg = (
            f"{definition.name!r}: its chunk_lengths gave {len(lengths)} axes, "
            f"for a shape of {len(shape)}"
        )
        raise ValueError(msg)
    return lengths, ()


__all__ = [
    "RAW_BYTES_NAME",
    "RAW_BYTES_NAME_PATTERN",
    "STATIC_SIZE_ALIASES",
    "ArrayExtraFieldDefinition",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkKeyEncodingDefinition",
    "CodecDefinition",
    "CodecKind",
    "CodecSize",
    "DataTypeDefinition",
    "ExtraFieldDefinition",
    "GroupExtraFieldDefinition",
    "Lengths",
    "StorageClass",
    "StorageTransformerDefinition",
    "chunk_grid_lengths",
    "multi_byte",
    "no_pipelines",
    "single_byte",
    "storage_of",
    "unknown_chunk",
    "unknown_lengths",
    "unknown_storage",
    "variable_length",
]
