"""In-memory models for Zarr group and consolidated metadata documents."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, TypeGuard, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    is_canonical_json,
    not_an_object,
    outside_of,
    refine_json,
    refine_user_data,
    with_input,
)
from zarr_metadata._json import prefixed as _prefix
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._array import (
    ZarrV3ArrayMetadata,
    array_json,
    array_model,
    must_understand_subset,
    read_array_metadata_v3,
)
from zarr_metadata.model._validation import (
    GROUP_METADATA_REQUIRED_KEYS_V3,
    GROUP_METADATA_STANDARD_KEYS_V3,
    NO_SCOPE,
    ArrayMembersV3,
    StoreKey,
    ZarrV3ArrayMetadataReading,
    attributes_of,
    check_literal,
    construct,
    dump_store_json,
    load_store_json,
    missing_keys,
    other_members,
    overlapping,
    parse_group_metadata_v2,
    read_array_v3,
    unexpected_keys,
)
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.consolidated import ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3.array import ZarrV3ExtensionField
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY
from zarr_metadata.v3.group import ZARR_V3_GROUP_METADATA_STORE_KEY, ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v2.consolidated import ZarrV2ConsolidatedMetadataStoreKey
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON, ZarrV2GroupMetadataStoreKey
    from zarr_metadata.v3._definition import Resolved
    from zarr_metadata.v3.consolidated import ZarrV3ConsolidatedMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSONPartial, ZarrV3GroupMetadataStoreKey


class ZarrV3GroupMetadataUpdate(TypedDict, total=False, extra_items=ZarrV3ExtensionField | UNSET):
    """The members `ZarrV3GroupMetadata.update` puts in place: each as a document writes it, or `UNSET` to leave it out.

    `consolidated_metadata` is a member the spec does not define, so it is
    one of the extra items: given, its documents are read; left out, the
    group keeps the models it holds.
    """

    attributes: Mapping[str, JSONValue] | UNSET


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3GroupMetadata:
    """In-memory model of a v3 group metadata document.

    A canonical, semantically lossless representation of the `zarr.json`
    content for a group. The `consolidated_metadata` reference-implementation
    convention is modeled as a typed field holding the model of each
    document it holds; every other unknown top-level key lands in
    `extra_fields` verbatim. A model holds no scope, as
    `ZarrV3ArrayMetadata` holds none: `from_json`, `create_default` and
    `update` each take the one they read new JSON in. It checks itself
    when it is built, as `ZarrV3ArrayMetadata` does, and holds its members
    as the read refines them: its own members -- each document its
    consolidated metadata holds is a model, which checked itself -- or the
    constructor raises `MetadataValidationError` with every problem.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["group"] = field(default="group", init=False)
    attributes: dict[str, JSONValue]
    consolidated_metadata: ZarrV3ConsolidatedMetadata | UNSET
    extra_fields: dict[str, ZarrV3ExtensionField]

    def __post_init__(self) -> None:
        # The runtime half of the annotations.
        extra = cast("object", self.extra_fields)
        if not isinstance(extra, Mapping):
            msg = f"extra_fields: expected a mapping of names to JSON, got {extra!r}"
            raise TypeError(msg)
        consolidated = cast("object", self.consolidated_metadata)
        if consolidated is not UNSET and not isinstance(consolidated, ZarrV3ConsolidatedMetadata):
            msg = f"consolidated_metadata: expected ZarrV3ConsolidatedMetadata or UNSET, got {consolidated!r}"
            raise TypeError(msg)
        # Its own members, each document its consolidated metadata holds
        # being a model that checked itself, held as the read refines them.
        reading, members = read_group_v3(_own_document(self, whole=True), NO_SCOPE)
        problems = (*overlapping(self.extra_fields, _RESERVED_V3, "group"), *reading.problems)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        own = cast("GroupMembersV3", members)
        object.__setattr__(self, "attributes", own.attributes)
        object.__setattr__(self, "extra_fields", own.extra_fields)

    @classmethod
    def create_default(
        cls,
        *,
        context: Context = CORE_AND_EXTENSIONS,
        **members: Unpack[ZarrV3GroupMetadataJSONPartial],
    ) -> ZarrV3GroupMetadata:
        """A group with no attributes, or the one `members` of its document make of it, read in `context`; `MetadataValidationError` when its document has a problem."""
        return cls.from_json({"zarr_format": 3, "node_type": "group", **members}, context=context)

    def update(
        self, *, context: Context, **members: Unpack[ZarrV3GroupMetadataUpdate]
    ) -> ZarrV3GroupMetadata:
        """This model with `members` in their place, each read in `context`; `UNSET` leaves one out.

        Only the members given are read. A `consolidated_metadata` given
        has each document it holds read in `context`; left out, the group
        keeps the models it holds, however each was read, and reads none
        of them again. `MetadataValidationError` when the document the
        members make has a problem.
        """
        document = {**_own_document(self, whole=True), **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        updated = type(self).from_json(document, context=context)
        if ZARR_V3_CONSOLIDATED_METADATA_KEY in members:
            return updated
        return dataclasses.replace(updated, consolidated_metadata=self.consolidated_metadata)

    def to_json(self) -> ZarrV3GroupMetadataJSON:
        """The document as JSON, sharing no mutable state with the model.

        `attributes` when not empty, `consolidated_metadata` when set, and
        each extra field as held.
        """
        return cast("ZarrV3GroupMetadataJSON", copied(cast("JSONValue", group_json(self))))

    @classmethod
    def from_json(
        cls, data: object, *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3GroupMetadata:
        """The model of `data`, a v3 group document read in `context`, with each document its consolidated metadata holds.

        `MetadataValidationError` with every problem the read finds. A
        `consolidated_metadata` of `null` is read as none, and not written
        back. A member the spec does not define is held in `extra_fields`.
        """
        reading = read_group_metadata_v3(data, context=context)
        if reading.metadata is None:
            raise MetadataValidationError(reading.problems)
        return reading.metadata

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """Extra fields the reader is obligated to understand.

        Everything in `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578). A compliant
        reader MUST fail to open the group if this contains any field it does
        not recognize; the model layer only partitions by obligation, since
        recognition is reader-specific.
        """
        return must_understand_subset(self.extra_fields)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3GroupMetadata:
        """The model of the group document at `zarr.json` in `mapping`, read in `context`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls.from_json(
            load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY), context=context
        )

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3GroupMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        A model, and each it holds, was checked when it was built, so its
        document is written as it is. `NaN`, `Infinity` and `-Infinity` in
        `attributes` are written as those bare tokens, as zarr-python writes
        them, which a strict JSON parser refuses.
        """
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: dump_store_json(group_json(self), indent=indent)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3ConsolidatedMetadata:
    """In-memory model of v3 inline consolidated metadata.

    Models the reference-implementation convention where consolidated metadata
    is embedded as an extension field on a group's `zarr.json`. Each entry in
    `metadata` is the model of a complete child document, array or group.
    `must_understand` is typed permissively as `bool` to mirror the document
    shape, but only `False` is valid; this is enforced at runtime. Each
    document it holds is a model, which checked itself when it was built.
    """

    kind: Literal["inline"] = field(default="inline", init=False)
    must_understand: bool = False
    metadata: dict[str, ZarrV3ArrayMetadata | ZarrV3GroupMetadata]

    def __post_init__(self) -> None:
        if self.must_understand is not False:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("must_understand",),
                        f"Invalid value for 'must_understand'. Expected False. "
                        f"Got {self.must_understand!r}.",
                        "invalid_value",
                    )
                ]
            )
        # The runtime half of the annotations.
        for path, node in cast("dict[object, object]", self.metadata).items():
            if not isinstance(path, str):
                msg = f"metadata: a document's path is a string, got {path!r}"
                raise TypeError(msg)
            if not isinstance(node, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
                msg = f"metadata[{path!r}]: expected a v3 array or group model, got {node!r}"
                raise TypeError(msg)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json(self) -> ZarrV3ConsolidatedMetadataJSON:
        """The `consolidated_metadata` member as JSON, sharing no mutable state with the model: its `kind`, `must_understand: false`, and each document by its path."""
        return cast(
            "ZarrV3ConsolidatedMetadataJSON", copied(cast("JSONValue", consolidated_json(self)))
        )

    @classmethod
    def from_json(
        cls, data: object, *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3ConsolidatedMetadata:
        """The model of `data`, a group's `consolidated_metadata` member, each document read once in `context`, as the array or group its `node_type` says.

        `MetadataValidationError` with every problem found.
        """
        readings, members, problems = _read_consolidated_v3(data, context)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        return construct(cls, metadata=_models(readings, members)[1])


def group_json(model: ZarrV3GroupMetadata) -> ZarrV3GroupMetadataJSON:
    """`model`'s document as JSON, holding the model's own values: what `to_key_value` serializes, which changes nothing, and `to_json` copies."""
    return cast("ZarrV3GroupMetadataJSON", _group_document(model, array_json, group_json))


def consolidated_json(model: ZarrV3ConsolidatedMetadata) -> ZarrV3ConsolidatedMetadataJSON:
    """`model`, a `consolidated_metadata` member, as JSON, holding the model's own values, as `group_json` holds them."""
    return cast(
        "ZarrV3ConsolidatedMetadataJSON", _consolidated_document(model, array_json, group_json)
    )


def _own_document(model: ZarrV3GroupMetadata, *, whole: bool) -> dict[str, object]:
    """`model`'s document without its consolidated metadata: the members that are the group's own.

    Empty `attributes` too when `whole`, as the constructor reads them,
    where a writer leaves them out. An extra field named as a member the
    document declares is no member of it, which the constructor reports.
    """
    out: dict[str, object] = {"zarr_format": model.zarr_format, "node_type": model.node_type}
    if whole or len(model.attributes) > 0:
        out["attributes"] = model.attributes
    out.update((key, value) for key, value in model.extra_fields.items() if key not in _RESERVED_V3)
    return out


_RESERVED_V3: Final = GROUP_METADATA_STANDARD_KEYS_V3 | {ZARR_V3_CONSOLIDATED_METADATA_KEY}
"""The members of a v3 group document the model holds apart from its extra fields."""


def _group_document(
    model: ZarrV3GroupMetadata,
    array: Callable[[ZarrV3ArrayMetadata], object],
    group: Callable[[ZarrV3GroupMetadata], object],
) -> dict[str, object]:
    """`model`'s document, each document its consolidated metadata holds as `array` or `group` gives it."""
    out = _own_document(model, whole=False)
    if model.consolidated_metadata is not UNSET:
        # Consolidated metadata is a known non-core top-level JSON field.
        out[ZARR_V3_CONSOLIDATED_METADATA_KEY] = _consolidated_document(
            model.consolidated_metadata, array, group
        )
    return out


def _consolidated_document(
    model: ZarrV3ConsolidatedMetadata,
    array: Callable[[ZarrV3ArrayMetadata], object],
    group: Callable[[ZarrV3GroupMetadata], object],
) -> dict[str, object]:
    """The member, each document it holds as `array` or `group` gives it."""
    # `must_understand` is emitted as the literal False: the field is typed
    # permissively as `bool`, but `__post_init__` guarantees the value.
    return {
        "kind": model.kind,
        "must_understand": False,
        "metadata": {
            path: array(node) if isinstance(node, ZarrV3ArrayMetadata) else group(node)
            for path, node in model.metadata.items()
        },
    }


def _no_documents() -> Mapping[str, ZarrV3NodeMetadataReading]:
    """What a group whose consolidated metadata holds none, or that has none, holds: nothing."""
    return {}


@dataclass(frozen=True, slots=True)
class ZarrV3GroupMetadataReading:
    """A v3 group document as a scope read it, whatever it holds: each document its consolidated metadata holds, as read, every problem, and the model when there is none."""

    consolidated: Mapping[str, ZarrV3NodeMetadataReading] = dataclasses.field(
        default_factory=_no_documents
    )
    """Each document its consolidated metadata holds, as `read_node_metadata_v3` reads one, by its path."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not a valid one."""
    metadata: ZarrV3GroupMetadata | None = None
    """The document's model when there is no problem; None otherwise."""

    def fields(self) -> Iterator[tuple[Loc, Resolved[Any]]]:
        """Each field of each document its consolidated metadata holds, as read, with where it sits in this document."""
        for path, reading in self.consolidated.items():
            for loc, node in reading.fields():
                yield (ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", path, *loc), node


@dataclass(frozen=True, slots=True)
class ZarrV3UnknownNodeReading:
    """A v3 document of no node type the spec defines -- its `node_type` missing, or neither `"array"` nor `"group"` -- or not an object at all: nothing else of it is read but its `zarr_format`, as its problems say.

    So a document of another format says so: zarr-python 2's draft of v3
    wrote a root `zarr.json` whose `zarr_format` is a URL, and a v2
    document names format 2.
    """

    problems: tuple[ValidationProblem, ...]
    """Why it is no node."""

    @property
    def metadata(self) -> None:
        """Its model: none, since no node type says which model it is."""
        return None

    def fields(self) -> Iterator[tuple[Loc, Resolved[Any]]]:
        """Its fields as read: none, since none of them is read."""
        return iter(())


ZarrV3NodeMetadataReading = TypeAliasType(
    "ZarrV3NodeMetadataReading",
    "ZarrV3ArrayMetadataReading | ZarrV3GroupMetadataReading | ZarrV3UnknownNodeReading",
)
"""A v3 `zarr.json` as `read_node_metadata_v3` reads it: as the array or group its `node_type` says, or as neither."""


def read_node_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3NodeMetadataReading:
    """`value`, a v3 `zarr.json`, read in `context` as the node its `node_type` says it is.

    The node type is the tag of a union, as pydantic's discriminator and
    zod's discriminated union read one: an array is read as
    `read_array_metadata_v3` reads it, a group as `read_group_metadata_v3`
    does, and a document that says neither, or is not an object, is
    `ZarrV3UnknownNodeReading`, with the problems, its `zarr_format`'s
    among them. So no caller reads `node_type` from JSON it has not read,
    and a document of another format says it is not v3.
    """
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_metadata_v3(value, context=context)
    if node_type == "group":
        return read_group_metadata_v3(value, context=context)
    return ZarrV3UnknownNodeReading(problems)


ZarrV3NodeMetadata = TypeAliasType(
    "ZarrV3NodeMetadata", "ZarrV3ArrayMetadata | ZarrV3GroupMetadata"
)
"""The model of a v3 `zarr.json`: an array's or a group's, as its `node_type` says."""


def node_metadata_from_json_v3(
    data: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3NodeMetadata:
    """The model of `data`, a v3 `zarr.json` read in `context`, as the node its `node_type` says.

    What `ZarrV3ArrayMetadata.from_json` or `ZarrV3GroupMetadata.from_json`
    gives, as pydantic's `TypeAdapter` validates a discriminated union.
    `MetadataValidationError` with every problem `read_node_metadata_v3`
    finds, a `node_type` that says neither among them.
    """
    reading = read_node_metadata_v3(data, context=context)
    if reading.metadata is None:
        raise MetadataValidationError(reading.problems)
    return reading.metadata


def node_metadata_from_key_value_v3(
    mapping: Mapping[StoreKey, bytes], *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3NodeMetadata:
    """The model of the document at `zarr.json` in `mapping`, read in `context` as the node its `node_type` says, as `node_metadata_from_json_v3` reads one.

    `MetadataValidationError` when the key is missing, its bytes are not
    JSON, or the document is not a valid array or group.
    """
    # An array's document and a group's are both at `zarr.json`.
    document = load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY)
    return node_metadata_from_json_v3(document, context=context)


def validate_node_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 `zarr.json`: those `validate_array_metadata_v3` or `validate_group_metadata_v3` finds in the node its `node_type` says it is, or why it says neither."""
    return _read_node_v3(value, context)[0].problems


def _read_node_v3(
    value: object, context: Context
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """`value` read as `read_node_metadata_v3` reads it, without models, and its members refined."""
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_v3(value, context)
    if node_type == "group":
        return read_group_v3(value, context)
    return ZarrV3UnknownNodeReading(problems), None


_NODE_TYPES: Final = ("array", "group")
"""The node types the spec defines."""


def _node_type(value: object) -> tuple[str | None, tuple[ValidationProblem, ...]]:
    """The node type `value` says it is, one of `_NODE_TYPES`; None, with the problems, when it says none of them, or is not an object.

    A document that says none is judged by its `zarr_format` too, so one
    of another format says it is not v3.
    """
    if not isinstance(value, Mapping):
        return None, not_an_object(value)
    document = cast("Mapping[object, object]", value)
    node_type = document.get("node_type")
    if isinstance(node_type, str) and node_type in _NODE_TYPES:
        return node_type, ()
    problems = [
        *missing_keys(frozenset({"zarr_format"}), document),
        *check_literal(document, "zarr_format", 3),
    ]
    if "node_type" not in document:
        problems.append(ValidationProblem(("node_type",), "missing required key", "missing_key"))
    else:
        problems.append(outside_of(("node_type",), node_type, _NODE_TYPES))
    return None, with_input(problems, document)


@dataclass(frozen=True, slots=True)
class GroupMembersV3:
    """What a read refined of a v3 group document, as the models hold it: its own members, and those of each document its consolidated metadata holds that a model can be built of."""

    attributes: dict[str, JSONValue] | None
    """Its attributes; None when they have a problem."""
    extra_fields: dict[str, JSONValue]
    """Each member the spec does not define that is JSON."""
    consolidated: Mapping[str, ArrayMembersV3 | GroupMembersV3] | UNSET
    """Each array its consolidated metadata holds that has no problem, and each group, by path; UNSET when it holds none."""


def read_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3GroupMetadataReading:
    """`value`, a v3 group document, as `context` read it, whatever it holds.

    Everything a read finds, in one: each document its consolidated
    metadata holds, read once, as `read_array_metadata_v3` and this read
    one; every problem `validate_group_metadata_v3` finds; and, when there
    is none, the group's model, whose consolidated metadata holds the
    models of those documents, which their readings hold too. A value that
    is not an object holds nothing.
    """
    reading, members = read_group_v3(value, context)
    if members is None:
        return reading
    return _with_models(reading, members)


def read_group_v3(
    value: object, context: Context
) -> tuple[ZarrV3GroupMetadataReading, GroupMembersV3 | None]:
    """`value`, a v3 group document, as `context` read it, without models, and its members refined; None when it is not an object.

    A field a scope already read -- a model's own, handed back -- is taken
    as it is, as `read_array_v3` takes one.
    """
    if not isinstance(value, Mapping):
        return ZarrV3GroupMetadataReading(problems=not_an_object(value)), None
    doc = cast("Mapping[object, object]", value)
    found: list[ValidationProblem] = list(missing_keys(GROUP_METADATA_REQUIRED_KEYS_V3, doc))
    extra_fields, others = other_members(
        doc,
        GROUP_METADATA_STANDARD_KEYS_V3,
        additional_reserved_keys=frozenset({ZARR_V3_CONSOLIDATED_METADATA_KEY}),
    )
    found.extend(others)
    found.extend(check_literal(doc, "zarr_format", 3))
    found.extend(check_literal(doc, "node_type", "group"))
    attributes: dict[str, JSONValue] | None = {}
    if "attributes" in doc:
        attributes, problems = attributes_of(doc["attributes"])
        found.extend(problems)
    # consolidated_metadata: null, which a historical zarr-python bug wrote,
    # is read as none, so those stores stay readable; the model never
    # writes it back.
    raw = doc.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    consolidated: dict[str, ZarrV3NodeMetadataReading] = {}
    held: Mapping[str, ArrayMembersV3 | GroupMembersV3] | UNSET = UNSET
    if raw is not None:
        consolidated, held, inside = _read_consolidated_v3(raw, context)
        found.extend(_prefix(ZARR_V3_CONSOLIDATED_METADATA_KEY, inside))
    reading = ZarrV3GroupMetadataReading(consolidated, with_input(found, doc))
    return reading, GroupMembersV3(attributes, extra_fields, held)


_CONSOLIDATED_MEMBERS: Final = ("kind", "must_understand", "metadata")
"""The members of an inline `consolidated_metadata`, in the order the convention declares them."""


def _read_consolidated_v3(
    value: object, context: Context
) -> tuple[
    dict[str, ZarrV3NodeMetadataReading],
    dict[str, ArrayMembersV3 | GroupMembersV3],
    tuple[ValidationProblem, ...],
]:
    """An inline `consolidated_metadata` member, as `context` read it: each document it holds, read once, as `read_node_metadata_v3` reads one, by its path; the members of each a model can be built of; and every problem, located in the member."""
    if not isinstance(value, Mapping):
        return {}, {}, (ValidationProblem((), "expected an object", "invalid_type"),)
    env = cast("Mapping[object, object]", value)
    # Missing members are reported in the order the envelope declares them.
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in _CONSOLIDATED_MEMBERS
        if key not in env
    ]
    problems.extend(unexpected_keys(frozenset(_CONSOLIDATED_MEMBERS), env))
    problems.extend(check_literal(env, "kind", "inline"))
    problems.extend(check_literal(env, "must_understand", False))
    readings: dict[str, ZarrV3NodeMetadataReading] = {}
    members: dict[str, ArrayMembersV3 | GroupMembersV3] = {}
    entries = env.get("metadata")
    if "metadata" in env and not isinstance(entries, Mapping):
        problems.append(ValidationProblem(("metadata",), "expected an object", "invalid_type"))
    elif isinstance(entries, Mapping):
        for key, entry in cast("Mapping[object, object]", entries).items():
            if not isinstance(key, str):
                problems.append(
                    ValidationProblem(("metadata",), f"non-string key {key!r}", "invalid_type")
                )
                continue
            readings[key], child = _read_node_v3(entry, context)
            if child is not None:
                members[key] = child
            problems.extend(_prefix("metadata", _prefix(key, readings[key].problems)))
    return readings, members, tuple(problems)


def _with_models(
    reading: ZarrV3GroupMetadataReading, members: GroupMembersV3
) -> ZarrV3GroupMetadataReading:
    """`reading`, holding the model of each document its consolidated metadata holds that has no problem, and its own when it has none."""
    readings, models = (
        (reading.consolidated, {})
        if members.consolidated is UNSET
        else _models(reading.consolidated, members.consolidated)
    )
    if len(reading.problems) != 0:
        return dataclasses.replace(reading, consolidated=readings)
    model = construct(
        ZarrV3GroupMetadata,
        attributes=cast("dict[str, JSONValue]", members.attributes),
        consolidated_metadata=(
            UNSET
            if members.consolidated is UNSET
            else construct(ZarrV3ConsolidatedMetadata, metadata=models)
        ),
        extra_fields=members.extra_fields,
    )
    return dataclasses.replace(reading, consolidated=readings, metadata=model)


def _models(
    readings: Mapping[str, ZarrV3NodeMetadataReading],
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3],
) -> tuple[
    dict[str, ZarrV3NodeMetadataReading],
    dict[str, ZarrV3ArrayMetadata | ZarrV3GroupMetadata],
]:
    """Each reading, holding its model when a model can be built of its document, and those models, by path."""
    held: dict[str, ZarrV3NodeMetadataReading] = dict(readings)
    models: dict[str, ZarrV3ArrayMetadata | ZarrV3GroupMetadata] = {}
    for path, child in members.items():
        reading = readings[path]
        if isinstance(reading, ZarrV3ArrayMetadataReading):
            array = array_model(reading, cast("ArrayMembersV3", child))
            held[path], models[path] = dataclasses.replace(reading, metadata=array), array
        elif isinstance(reading, ZarrV3GroupMetadataReading):
            group = _with_models(reading, cast("GroupMembersV3", child))
            held[path] = group
            if group.metadata is not None:
                models[path] = group.metadata
    return held, models


def validate_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 group document.

    Unknown top-level keys are allowed (they map to `extra_fields`); a
    reader must understand each one that does not say `must_understand:
    false`, which the model reports as `must_understand_fields`. A
    `consolidated_metadata` member, if present, is validated too: its
    envelope, and each document it holds by its path, each array read as
    `validate_array_metadata_v3` reads one, in `context`. These are the
    `problems` of `read_group_metadata_v3`, which holds what was read to
    find them.
    """
    return read_group_v3(value, context)[0].problems


def is_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> TypeGuard[ZarrV3GroupMetadataJSON]:
    """Whether `value` is a v3 group document `validate_group_metadata_v3` finds nothing wrong with, written with tuples."""
    return is_canonical_json(value, finite=False) and not validate_group_metadata_v3(
        value, context=context
    )


def parse_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or raise `MetadataValidationError`."""
    problems = validate_group_metadata_v3(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(value))


class ZarrV2GroupMetadataPartial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ZarrV2GroupMetadata`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `ZarrV2GroupMetadata.update` and
    `create_default`. The `init=False` field `zarr_format` is intentionally
    excluded, since it cannot be passed to `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_group.py::test_group_partial_keys_match_settable_model_fields`.
    """

    attributes: dict[str, JSONValue] | UNSET


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV2GroupMetadata:
    """In-memory model of a v2 group metadata document.

    A canonical, lossless representation of the `.zgroup` content plus the
    sibling `.zattrs` attributes, folded into a single in-memory value
    (mirroring the merged `ZarrV2GroupMetadataJSON` document form). `attributes` is
    `UNSET` when no `.zattrs` file (or merged `attributes` key) exists —
    distinct from an explicit empty `.zattrs`, which is `{}` and round-trips
    as a file. A model checks itself when it is built: its document has no
    problem `validate_group_metadata_v2` finds, or the constructor raises
    `MetadataValidationError`.
    """

    zarr_format: Literal[2] = field(default=2, init=False)
    attributes: dict[str, JSONValue] | UNSET

    def __post_init__(self) -> None:
        # Held as a read refines them, in containers of its own.
        object.__setattr__(
            self, "attributes", _v2_attributes(parse_group_metadata_v2(self.to_json()))
        )

    @classmethod
    def create_default(cls, **overrides: Unpack[ZarrV2GroupMetadataPartial]) -> ZarrV2GroupMetadata:
        """
        Create a default (empty) v2 group metadata model, with optional overrides.

        The default is a structurally-valid group with no attributes — the group
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`).
        """
        default = cls(attributes=UNSET)
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[ZarrV2GroupMetadataPartial]) -> ZarrV2GroupMetadata:
        """
        Return a new `ZarrV2GroupMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV2GroupMetadataPartial` can be updated; the fixed `zarr_format`
        is rejected at the type level. Each given field fully replaces its
        previous value. `MetadataValidationError` when the document the
        change makes has a problem, as the model checks itself when it is
        built.
        """
        return dataclasses.replace(self, **kwargs)

    def to_json(self) -> ZarrV2GroupMetadataJSON:
        """Return the merged in-memory document form.

        `attributes` is included when set (even empty). This is not the
        on-disk `.zgroup` content: a conforming `.zgroup` must exclude
        `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313; https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        # to_json output shares no mutable state with the model.
        out: ZarrV2GroupMetadataJSON = {"zarr_format": self.zarr_format}
        if self.attributes is not UNSET:
            out["attributes"] = copy.deepcopy(self.attributes)
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV2GroupMetadata:
        """The model of `data`, a v2 group document with its attributes under `attributes`.

        `MetadataValidationError` with every problem `validate_group_metadata_v2`
        finds. The model shares no mutable state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = copy.deepcopy(parse_group_metadata_v2(data))
        return construct(cls, attributes=_v2_attributes(parsed))

    @classmethod
    def from_key_value(cls, mapping: Mapping[StoreKey, bytes]) -> ZarrV2GroupMetadata:
        """The model of the group at `.zgroup` in `mapping`, with the attributes at `.zattrs` when there is one.

        `MetadataValidationError` when `.zgroup` is missing, bytes are not
        JSON, `.zgroup` holds `attributes`, or the document is not valid.
        """
        zgroup_raw = load_store_json(mapping, ZARR_V2_GROUP_METADATA_STORE_KEY)
        if not isinstance(zgroup_raw, Mapping):
            return cls.from_json(zgroup_raw)
        zgroup = cast("Mapping[str, object]", zgroup_raw)
        if "attributes" in zgroup:
            # A key `.zgroup` does not declare: its attributes are `.zattrs`.
            refused = ValidationProblem(
                ("attributes",), "unexpected key 'attributes'", "unknown_key"
            )
            raise MetadataValidationError(with_input((refused,), zgroup))
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls.from_json({**zgroup, "attributes": zattrs})
        return cls.from_json(zgroup)

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2GroupMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zgroup` without the attributes, and `.zattrs` with them when they are set, even empty.

        A model was checked when it was built, so its document is written
        as it is.
        """
        # Attributes live only in the sibling `.zattrs` file; the `.zgroup`
        # document must exclude them. The `.zattrs` key is present exactly
        # when attributes are set (even empty) — UNSET emits no file.
        document = self.to_json()
        zgroup = {k: v for k, v in document.items() if k != "attributes"}
        out: dict[ZarrV2GroupMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_GROUP_METADATA_STORE_KEY: dump_store_json(zgroup, indent=indent)
        }
        if "attributes" in document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                document["attributes"], indent=indent
            )
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV2ConsolidatedMetadata:
    """In-memory model of a v2 `.zmetadata` document.

    The `metadata` map holds the flat file-keyed entries (`"path/.zarray"`,
    `"path/.zattrs"`, ...) verbatim, preserving the normalized JSON tree.
    Entries are deliberately NOT merged into per-node models: which nodes had
    a `.zattrs` file at all is information the canonical representation must
    keep. Interpreting entries into node models is consumer work. A model
    checks itself when it is built, as `from_json` checks a document, or
    the constructor raises `MetadataValidationError`.
    """

    zarr_consolidated_format: Literal[1] = field(default=1, init=False)
    metadata: dict[str, JSONValue]

    def __post_init__(self) -> None:
        # Held as a read refines them, in containers of its own.
        document = {"zarr_consolidated_format": 1, "metadata": self.metadata}
        refined, problems = _read_consolidated_v2(document)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        object.__setattr__(self, "metadata", refined)

    def to_json(self) -> dict[str, JSONValue]:
        """The `.zmetadata` document as JSON, sharing no mutable state with the model."""
        # to_json output shares no mutable state with the model.
        return {
            "zarr_consolidated_format": self.zarr_consolidated_format,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_json(cls, data: object) -> ZarrV2ConsolidatedMetadata:
        """The model of `data`, a `.zmetadata` document, its entries held as written.

        `MetadataValidationError` with every problem: a member missing or
        unexpected, a format other than 1, an entry that is not JSON. A
        `.zattrs` entry is user data, and may hold `NaN`, `Infinity` and
        `-Infinity`.
        """
        refined, problems = _read_consolidated_v2(data)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        return construct(cls, metadata=refined)

    @classmethod
    def from_key_value(cls, mapping: Mapping[StoreKey, bytes]) -> ZarrV2ConsolidatedMetadata:
        """The model of the document at `.zmetadata` in `mapping`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls.from_json(load_store_json(mapping, ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ConsolidatedMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `.zmetadata`, indented by `indent`.

        A model was checked when it was built, so its document is written
        as it is.
        """
        return {
            ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY: dump_store_json(self.to_json(), indent=indent)
        }


def _read_consolidated_v2(
    data: object,
) -> tuple[dict[str, JSONValue], tuple[ValidationProblem, ...]]:
    """`data`, a `.zmetadata` document: each entry as read, and every problem, located in the document.

    Each entry is the document its key names: a `.zattrs` is user data, and
    any other is JSON by RFC 8259.
    """
    normalized = arrays_to_tuples(data)
    if not isinstance(normalized, Mapping):
        return {}, not_an_object(data)
    doc = cast("Mapping[object, object]", normalized)
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in ("zarr_consolidated_format", "metadata")
        if key not in doc
    ]
    problems.extend(unexpected_keys(frozenset({"zarr_consolidated_format", "metadata"}), doc))
    problems.extend(check_literal(doc, "zarr_consolidated_format", 1))
    refined: dict[str, JSONValue] = {}
    if "metadata" in doc:
        entries = doc["metadata"]
        if not isinstance(entries, Mapping) or not all(
            isinstance(k, str) for k in cast("Mapping[object, object]", entries)
        ):
            problems.append(
                ValidationProblem(
                    ("metadata",), "expected an object with string keys", "invalid_type"
                )
            )
        else:
            for key, value in cast("Mapping[str, object]", entries).items():
                refine = (
                    refine_user_data
                    if key.rsplit("/", 1)[-1] == ZARR_V2_ATTRIBUTES_STORE_KEY
                    else refine_json
                )
                entry, found = refine(value, ("metadata", key))
                problems.extend(found)
                refined[key] = entry
    return refined, with_input(problems, data)


def _v2_attributes(document: ZarrV2GroupMetadataJSON) -> dict[str, JSONValue] | UNSET:
    """The attributes of the v2 group model of `document`, which `parse_group_metadata_v2` gave: `UNSET` when it holds none."""
    return dict(document["attributes"]) if "attributes" in document else UNSET
