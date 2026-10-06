"""In-memory models for Zarr group and consolidated metadata documents."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeGuard, TypeVar, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    is_canonical_json,
    json_text,
    nested_past_the_levels,
    not_an_object,
    outside_of,
    refine_json,
    refine_user_data,
    shown,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._json import prefixed as _prefix
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._array import (
    ZarrV3ArrayMetadata,
    located_conflicts,
    must_understand_subset,
    read_array_metadata_v3,
)
from zarr_metadata.model._validation import (
    GROUP_METADATA_REQUIRED_KEYS_V3,
    GROUP_METADATA_STANDARD_KEYS_V3,
    ArrayMembersV3,
    StoreKey,
    ZarrV3ArrayMetadataReading,
    attributes_of,
    check_literal,
    construct,
    dump_store_json,
    load_store_json,
    members_past_the_levels,
    missing_keys,
    other_members,
    parse_group_metadata_v2,
    read_array_v3,
    unexpected_keys,
)
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.consolidated import ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY
from zarr_metadata.v3._hierarchy import NodeType, hierarchy_problems, path_faults, said
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3._scope import Claims, ScopeConflictError, claims_of
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
    one of the extra items: given, its documents are read in place of the
    document's; left out, the document's are read again as part of the
    whole.
    """

    attributes: Mapping[str, JSONValue] | UNSET


class ZarrV3GroupMetadata:
    """A v3 group document, and the scope it was read in.

    The model is the pair, as `ZarrV3ArrayMetadata` is: `to_json` is the
    document as written, refined, and `context` the scope. `attributes`
    and `extra_fields` are views of what the read refined. The
    `consolidated_metadata` reference-implementation convention is a
    `ZarrV3ConsolidatedMetadata` view of the same pair: each document it
    holds is a model of this scope, built from this one read. Built only
    by reading: the constructor reads `document` in `context` and raises
    `MetadataValidationError` with every problem, a nested document's
    located under `consolidated_metadata.metadata.<path>`.
    """

    __slots__ = (
        "_claims",
        "_consolidated",
        "_context",
        "_document",
        "_key",
        "_members",
        "_reading",
    )

    zarr_format: Final = 3
    node_type: Final = "group"

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_AND_EXTENSIONS if context is None else context
        reading, members = read_group_v3(document, scope)
        if members is None or len(reading.problems) != 0:
            raise MetadataValidationError(reading.problems)
        refined, _ = refine_user_data(document)
        self._adopt(cast("dict[str, JSONValue]", refined), scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3GroupMetadataReading,
        members: GroupMembersV3,
    ) -> ZarrV3GroupMetadata:
        """A model of a document a read found nothing wrong with, holding that reading: no second read."""
        model = object.__new__(cls)
        model._adopt(document, context, reading, members)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3GroupMetadataReading,
        members: GroupMembersV3,
    ) -> None:
        self._document = document
        self._context = context
        self._members = members
        held: Mapping[str, ZarrV3NodeMetadataReading] = reading.consolidated
        if members.consolidated is UNSET:
            self._consolidated: ZarrV3ConsolidatedMetadata | UNSET = UNSET
        else:
            member = cast("dict[str, JSONValue]", document[ZARR_V3_CONSOLIDATED_METADATA_KEY])
            documents = cast("dict[str, JSONValue]", member["metadata"])
            models = _nested_models(documents, context, reading.consolidated, members.consolidated)
            # One model per document, of this scope: each nested reading
            # holds the model the group holds.
            held = {
                path: (models[path].reading if path in models else nested)
                for path, nested in reading.consolidated.items()
            }
            self._consolidated = ZarrV3ConsolidatedMetadata._of(  # pyright: ignore[reportPrivateUsage]
                member, context, models
            )
        # The reading holds the model it built, however the model was built.
        self._reading = dataclasses.replace(reading, consolidated=held, metadata=self)
        self._key = group_key(self)
        self._claims = MappingProxyType(claims_of(reading.fields()))

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
        """The scope the document was read in, which `update` reads new members in."""
        return self._context

    @property
    def reading(self) -> ZarrV3GroupMetadataReading:
        """The document as the scope read it: each document its consolidated metadata holds, as read."""
        return self._reading

    @property
    def claims(self) -> Claims:
        """What the reading claimed of each name the document writes, in the documents it holds, keyed as the scope files it."""
        return self._claims

    def to_json(self) -> ZarrV3GroupMetadataJSON:
        """The document as written, refined, sharing nothing with the model."""
        return cast("ZarrV3GroupMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3GroupMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `NaN`, `Infinity` and `-Infinity` in `attributes` are written as
        those bare tokens, as zarr-python writes them, which a strict JSON
        parser refuses.
        """
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == cast("ZarrV3GroupMetadata", other)._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __reduce__(self) -> tuple[type[ZarrV3GroupMetadata], tuple[object, Context]]:
        # The pair, read again on load.
        return type(self), (self._document, self._context)

    # --- typed views ------------------------------------------------------

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The attributes, a read-only view; empty when the document writes none."""
        return MappingProxyType(cast("dict[str, JSONValue]", self._members.attributes))

    @property
    def extra_fields(self) -> Mapping[str, ZarrV3ExtensionField]:
        """Each member the spec does not define, `consolidated_metadata` apart, by name: a read-only view."""
        return MappingProxyType(self._members.extra_fields)

    @property
    def consolidated_metadata(self) -> ZarrV3ConsolidatedMetadata | UNSET:
        """The `consolidated_metadata` member as a model of this scope; `UNSET` when the document writes none."""
        return self._consolidated

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

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV3GroupMetadataUpdate]) -> ZarrV3GroupMetadata:
        """This model with `members`, JSON, in place of the document's, `UNSET` leaving one out, read in this model's own scope.

        A `consolidated_metadata` given is read; left out, the document's
        is read again as part of the whole. `MetadataValidationError` when
        the document they make has a problem.
        """
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: Context | None = None) -> ZarrV3GroupMetadata:
        """This document read in `context`, whatever that changes; `MetadataValidationError` when it has a problem there. The reading is kept when `context` reads every claim identically."""
        scope = CORE_AND_EXTENSIONS if context is None else context
        if scope.disagreements(self._claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: Context | None = None) -> ZarrV3GroupMetadata:
        """This document read in `context`, which may claim what this scope left unclaimed and contradict nothing.

        `ScopeConflictError` naming each name `context` reads by another
        definition, or by none, and where each sits, in the documents the
        consolidated metadata holds too; `MetadataValidationError` when a
        name `context` claims refuses what was written under it.
        """
        scope = CORE_AND_EXTENSIONS if context is None else context
        found = scope.disagreements(self._claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

    def refines(self, other: ZarrV3GroupMetadata) -> bool:
        """Whether this model holds everything `other` holds: the same attributes and extra fields, and consolidated metadata whose every document refines its counterpart."""
        if type(other) is not type(self):
            return False
        if json_text(dict(self.attributes)) != json_text(dict(other.attributes)):
            return False
        if json_text(dict(self.extra_fields)) != json_text(dict(other.extra_fields)):
            return False
        mine, theirs = self._consolidated, other._consolidated
        if mine is UNSET or theirs is UNSET:
            return mine is UNSET and theirs is UNSET
        return mine.refines(theirs)

    # --- constructors -----------------------------------------------------

    @classmethod
    def create_default(
        cls,
        *,
        context: Context | None = None,
        **members: Unpack[ZarrV3GroupMetadataJSONPartial],
    ) -> ZarrV3GroupMetadata:
        """A group with no attributes, or the one `members` of its document make of it, read in `context`; `MetadataValidationError` when its document has a problem."""
        return cls({"zarr_format": 3, "node_type": "group", **members}, context=context)

    @classmethod
    def from_json(cls, data: object, *, context: Context | None = None) -> ZarrV3GroupMetadata:
        """The model of `data`, a v3 group document read in `context`, with each document its consolidated metadata holds.

        `MetadataValidationError` with every problem the read finds. A
        `consolidated_metadata` of `null`, which a zarr-python 3.0.x bug
        wrote, is a value the document wrote, and no object: a problem,
        as the spec says an object; `read_repaired_node_metadata_v3` reads
        such a store. A member the spec does not define is held in
        `extra_fields`.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
    ) -> ZarrV3GroupMetadata:
        """The model of the group document at `zarr.json` in `mapping`, read in `context`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls(load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY), context=context)


class ZarrV3ConsolidatedMetadata:
    """A group's inline `consolidated_metadata` member, and the scope it was read in.

    Models the reference-implementation convention where consolidated
    metadata is embedded as an extension field on a group's `zarr.json`.
    `metadata` maps each path to the model of the complete document there,
    array or group, of this scope: a view of the group's pair when a group
    holds it, built from the group's one read; or of its own pair, when
    the member is read on its own. `kind` is `inline` and `must_understand`
    `False`, by declaration. The documents and the group make the
    hierarchy below the group, the group its root, each at its node's
    path without the leading `/`: the node at `/a/b` at `a/b`.
    """

    __slots__ = ("_context", "_document", "_key", "_metadata")

    kind: Final = "inline"
    must_understand: Final = False

    def __init__(self, member: object, context: Context | None = None) -> None:
        scope = CORE_AND_EXTENSIONS if context is None else context
        # The member sits under a group's key wherever it is read, so the
        # levels a reader walks are counted from there, as in the group.
        readings, members, problems = _read_consolidated_v3(
            member, scope, (ZARR_V3_CONSOLIDATED_METADATA_KEY,)
        )
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        refined, _ = refine_user_data(member)
        document = cast("dict[str, JSONValue]", refined)
        documents = cast("dict[str, JSONValue]", document["metadata"])
        self._adopt(document, scope, _nested_models(documents, scope, readings, members))

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        metadata: dict[str, ZarrV3NodeMetadata],
    ) -> ZarrV3ConsolidatedMetadata:
        """The member of a group a read found nothing wrong with, holding the models that read built."""
        model = object.__new__(cls)
        model._adopt(document, context, metadata)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: Context,
        metadata: dict[str, ZarrV3NodeMetadata],
    ) -> None:
        self._document = document
        self._context = context
        self._metadata = metadata
        self._key = consolidated_key(self)
        # Hidden from the readings, which hold their own models; see `metadata`.

    @property
    def context(self) -> Context:
        """The scope the documents were read in."""
        return self._context

    @property
    def metadata(self) -> Mapping[str, ZarrV3NodeMetadata]:
        """The model of each document, by its path below the group: a read-only view."""
        return MappingProxyType(self._metadata)

    def to_json(self) -> ZarrV3ConsolidatedMetadataJSON:
        """The member as written, refined, sharing nothing with the model."""
        return cast("ZarrV3ConsolidatedMetadataJSON", copied(self._document))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == cast("ZarrV3ConsolidatedMetadata", other)._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __reduce__(self) -> tuple[type[ZarrV3ConsolidatedMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

    def refines(self, other: ZarrV3ConsolidatedMetadata) -> bool:
        """Whether every document this holds refines the one `other` holds at the same path, and neither holds a path the other does not; False of what is not consolidated metadata."""
        if type(other) is not type(self):
            return False
        if self._metadata.keys() != other._metadata.keys():
            return False
        return all(
            _node_refines(self._metadata[path], other._metadata[path]) for path in self._metadata
        )

    @classmethod
    def from_json(
        cls, data: object, *, context: Context | None = None
    ) -> ZarrV3ConsolidatedMetadata:
        """The model of `data`, a group's `consolidated_metadata` member, each document read once in `context`, as the array or group its `node_type` says; `MetadataValidationError` with every problem found."""
        return cls(data, context=context)


def _node_refines(node: ZarrV3NodeMetadata, other: ZarrV3NodeMetadata) -> bool:
    """Whether `node` refines `other`, as the models of one kind refine each other; models of two kinds do not."""
    if isinstance(node, ZarrV3ArrayMetadata):
        return isinstance(other, ZarrV3ArrayMetadata) and node.refines(other)
    return isinstance(other, ZarrV3GroupMetadata) and node.refines(other)


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
    value: object, context: Context, at: tuple[str | int, ...] = ()
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """`value` read as `read_node_metadata_v3` reads it, without models, and its members refined.

    `at` is where it sits in the document handed in, so the levels a
    reader walks are counted from that one's root: a document past them
    is the problem `_refine` reports for a container there, and not read.
    """
    past = nested_past_the_levels(value, at)
    if past is not None:
        return ZarrV3UnknownNodeReading(within((past,), at)), None
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_v3(value, context, at=at)
    if node_type == "group":
        return read_group_v3(value, context, at=at)
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
    # A document nested past the levels a reader walks refines to nothing:
    # it has problems, and no model is built of it or of what it holds.
    refined, _ = refine_user_data(value)
    document: dict[str, JSONValue] = (
        cast("dict[str, JSONValue]", refined) if isinstance(refined, Mapping) else {}
    )
    return _with_models(reading, members, document, context)


def read_group_v3(
    value: object, context: Context, *, at: tuple[str | int, ...] = ()
) -> tuple[ZarrV3GroupMetadataReading, GroupMembersV3 | None]:
    """`value`, a v3 group document, as `context` read it, without models, and its members refined; None when it is not an object.

    Every value is JSON: a field object built by hand, in a document
    `consolidated_metadata` holds, is not, and is refused as
    `read_array_v3` refuses one it is not told to hold. `at` is where the
    document sits in the one handed in, as `read_array_v3` takes it: each
    document its consolidated metadata holds is read from where it sits,
    so a chain of them is bounded by the levels a reader walks, counted
    from the outermost root.
    """
    if not isinstance(value, Mapping):
        return ZarrV3GroupMetadataReading(problems=not_an_object(value)), None
    doc = cast("Mapping[object, object]", value)
    found: list[ValidationProblem] = list(missing_keys(GROUP_METADATA_REQUIRED_KEYS_V3, doc))
    past = members_past_the_levels(doc, at)
    found.extend(past.values())
    whole, doc = doc, {key: item for key, item in doc.items() if key not in past}
    extra_fields, others = other_members(
        doc,
        GROUP_METADATA_STANDARD_KEYS_V3,
        additional_reserved_keys=frozenset({ZARR_V3_CONSOLIDATED_METADATA_KEY}),
        at=at,
    )
    found.extend(others)
    found.extend(check_literal(doc, "zarr_format", 3))
    found.extend(check_literal(doc, "node_type", "group"))
    attributes: dict[str, JSONValue] | None = {}
    if "attributes" in doc:
        attributes, problems = attributes_of(doc["attributes"], at)
        found.extend(problems)
    raw = doc.get(ZARR_V3_CONSOLIDATED_METADATA_KEY, UNSET)
    consolidated: dict[str, ZarrV3NodeMetadataReading] = {}
    held: Mapping[str, ArrayMembersV3 | GroupMembersV3] | UNSET = UNSET
    if raw is not UNSET:
        consolidated, held, inside = _read_consolidated_v3(
            raw, context, (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY)
        )
        found.extend(_prefix(ZARR_V3_CONSOLIDATED_METADATA_KEY, inside))
    reading = ZarrV3GroupMetadataReading(consolidated, with_input(found, whole))
    return reading, GroupMembersV3(attributes, extra_fields, held)


T = TypeVar("T")

_CONSOLIDATED_MEMBERS: Final = ("kind", "must_understand", "metadata")
"""The members of an inline `consolidated_metadata`, in the order the convention declares them."""


def _read_consolidated_v3(
    value: object, context: Context, at: tuple[str | int, ...] = ()
) -> tuple[
    dict[str, ZarrV3NodeMetadataReading],
    dict[str, ArrayMembersV3 | GroupMembersV3],
    tuple[ValidationProblem, ...],
]:
    """An inline `consolidated_metadata` member, as `context` read it: each document it holds, read once, as `read_node_metadata_v3` reads one, by its path; the members of each a model can be built of; and every problem, located in the member.

    `at` is where the member sits in the document handed in. Each container
    the reader descends into -- the member, its `metadata`, each document
    -- is judged where it sits, as `_refine` judges one, so a chain of
    documents is bounded by the levels a reader walks.
    """
    past = nested_past_the_levels(value, at)
    if past is not None:
        return {}, {}, within((past,), at)
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
    node_types: dict[str, NodeType | None] = {}
    entries = env.get("metadata")
    past = nested_past_the_levels(entries, (*at, "metadata"))
    if "metadata" in env and not isinstance(entries, Mapping):
        problems.append(ValidationProblem(("metadata",), "expected an object", "invalid_type"))
    elif isinstance(entries, Mapping) and past is not None:
        problems.extend(within((past,), at))
    elif isinstance(entries, Mapping):
        for key, entry in cast("Mapping[object, object]", entries).items():
            if not isinstance(key, str):
                problems.append(
                    ValidationProblem(
                        ("metadata",), f"non-string key {shown_key(key)}", "invalid_type"
                    )
                )
                continue
            faults = _key_problems(key)
            problems.extend(faults)
            readings[key], child = _read_node_v3(entry, context, (*at, "metadata", key))
            if child is not None:
                members[key] = child
            problems.extend(_prefix("metadata", _prefix(key, readings[key].problems)))
            if len(faults) == 0:
                node_types[key] = _node_type_of(readings[key])
    problems.extend(_hierarchy_problems(node_types))
    for key in node_types:
        problems.extend(
            _nested_listing_problems(
                key, _reading_listing(readings[key]), node_types, _node_type_of
            )
        )
    return readings, members, tuple(problems)


def group_key(model: ZarrV3GroupMetadata) -> tuple[object, ...]:
    """What `==` and `hash` compare of a v3 group model: its attributes and extra fields as JSON text, and what its consolidated metadata holds, by `consolidated_key`."""
    consolidated = model.consolidated_metadata
    return (
        json_text(dict(model.attributes)),
        UNSET if consolidated is UNSET else consolidated._key,  # pyright: ignore[reportPrivateUsage]
        json_text(dict(model.extra_fields)),
    )


def consolidated_key(model: ZarrV3ConsolidatedMetadata) -> tuple[object, ...]:
    """What `==` and `hash` compare of consolidated metadata: each document's key, by its path, in path order."""
    return tuple(
        (path, node._key)  # pyright: ignore[reportPrivateUsage]
        for path, node in sorted(model.metadata.items(), key=lambda item: item[0])
    )


def _key_problems(key: str) -> list[ValidationProblem]:
    """What keeps `key` from being where consolidated metadata keeps a document, said in one problem at the key.

    Consolidated metadata holds the hierarchy below its group, the group its
    root, and the reference implementation keeps the document of each node
    at the node's path in that hierarchy without its leading `/`: the node
    at `/a/b` at the key `a/b`.
    """
    faults = _below_faults(key)
    if len(faults) == 0:
        return []
    message = f"expected the path of a node below the group, got {shown(key)}, which {said(faults)}"
    return [ValidationProblem(("metadata", key), message, "invalid_value")]


def _nested_listing_problems(
    key: str,
    listing: Mapping[str, T],
    node_types: Mapping[str, NodeType | None],
    node_type: Callable[[T], NodeType | None],
) -> list[ValidationProblem]:
    """What is wrong with `listing`, the own consolidated listing of the group at `key`, against `node_types`, the group's flat listing: each problem at the nested entry.

    The reference implementation lists every node below the group in the
    group's own listing, flat, and gives each group it lists an empty
    listing of its own. So a node a listed group lists is one the group
    lists too, at the joined key, of the same node type: one it lists
    alone would be dropped by the reference reader, and one it lists as
    another type contradicts the tree. Only the listing's own entries are
    judged: what a group listed there lists in turn is that group's own to
    judge, when its document is read. `node_type` says what each entry is,
    so readings and models are judged alike.
    """
    problems: list[ValidationProblem] = []
    for path, entry in listing.items():
        if len(_below_faults(path)) != 0:
            # Its own reader reports a key that is no node's path.
            continue
        joined = f"{key}/{path}"
        here = ("metadata", key, ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", path)
        if joined not in node_types:
            message = (
                f"expected a node the group lists, got {shown(f'/{joined}')}, which "
                f"{shown(f'/{key}')} lists alone"
            )
            problems.append(ValidationProblem(here, message, "invalid_value"))
            continue
        listed, nested = node_types[joined], node_type(entry)
        if listed is not None and nested is not None and listed != nested:
            message = (
                f"expected {_an(listed)}, as the group lists {shown(f'/{joined}')}, "
                f"got {_an(nested)}"
            )
            problems.append(ValidationProblem(here, message, "invalid_value"))
    return problems


def _an(node_type: NodeType) -> str:
    return "an array" if node_type == "array" else "a group"


def _reading_listing(reading: ZarrV3NodeMetadataReading) -> Mapping[str, ZarrV3NodeMetadataReading]:
    """What a reading lists in its own consolidated metadata: nothing, for an array's or one of no node type."""
    if isinstance(reading, ZarrV3GroupMetadataReading):
        return reading.consolidated
    return {}


def _hierarchy_problems(node_types: Mapping[str, NodeType | None]) -> list[ValidationProblem]:
    """What keeps the documents consolidated metadata keeps and its group from making a hierarchy, the group its root, as `hierarchy_problems` judges one: each at the key it is about.

    `node_types` gives the node type of each document by its key, None for
    a document of no node type the spec defines, and holds only keys
    `_key_problems` finds nothing wrong with.
    """
    nodes: dict[str, NodeType | None] = {"/": "group"}
    nodes.update((f"/{key}", node_type) for key, node_type in node_types.items())
    return [
        ValidationProblem(("metadata", cast("str", found.loc[0])[1:]), found.message, found.kind)
        for found in hierarchy_problems(nodes)
    ]


def _node_type_of(reading: ZarrV3NodeMetadataReading) -> NodeType | None:
    """The node type a document says it is, as its reading tells; None when it says none."""
    if isinstance(reading, ZarrV3ArrayMetadataReading):
        return "array"
    if isinstance(reading, ZarrV3GroupMetadataReading):
        return "group"
    return None


def _below_faults(path: str) -> list[str]:
    """What keeps `path` from being the path of a node below a group, relative to the group, each said."""
    if path == "":
        return ["is the group's own"]
    if path.startswith("/"):
        return ['starts with "/"']
    return path_faults(f"/{path}")


def _with_models(
    reading: ZarrV3GroupMetadataReading,
    members: GroupMembersV3,
    document: dict[str, JSONValue],
    context: Context,
) -> ZarrV3GroupMetadataReading:
    """`reading`, holding the model of each document its consolidated metadata holds that has no problem, and its own when it has none: each built from `document`, this read's, and `context`."""
    if len(reading.problems) == 0:
        model = ZarrV3GroupMetadata._of(document, context, reading, members)  # pyright: ignore[reportPrivateUsage]
        return model.reading
    # A document with problems may hold a member that is no object, or
    # whose `metadata` is none: then no document in it was read.
    member = document.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    entries = member.get("metadata") if isinstance(member, Mapping) else None
    if members.consolidated is UNSET or not isinstance(entries, Mapping):
        return reading
    models = _nested_models(
        cast("Mapping[str, JSONValue]", entries),
        context,
        reading.consolidated,
        members.consolidated,
    )
    held = {
        path: (models[path].reading if path in models else nested)
        for path, nested in reading.consolidated.items()
    }
    return dataclasses.replace(reading, consolidated=held)


def _nested_models(
    documents: Mapping[str, JSONValue],
    context: Context,
    readings: Mapping[str, ZarrV3NodeMetadataReading],
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3],
) -> dict[str, ZarrV3NodeMetadata]:
    """The model of each document in `documents`, a `metadata` member's, a model can be built of, in `context`: the one its reading holds already when that is of `context`, else built from its reading and members, not read again."""
    models: dict[str, ZarrV3NodeMetadata] = {}
    for path, child in members.items():
        reading = readings[path]
        held = reading.metadata
        if held is not None and held.context is context:
            models[path] = held
            continue
        document = cast("dict[str, JSONValue]", documents[path])
        if isinstance(reading, ZarrV3ArrayMetadataReading):
            models[path] = ZarrV3ArrayMetadata._of(  # pyright: ignore[reportPrivateUsage]
                document, context, reading, cast("ArrayMembersV3", child)
            )
        elif isinstance(reading, ZarrV3GroupMetadataReading) and len(reading.problems) == 0:
            models[path] = ZarrV3GroupMetadata._of(  # pyright: ignore[reportPrivateUsage]
                document, context, reading, cast("GroupMembersV3", child)
            )
    return models


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

    def __eq__(self, other: object) -> bool:
        """Whether `other` models the same group: the same document, as JSON text, which takes `NaN` for itself; equal models hash alike."""
        if type(other) is not type(self):
            return NotImplemented
        return json_text(self.to_json()) == json_text(cast("ZarrV2GroupMetadata", other).to_json())

    def __hash__(self) -> int:
        return hash(json_text(self.to_json()))

    def to_json(self) -> ZarrV2GroupMetadataJSON:
        """Return the merged in-memory document form.

        `attributes` is included when set (even empty). This is not the
        on-disk `.zgroup` content: a conforming `.zgroup` must exclude
        `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313; https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        # to_json output shares no mutable state with the model: the document
        # is copied whole, one frame for each level of nesting.
        out: ZarrV2GroupMetadataJSON = {"zarr_format": self.zarr_format}
        if self.attributes is not UNSET:
            out["attributes"] = self.attributes
        return cast("ZarrV2GroupMetadataJSON", copied(cast("JSONValue", out)))

    @classmethod
    def from_json(cls, data: object) -> ZarrV2GroupMetadata:
        """The model of `data`, a v2 group document with its attributes under `attributes`.

        `MetadataValidationError` with every problem `validate_group_metadata_v2`
        finds. The model shares no mutable state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = cast(
            "ZarrV2GroupMetadataJSON", copied(cast("JSONValue", parse_group_metadata_v2(data)))
        )
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

    def __eq__(self, other: object) -> bool:
        """Whether `other` holds the same document: the same JSON text; equal ones hash alike."""
        if type(other) is not type(self):
            return NotImplemented
        return json_text(self.to_json()) == json_text(
            cast("ZarrV2ConsolidatedMetadata", other).to_json()
        )

    def __hash__(self) -> int:
        return hash(json_text(self.to_json()))

    def to_json(self) -> dict[str, JSONValue]:
        """The `.zmetadata` document as JSON, sharing no mutable state with the model."""
        # to_json output shares no mutable state with the model.
        return {
            "zarr_consolidated_format": self.zarr_consolidated_format,
            "metadata": copied(self.metadata),
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
    # Each entry is refined below, arrays as tuples, no deeper than a
    # reader walks; the document around them is walked as it is.
    if not isinstance(data, Mapping):
        return {}, not_an_object(data)
    doc = cast("Mapping[object, object]", data)
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
    return refined, with_input(problems, doc)


def _v2_attributes(document: ZarrV2GroupMetadataJSON) -> dict[str, JSONValue] | UNSET:
    """The attributes of the v2 group model of `document`, which `parse_group_metadata_v2` gave: `UNSET` when it holds none."""
    return dict(document["attributes"]) if "attributes" in document else UNSET
