"""In-memory models for Zarr group and consolidated metadata documents."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypeGuard, TypeVar, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    frozen,
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
    ZarrV2ArrayMetadata,
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
    dump_store_json,
    load_store_json,
    members_past_the_levels,
    missing_keys,
    other_members,
    parse_group_metadata_v2,
    read_array_v2,
    read_array_v3,
    reading_of,
    unexpected_keys,
    validate_group_metadata_v2,
)
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.consolidated import ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY
from zarr_metadata.v3._hierarchy import NodeType, hierarchy_problems, path_faults, said
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3._scope import Claims, Conflict, ScopeConflictError, claims_of, kind_name
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
    from zarr_metadata.v3._definition import Definition, Resolved
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.consolidated import ZarrV3ConsolidatedMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSONPartial, ZarrV3GroupMetadataStoreKey


ZarrV3NodeMetadataInput = TypeAliasType(
    "ZarrV3NodeMetadataInput",
    "ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON | ZarrV3ArrayMetadata | ZarrV3GroupMetadata",
)
"""What consolidated metadata lists at a path when given to a constructor or `update`: a document, or a model of it."""


class ZarrV3ConsolidatedMetadataInput(TypedDict, closed=True):
    """The `consolidated_metadata` member as a constructor or `update` takes it: as a document writes it, each entry a document or a node model.

    A node model is accepted when the group's scope reads every claim of
    it identically, or claims what the model's scope left unclaimed -- it
    is then read again there -- and refused, with a problem at its path,
    where the two scopes read a name differently, or the group's scope
    leaves it unclaimed.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3NodeMetadataInput]


class ZarrV3GroupMetadataUpdate(TypedDict, total=False, extra_items=ZarrV3ExtensionField | UNSET):
    """The members `ZarrV3GroupMetadata.update` puts in place: each as a document writes it, or `UNSET` to leave it out.

    `consolidated_metadata` is given as a document writes it, each entry a
    document or a node model, as `ZarrV3ConsolidatedMetadataInput` says,
    or as another group's `ZarrV3ConsolidatedMetadata`, whose models are
    taken; left out, the document's are read again as part of the whole.
    """

    attributes: Mapping[str, JSONValue] | UNSET
    consolidated_metadata: ZarrV3ConsolidatedMetadataInput | ZarrV3ConsolidatedMetadata | UNSET


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
        "_shown",
    )

    zarr_format: Final = 3
    node_type: Final = "group"

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_AND_EXTENSIONS if context is None else context
        reading, members = read_group_v3(document, scope)
        if members is None or len(reading.problems) != 0:
            raise MetadataValidationError(reading.problems)
        refined, _ = refine_user_data(documents_for(document))
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
        # What the model shows of its members, read-only at every level.
        self._shown = (
            frozen(cast("JSONValue", members.attributes)),
            frozen(cast("JSONValue", members.extra_fields)),
        )
        held: Mapping[str, ZarrV3NodeMetadataReading] = reading.consolidated
        if members.consolidated is UNSET:
            self._consolidated: ZarrV3ConsolidatedMetadata | UNSET = UNSET
        else:
            member = cast("dict[str, JSONValue]", document[ZARR_V3_CONSOLIDATED_METADATA_KEY])
            documents = cast("dict[str, JSONValue]", member["metadata"])
            models = _nested_models(documents, context, reading.consolidated, members.consolidated)
            # One model per document, of this scope: each nested reading
            # holds the model the group holds.
            held = MappingProxyType(
                {
                    path: (models[path].reading if path in models else nested)
                    for path, nested in reading.consolidated.items()
                }
            )
            self._consolidated = ZarrV3ConsolidatedMetadata._of(  # pyright: ignore[reportPrivateUsage]
                member, context, models
            )
        # The reading holds the model it built, however the model was built;
        # what it holds of the nested documents is read-only, as the model is.
        self._reading = dataclasses.replace(
            reading, consolidated=MappingProxyType(dict(held)), metadata=self
        )
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
        """The attributes, read-only at every level; empty when the document writes none."""
        return cast("Mapping[str, JSONValue]", self._shown[0])

    @property
    def extra_fields(self) -> Mapping[str, ZarrV3ExtensionField]:
        """Each member the spec does not define, `consolidated_metadata` apart, by name: read-only at every level."""
        return cast("Mapping[str, ZarrV3ExtensionField]", self._shown[1])

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
        mine, theirs = self._members, other._members
        if json_text(cast("JSONValue", mine.attributes)) != json_text(
            cast("JSONValue", theirs.attributes)
        ):
            return False
        if json_text(mine.extra_fields) != json_text(theirs.extra_fields):
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
        refined, _ = refine_user_data(_member_documents_for(member))
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

    def __reduce__(self) -> tuple[Callable[..., object], tuple[object, ...]]:
        # A reading that holds its model pickles and copies as the model
        # does, and comes back as that model's own reading, so one model
        # per document still; one without is built again from the dict its
        # read-only view views, which pickles where the view does not.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return (_group_reading, (dict(self.consolidated), self.problems, None))


def _group_reading(
    consolidated: dict[str, ZarrV3NodeMetadataReading],
    problems: tuple[ValidationProblem, ...],
    metadata: ZarrV3GroupMetadata | None,
) -> ZarrV3GroupMetadataReading:
    """A group reading built again from what `ZarrV3GroupMetadataReading.__reduce__` gives, its documents held read-only."""
    return ZarrV3GroupMetadataReading(MappingProxyType(consolidated), problems, metadata)


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
    value: object, *, context: Context | None = None
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
    scope = CORE_AND_EXTENSIONS if context is None else context
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_metadata_v3(value, context=scope)
    if node_type == "group":
        return read_group_metadata_v3(value, context=scope)
    return ZarrV3UnknownNodeReading(problems)


ZarrV3NodeMetadata = TypeAliasType(
    "ZarrV3NodeMetadata", "ZarrV3ArrayMetadata | ZarrV3GroupMetadata"
)
"""The model of a v3 `zarr.json`: an array's or a group's, as its `node_type` says."""


def node_metadata_from_json_v3(
    data: object, *, context: Context | None = None
) -> ZarrV3NodeMetadata:
    """The model of `data`, a v3 `zarr.json` read in `context`, as the node its `node_type` says.

    What `ZarrV3ArrayMetadata.from_json` or `ZarrV3GroupMetadata.from_json`
    gives, as pydantic's `TypeAdapter` validates a discriminated union.
    `MetadataValidationError` with every problem `read_node_metadata_v3`
    finds, a `node_type` that says neither among them.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    reading = read_node_metadata_v3(data, context=scope)
    if reading.metadata is None:
        raise MetadataValidationError(reading.problems)
    return reading.metadata


def node_metadata_from_key_value_v3(
    mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
) -> ZarrV3NodeMetadata:
    """The model of the document at `zarr.json` in `mapping`, read in `context` as the node its `node_type` says, as `node_metadata_from_json_v3` reads one.

    `MetadataValidationError` when the key is missing, its bytes are not
    JSON, or the document is not a valid array or group.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    # An array's document and a group's are both at `zarr.json`.
    document = load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY)
    return node_metadata_from_json_v3(document, context=scope)


def validate_node_metadata_v3(
    value: object, *, context: Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 `zarr.json`: those `validate_array_metadata_v3` or `validate_group_metadata_v3` finds in the node its `node_type` says it is, or why it says neither."""
    scope = CORE_AND_EXTENSIONS if context is None else context
    return _read_node_v3(value, scope)[0].problems


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
    value: object, *, context: Context | None = None
) -> ZarrV3GroupMetadataReading:
    """`value`, a v3 group document, as `context` read it, whatever it holds.

    Everything a read finds, in one: each document its consolidated
    metadata holds, read once, as `read_array_metadata_v3` and this read
    one; every problem `validate_group_metadata_v3` finds; and, when there
    is none, the group's model, whose consolidated metadata holds the
    models of those documents, which their readings hold too. A value that
    is not an object holds nothing.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    reading, members = read_group_v3(value, scope)
    if members is None:
        return reading
    return _with_models(reading, members, documents_for(value), scope)


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
    reading = ZarrV3GroupMetadataReading(MappingProxyType(consolidated), with_input(found, whole))
    return reading, GroupMembersV3(attributes, extra_fields, held)


T = TypeVar("T")

_CONSOLIDATED_MEMBERS: Final = ("kind", "must_understand", "metadata")
"""The members of an inline `consolidated_metadata`, in the order the convention declares them."""

_CONSOLIDATED_ENVELOPE: Final[dict[str, object]] = {"kind": "inline", "must_understand": False}
"""What the member declares, by declaration."""


def documents_for(value: object) -> object:
    """`value`, a v3 group document, with each node model its consolidated metadata lists replaced by that model's document, and a `ZarrV3ConsolidatedMetadata` given as the member by the member it holds; `value` itself when it holds none.

    What a group's own document is built from, so a document built of
    models is JSON as any other, each child written as the child wrote it.
    """
    if not isinstance(value, Mapping):
        return value
    document = cast("Mapping[object, object]", value)
    given = document.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    member = _member_documents_for(given)
    if member is given:
        return cast("object", value)
    return {**document, ZARR_V3_CONSOLIDATED_METADATA_KEY: member}


def _member_documents_for(member: object) -> object:
    """A `consolidated_metadata` member with each node model it lists replaced by its document; `member` itself when it lists none."""
    if isinstance(member, ZarrV3ConsolidatedMetadata):
        return member._document  # pyright: ignore[reportPrivateUsage]
    if not isinstance(member, Mapping):
        return member
    entries = cast("Mapping[object, object]", member).get("metadata")
    if not isinstance(entries, Mapping):
        return cast("object", member)
    replaced: dict[object, object] = {}
    changed = False
    for path, entry in cast("Mapping[object, object]", entries).items():
        if isinstance(entry, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
            replaced[path] = entry._document  # pyright: ignore[reportPrivateUsage]
            changed = True
        else:
            # A document listed here may list models of its own.
            replaced[path] = documents_for(entry)
            changed = changed or replaced[path] is not entry
    if not changed:
        return cast("object", member)
    return {**cast("Mapping[object, object]", member), "metadata": replaced}


def _read_node_model(
    entry: ZarrV3ArrayMetadata | ZarrV3GroupMetadata, context: Context, at: Loc
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """`entry`, a node model given where a document is listed, as `context` takes it: its own reading when `context` reads every claim of it identically, a read of its document when `context` claims more, and problems where `context` reads a name otherwise, or by none.

    A model may join a group when its claims refine into the group's
    scope. A gain reads the document again there, so a problem a newly
    claimed definition finds is reported where it sits.
    """
    document = entry._document  # pyright: ignore[reportPrivateUsage]
    found = context.disagreements(entry.claims)
    if len(found.conflicts) != 0:
        # Read again in the group's scope, so the reading is of that scope,
        # as every reading the group holds is; the conflicts are its
        # problems, each where the field sits.
        reading, _ = _read_node_v3(document, context, at)
        written = {loc: field.name for loc, field in entry.reading.fields()}
        problems = tuple(
            ValidationProblem(
                conflict.loc if conflict.loc is not None else (),
                _conflict_said(
                    conflict, None if conflict.loc is None else written.get(conflict.loc)
                ),
                "invalid_value",
            )
            for conflict in located_conflicts(entry.reading.fields(), found.conflicts)
        )
        # The conflicts first: the cause, before what the re-read finds of it.
        return _with_problems(reading, (*problems, *reading.problems)), None
    if found.agrees:
        # The model's own reading, unless the document sits too deep
        # where it is listed, which a read from there reports.
        _, past = refine_user_data(document, at)
        if len(past) != 0:
            return _read_node_v3(document, context, at)
        moved = entry.with_context(context)
        return moved.reading, moved._members  # pyright: ignore[reportPrivateUsage]
    return _read_node_v3(document, context, at)


def _conflict_said(conflict: Conflict, written: str | None) -> str:
    """What a conflict between a model entry's scope and the group's says: the kind and the name as the document writes it, what read it, and how the group's scope reads it -- by another definition, told apart from the model's where the two print alike, or by none."""
    kind, filed = conflict.key
    name = filed if written is None else written
    claimed = _definition_said(conflict.claimed)
    head = f"expected a document read in the group's scope, got a model that reads the {kind_name(kind)} {name!r}"
    if conflict.found is None:
        return f"{head} by {claimed}, which the group's scope leaves unclaimed"
    found = _definition_said(conflict.found)
    if claimed == found:
        return f"{head} by another definition than the one the group's scope reads it by, {found}"
    return f"{head} by {claimed}, which the group's scope reads by {found}"


def _definition_said(definition: Definition[Any] | None) -> str:
    """A definition as a message tells it from another of the same name: by the TypedDict its configuration is."""
    if definition is None:
        return "no definition"
    return f"{definition!r} of {definition.configuration.__qualname__}"


def _with_problems(
    reading: ZarrV3NodeMetadataReading, problems: tuple[ValidationProblem, ...]
) -> ZarrV3NodeMetadataReading:
    """`reading`, a read of a document in the group's scope, with `problems` as its problems and no model."""
    if isinstance(reading, (ZarrV3GroupMetadataReading, ZarrV3ArrayMetadataReading)):
        return dataclasses.replace(reading, problems=problems, metadata=None)
    return dataclasses.replace(reading, problems=problems)


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
    if isinstance(value, ZarrV3ConsolidatedMetadata):
        # Another group's member, given whole: its models at their paths.
        value = {**_CONSOLIDATED_ENVELOPE, "metadata": dict(value.metadata)}
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
            if isinstance(entry, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
                readings[key], child = _read_node_model(entry, context, (*at, "metadata", key))
            else:
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
    members = model._members  # pyright: ignore[reportPrivateUsage]
    return (
        json_text(cast("JSONValue", members.attributes)),
        UNSET if consolidated is UNSET else consolidated._key,  # pyright: ignore[reportPrivateUsage]
        json_text(members.extra_fields),
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
    value: object,
    context: Context,
    at: Loc = (),
) -> ZarrV3GroupMetadataReading:
    """`reading`, holding the model of each document its consolidated metadata holds that has no problem, and its own when it has none: each built from `value`, the document this read read, in `context`.

    A document with a problem a reader walks past -- a key that is no
    string, a value that is no JSON, a level past the cap -- refines to
    nothing as a whole; each document its consolidated metadata holds is
    then refined on its own, from where it sits in the document handed in
    -- `at` is where this one sits -- so the ones without a problem still
    have their models, and a listed group with a problem of its own holds
    those of its own listing.
    """
    if len(reading.problems) == 0:
        refined, _ = refine_user_data(value)
        document = cast("dict[str, JSONValue]", refined)
        model = ZarrV3GroupMetadata._of(document, context, reading, members)  # pyright: ignore[reportPrivateUsage]
        return model.reading
    if members.consolidated is UNSET or not isinstance(value, Mapping):
        return reading
    member = cast("Mapping[object, object]", value).get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    if not isinstance(member, Mapping):
        return reading
    entries = cast("Mapping[object, object]", member).get("metadata")
    if not isinstance(entries, Mapping):
        return reading
    held_entries = cast("Mapping[object, object]", entries)
    documents: dict[str, JSONValue] = {}
    for path in members.consolidated:
        if path not in held_entries:
            continue
        entry, problems = refine_user_data(
            held_entries[path], (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", path)
        )
        if len(problems) == 0 and isinstance(entry, Mapping):
            documents[path] = entry
    models = _nested_models(
        documents,
        context,
        reading.consolidated,
        {path: child for path, child in members.consolidated.items() if path in documents},
    )
    held = dict(reading.consolidated)
    for path, child in members.consolidated.items():
        nested = reading.consolidated[path]
        if path in models:
            held[path] = models[path].reading
        elif (
            path in held_entries
            and isinstance(nested, ZarrV3GroupMetadataReading)
            and isinstance(child, GroupMembersV3)
        ):
            # A listed group with a problem of its own -- one the reader
            # refused, or one it walked past -- still holds, in its
            # reading, a model of each document in its own listing that
            # has none.
            held[path] = _with_models(
                nested,
                child,
                held_entries[path],
                context,
                (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", path),
            )
    return dataclasses.replace(reading, consolidated=MappingProxyType(held))


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
    value: object, *, context: Context | None = None
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
    scope = CORE_AND_EXTENSIONS if context is None else context
    return read_group_v3(value, scope)[0].problems


def is_group_metadata_v3(
    value: object, *, context: Context | None = None
) -> TypeGuard[ZarrV3GroupMetadataJSON]:
    """Whether `value` is a v3 group document `validate_group_metadata_v3` finds nothing wrong with, written with tuples."""
    scope = CORE_AND_EXTENSIONS if context is None else context
    return is_canonical_json(value, finite=False) and not validate_group_metadata_v3(
        value, context=scope
    )


def parse_group_metadata_v3(
    value: object, *, context: Context | None = None
) -> ZarrV3GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or raise `MetadataValidationError`."""
    scope = CORE_AND_EXTENSIONS if context is None else context
    problems = validate_group_metadata_v3(value, context=scope)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(documents_for(value)))


class ZarrV2GroupMetadataUpdate(TypedDict, total=False):
    """The members `ZarrV2GroupMetadata.update` puts in place: `attributes` as a `.zattrs` writes them, or `UNSET` for no `.zattrs`."""

    attributes: Mapping[str, JSONValue] | UNSET


class ZarrV2GroupMetadata:
    """A v2 group document, and the scope it was read in.

    The pair, as the v3 models are: `to_json` is the merged document --
    `.zgroup`, and `attributes` when a `.zattrs` holds them -- as written,
    refined. `attributes` is `UNSET` when no `.zattrs` exists, distinct
    from an empty one. A group holds no field a scope reads, so the scope
    is held for uniformity: `update` reads new attributes in it, and no
    other scope conflicts with the reading. Built only by reading: the
    constructor raises `MetadataValidationError` with every problem. Two
    groups are equal when their attributes are written alike, as
    `group_key_v2` says.
    """

    __slots__ = ("_attributes", "_context", "_document", "_key", "_shown")

    _attributes: dict[str, JSONValue] | UNSET
    _context: Context
    _document: dict[str, JSONValue]
    _key: tuple[object, ...]
    _shown: object

    zarr_format: Final = 2

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_V2 if context is None else context
        parsed = parse_group_metadata_v2(document, context=scope)
        refined, _ = refine_user_data(document)
        self._adopt(cast("dict[str, JSONValue]", refined), scope, _v2_attributes(parsed))

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        attributes: dict[str, JSONValue] | UNSET,
    ) -> ZarrV2GroupMetadata:
        """A model of a document a read found nothing wrong with: no second read."""
        model = object.__new__(cls)
        model._adopt(document, context, attributes)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: Context,
        attributes: dict[str, JSONValue] | UNSET,
    ) -> None:
        self._document = document
        self._context = context
        self._attributes = attributes
        self._shown = UNSET if attributes is UNSET else frozen(attributes)
        self._key = group_key_v2(self)

    @property
    def context(self) -> Context:
        """The scope the document was read in, which `update` reads new attributes in."""
        return self._context

    @property
    def claims(self) -> Claims:
        """What the reading claimed: nothing, since a group holds no field."""
        return MappingProxyType({})

    @property
    def attributes(self) -> Mapping[str, JSONValue] | UNSET:
        """The user attributes a `.zattrs` holds, read-only at every level; `UNSET` when there is no `.zattrs`."""
        return cast("Mapping[str, JSONValue] | UNSET", self._shown)

    def to_json(self) -> ZarrV2GroupMetadataJSON:
        """The merged document as written, refined, sharing nothing with the model.

        `attributes` is included when set, even empty. This is not the
        on-disk `.zgroup`, which excludes them: `to_key_value` splits the
        document as a store holds it
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313; https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        return cast("ZarrV2GroupMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2GroupMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zgroup` without the attributes, and `.zattrs` with them when they are set, even empty."""
        zgroup = {key: value for key, value in self._document.items() if key != "attributes"}
        out: dict[ZarrV2GroupMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_GROUP_METADATA_STORE_KEY: dump_store_json(zgroup, indent=indent)
        }
        if "attributes" in self._document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                self._document["attributes"], indent=indent
            )
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == cast("ZarrV2GroupMetadata", other)._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __reduce__(self) -> tuple[type[ZarrV2GroupMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

    def update(self, **members: Unpack[ZarrV2GroupMetadataUpdate]) -> ZarrV2GroupMetadata:
        """This model with `attributes` in place of the document's, `UNSET` leaving them out, read in this model's own scope; `MetadataValidationError` when the document they make has a problem."""
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: Context | None = None) -> ZarrV2GroupMetadata:
        """This document read in `context`: the same group, holding that scope."""
        scope = CORE_V2 if context is None else context
        return self._of(self._document, scope, self._attributes)

    def refined_in(self, context: Context | None = None) -> ZarrV2GroupMetadata:
        """This document read in `context`: a group holds no field, so no scope conflicts with its reading, and this is `with_context`."""
        return self.with_context(context)

    def refines(self, other: ZarrV2GroupMetadata) -> bool:
        """Whether this group holds everything `other` holds: its attributes written alike; False of what is not a v2 group."""
        return type(other) is type(self) and self._key == other._key

    @classmethod
    def create_default(
        cls, *, context: Context | None = None, **overrides: Unpack[ZarrV2GroupMetadataUpdate]
    ) -> ZarrV2GroupMetadata:
        """A group with no `.zattrs`, or the one `overrides` make of it, read in `context`; `MetadataValidationError` when the document they make has a problem."""
        given = {key: value for key, value in overrides.items() if value is not UNSET}
        return cls({"zarr_format": 2, **given}, context=context)

    @classmethod
    def from_json(cls, data: object, *, context: Context | None = None) -> ZarrV2GroupMetadata:
        """The model of `data`, a v2 group document with its attributes under `attributes`, read in `context`; `MetadataValidationError` with every problem."""
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
    ) -> ZarrV2GroupMetadata:
        """The model of the group at `.zgroup` in `mapping`, with the attributes at `.zattrs` when there is one, read in `context`.

        `MetadataValidationError` when `.zgroup` is missing, bytes are not
        JSON, `.zgroup` holds `attributes`, or the document is not valid.
        """
        zgroup_raw = load_store_json(mapping, ZARR_V2_GROUP_METADATA_STORE_KEY)
        if not isinstance(zgroup_raw, Mapping):
            return cls(zgroup_raw, context=context)
        zgroup = cast("Mapping[str, object]", zgroup_raw)
        if "attributes" in zgroup:
            # A key `.zgroup` does not declare: its attributes are `.zattrs`.
            refused = ValidationProblem(
                ("attributes",), "unexpected key 'attributes'", "unknown_key"
            )
            raise MetadataValidationError(with_input((refused,), zgroup))
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls({**zgroup, "attributes": zattrs}, context=context)
        return cls(zgroup, context=context)


def group_key_v2(model: ZarrV2GroupMetadata) -> tuple[object, ...]:
    """What `==` and `hash` compare of a v2 group model: its attributes as JSON text, or `UNSET` when there is no `.zattrs`."""
    attributes = model._attributes  # pyright: ignore[reportPrivateUsage]
    return (UNSET if attributes is UNSET else json_text(attributes),)


ZarrV2NodeMetadata: TypeAlias = "ZarrV2ArrayMetadata | ZarrV2GroupMetadata"
"""The model of one node a v2 `.zmetadata` document holds: an array, or a group."""


class ZarrV2ConsolidatedMetadata:
    """A v2 `.zmetadata` document, and the scope its nodes were read in.

    `metadata` holds the flat file-keyed entries (`"path/.zarray"`,
    `"path/.zattrs"`, ...) as written, refined: which nodes had a
    `.zattrs` at all is kept. `nodes` is each `.zarray` or `.zgroup`
    entry, merged with its sibling `.zattrs`, as a model of this scope,
    keyed by the node's path, `""` for the root; a `.zattrs` with no
    sibling is kept and makes no node, and any other entry is JSON, kept.
    Built only by reading: the constructor raises `MetadataValidationError`
    with every problem, each located under its entry. Two documents are
    equal when each node means the same and the other entries are written
    alike, as `consolidated_key_v2` says; `refines`, `with_context` and
    `refined_in` go through the nodes.
    """

    __slots__ = ("_context", "_document", "_key", "_nodes", "_shown")

    _context: Context
    _document: dict[str, JSONValue]
    _key: tuple[object, ...]
    _nodes: dict[str, ZarrV2NodeMetadata]
    _shown: object

    zarr_consolidated_format: Final = 1

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_V2 if context is None else context
        entries, nodes, problems = _read_consolidated_v2(document, scope)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        self._adopt({"zarr_consolidated_format": 1, "metadata": entries}, scope, nodes)

    @classmethod
    def _of(
        cls, document: dict[str, JSONValue], context: Context, nodes: dict[str, ZarrV2NodeMetadata]
    ) -> ZarrV2ConsolidatedMetadata:
        """A model of a document a read found nothing wrong with, holding the nodes that read built."""
        model = object.__new__(cls)
        model._adopt(document, context, nodes)
        return model

    def _adopt(
        self, document: dict[str, JSONValue], context: Context, nodes: dict[str, ZarrV2NodeMetadata]
    ) -> None:
        self._document = document
        self._context = context
        self._nodes = nodes
        self._shown = frozen(document["metadata"])
        self._key = consolidated_key_v2(self)

    @property
    def context(self) -> Context:
        """The scope the nodes were read in."""
        return self._context

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        """The entries as written, refined, by store key; read-only at every level."""
        return cast("Mapping[str, JSONValue]", self._shown)

    @property
    def nodes(self) -> Mapping[str, ZarrV2NodeMetadata]:
        """The model of each node, by its path below the root, `""` for the root: a read-only view."""
        return MappingProxyType(self._nodes)

    def to_json(self) -> dict[str, JSONValue]:
        """The `.zmetadata` document as written, refined, sharing nothing with the model."""
        return cast("dict[str, JSONValue]", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ConsolidatedMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `.zmetadata`, indented by `indent`."""
        return {
            ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == cast("ZarrV2ConsolidatedMetadata", other)._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __reduce__(self) -> tuple[type[ZarrV2ConsolidatedMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

    def with_context(self, context: Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """This document with every node read in `context`, whatever that changes; `MetadataValidationError` when a node has a problem there."""
        scope = CORE_V2 if context is None else context
        return type(self)(self._document, context=scope)

    def refined_in(self, context: Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """This document with every node read in `context`, which may claim what this scope left unclaimed and contradict nothing.

        `ScopeConflictError` naming each conflict, located at the node's
        entry; `MetadataValidationError` when a gain surfaces a problem.
        """
        scope = CORE_V2 if context is None else context
        conflicts: list[Conflict] = []
        for path, node in self._nodes.items():
            if not isinstance(node, ZarrV2ArrayMetadata):
                continue
            found = scope.disagreements(node.claims)
            key = (
                f"{path}/{ZARR_V2_ARRAY_METADATA_STORE_KEY}"
                if path != ""
                else ZARR_V2_ARRAY_METADATA_STORE_KEY
            )
            conflicts.extend(
                dataclasses.replace(
                    conflict,
                    loc=("metadata", key, *(() if conflict.loc is None else conflict.loc)),
                )
                for conflict in located_conflicts(node.reading.fields(), found.conflicts)
            )
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return self.with_context(scope)

    def refines(self, other: ZarrV2ConsolidatedMetadata) -> bool:
        """Whether every node this holds refines the one `other` holds at the same path, neither holds a path the other does not, and the other entries are written alike; False of what is not v2 consolidated metadata."""
        if type(other) is not type(self):
            return False
        if self._nodes.keys() != other._nodes.keys():
            return False
        if _other_entries_text(self) != _other_entries_text(other):
            return False
        return all(_v2_node_refines(self._nodes[path], other._nodes[path]) for path in self._nodes)

    @classmethod
    def from_json(
        cls, data: object, *, context: Context | None = None
    ) -> ZarrV2ConsolidatedMetadata:
        """The model of `data`, a `.zmetadata` document, its nodes read in `context`; `MetadataValidationError` with every problem."""
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
    ) -> ZarrV2ConsolidatedMetadata:
        """The model of the document at `.zmetadata` in `mapping`, read in `context`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls(
            load_store_json(mapping, ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY), context=context
        )


def _v2_node_refines(node: ZarrV2NodeMetadata, other: ZarrV2NodeMetadata) -> bool:
    """Whether `node` refines `other`, as models of one kind refine each other; models of two kinds do not."""
    if isinstance(node, ZarrV2ArrayMetadata):
        return isinstance(other, ZarrV2ArrayMetadata) and node.refines(other)
    return isinstance(other, ZarrV2GroupMetadata) and node.refines(other)


_NODE_FILES: Final = (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    ZARR_V2_ATTRIBUTES_STORE_KEY,
)


def _entries_by_path(entries: Mapping[str, JSONValue]) -> dict[str, dict[str, str]]:
    """The `.zarray`, `.zgroup` and `.zattrs` entries, by node path, then by file: the key each sits under."""
    by_path: dict[str, dict[str, str]] = {}
    for key in entries:
        path, _, name = key.rpartition("/")
        if name in _NODE_FILES:
            by_path.setdefault(path, {})[name] = key
    return by_path


def _other_entries_text(model: ZarrV2ConsolidatedMetadata) -> str:
    """The entries no node is read from -- an orphan `.zattrs`, any other key -- as JSON text: what `==` compares of them."""
    entries = cast("Mapping[str, JSONValue]", model._document["metadata"])  # pyright: ignore[reportPrivateUsage]
    consumed: set[str] = set()
    for path, names in _entries_by_path(entries).items():
        if path in model._nodes:  # pyright: ignore[reportPrivateUsage]
            consumed.update(names.values())
    return json_text({key: value for key, value in entries.items() if key not in consumed})


def consolidated_key_v2(model: ZarrV2ConsolidatedMetadata) -> tuple[object, ...]:
    """What `==` and `hash` compare of v2 consolidated metadata: each node by its path and its own key, and every other entry as JSON text."""
    nodes = model._nodes  # pyright: ignore[reportPrivateUsage]
    return (
        tuple(sorted((path, node._key) for path, node in nodes.items())),  # pyright: ignore[reportPrivateUsage]
        _other_entries_text(model),
    )


def _read_consolidated_v2(
    data: object, context: Context
) -> tuple[dict[str, JSONValue], dict[str, ZarrV2NodeMetadata], tuple[ValidationProblem, ...]]:
    """`data`, a `.zmetadata` document: each entry as read, the model of each node read in `context`, and every problem, located in the document.

    Each entry is the document its key names: a `.zattrs` is user data,
    any other is JSON by RFC 8259; a `.zarray` or `.zgroup` is read as the
    document it is, merged with its sibling `.zattrs`, its problems under
    its entry and the attributes' under the `.zattrs` entry. The nodes are
    empty when anything is wrong.
    """
    if not isinstance(data, Mapping):
        return {}, {}, not_an_object(data)
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
    nodes: dict[str, ZarrV2NodeMetadata] = {}
    if len(problems) == 0:
        for path, names in _entries_by_path(refined).items():
            node, found = _read_node_v2(path, names, refined, context)
            problems.extend(found)
            if node is not None:
                nodes[path] = node
    if len(problems) != 0:
        nodes = {}
    return refined, nodes, with_input(problems, doc)


def _read_node_v2(
    path: str, names: Mapping[str, str], entries: Mapping[str, JSONValue], context: Context
) -> tuple[ZarrV2NodeMetadata | None, list[ValidationProblem]]:
    """The node at `path`, read from its `.zarray` or `.zgroup` entry merged with its `.zattrs`, and every problem, located under the entries; None and no problem when there is only a `.zattrs`."""
    zarray = names.get(ZARR_V2_ARRAY_METADATA_STORE_KEY)
    zgroup = names.get(ZARR_V2_GROUP_METADATA_STORE_KEY)
    zattrs = names.get(ZARR_V2_ATTRIBUTES_STORE_KEY)
    if zarray is not None and zgroup is not None:
        return None, [
            ValidationProblem(
                ("metadata", zgroup),
                f"a node is an array or a group, not both: a {ZARR_V2_ARRAY_METADATA_STORE_KEY} is at "
                f"{path!r} too",
                "invalid_value",
            )
        ]
    key = zarray if zarray is not None else zgroup
    if key is None:
        return None, []
    document = entries[key]
    if not isinstance(document, Mapping):
        return None, [
            ValidationProblem(
                ("metadata", key), f"expected an object, got {shown(document)}", "invalid_type"
            )
        ]
    merged = dict(cast("Mapping[str, JSONValue]", document))
    if "attributes" in merged:
        return None, [
            ValidationProblem(
                ("metadata", key, "attributes"), "unexpected document member", "invalid_value"
            )
        ]
    if zattrs is not None:
        merged["attributes"] = entries[zattrs]

    def located(found: tuple[ValidationProblem, ...]) -> list[ValidationProblem]:
        placed: list[ValidationProblem] = []
        for problem in found:
            if zattrs is not None and problem.loc[:1] == ("attributes",):
                placed.append(
                    dataclasses.replace(problem, loc=("metadata", zattrs, *problem.loc[1:]))
                )
            else:
                placed.append(dataclasses.replace(problem, loc=("metadata", key, *problem.loc)))
        return placed

    if zarray is not None:
        reading, members = read_array_v2(merged, context)
        if members is None:
            return None, located(reading.problems)
        held = merged if "dimension_separator" in merged else {**merged, "dimension_separator": "."}
        return ZarrV2ArrayMetadata._of(held, context, reading, members), []  # pyright: ignore[reportPrivateUsage]
    found = validate_group_metadata_v2(merged, context=context)
    if len(found) != 0:
        return None, located(found)
    attributes: dict[str, JSONValue] | UNSET = (
        dict(cast("Mapping[str, JSONValue]", merged["attributes"]))
        if "attributes" in merged
        else UNSET
    )
    return ZarrV2GroupMetadata._of(merged, context, attributes), []  # pyright: ignore[reportPrivateUsage]


def _v2_attributes(document: ZarrV2GroupMetadataJSON) -> dict[str, JSONValue] | UNSET:
    """The attributes of the v2 group model of `document`, which `parse_group_metadata_v2` gave: `UNSET` when it holds none."""
    return dict(document["attributes"]) if "attributes" in document else UNSET
