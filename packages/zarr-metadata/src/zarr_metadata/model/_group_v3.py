"""The Zarr v3 group, its consolidated metadata, and the reader of a `zarr.json` of either node type.

These three are one module because they are mutually recursive:
consolidated metadata holds the documents of arrays and groups, each of
those is read as a node of either type, and a group's `update` takes a
consolidated metadata model. Split into modules, they would import each
other.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeGuard, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    frozen,
    is_canonical_json,
    is_object,
    json_text,
    nested_past_the_levels,
    not_an_object,
    outside_of,
    refine_user_data,
    shown,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._json import prefixed as _prefix
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._context import scoped
from zarr_metadata.definition._definition import UnclaimedField, fields_of, held
from zarr_metadata.definition._scope import (
    Claims,
    Conflict,
    ScopeConflictError,
    claims_of,
    definition_said,
    kind_name,
)
from zarr_metadata.definition._scope import refines as refines_field
from zarr_metadata.model._array_v3 import (
    ArrayMembersV3,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataReading,
    array_key_of,
    extra_fields_key,
    members_past_the_levels,
    must_understand_subset,
    no_extra_fields,
    read_array_metadata_v3,
    read_array_v3,
    read_extra_fields_v3,
)
from zarr_metadata.model._keyed import Keyed
from zarr_metadata.model._shared import (
    StoreKey,
    attributes_of,
    check_literal,
    dump_store_json,
    load_store_json,
    located_conflicts,
    missing_keys,
    reading_of,
    unexpected_keys,
)
from zarr_metadata.v3._common import ZarrV3ExtensionField

# A runtime import: `get_type_hints` resolves the public signatures, the guards among them.
from zarr_metadata.v3._context import ZarrV3Context  # noqa: TC001
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3._definition import GroupExtraFieldDefinition
from zarr_metadata.v3._hierarchy import hierarchy_problems, path_faults, said
from zarr_metadata.v3.consolidated import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ZarrV3ConsolidatedMetadataJSON,
)
from zarr_metadata.v3.group import ZARR_V3_GROUP_METADATA_STORE_KEY, ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._hierarchy import NodeType
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import AcceptedField, ResolvedField
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataStoreKey


# The standard top-level keys of a v3 group metadata document. Anything outside
# this set is an extension field.
_GROUP_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__required_keys__
)


_GROUP_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__optional_keys__
)


_GROUP_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    _GROUP_METADATA_REQUIRED_KEYS_V3 | _GROUP_METADATA_OPTIONAL_KEYS_V3
)


ZarrV3NodeMetadataInput = TypeAliasType(
    "ZarrV3NodeMetadataInput",
    "ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON | ZarrV3ArrayMetadata | ZarrV3GroupMetadata",
)
"""An entry of consolidated metadata, at a path, as `ZarrV3ConsolidatedMetadata` accepts it: a document, or a model of one.

An `Input` type is what a constructor accepts. A `JSON` TypedDict is
what a document writes, and an `Update` type is what `update` replaces.
Of these three, only an `Input` may hold a model.
"""


class ZarrV3ConsolidatedMetadataInput(TypedDict, closed=True):
    """The `consolidated_metadata` member as `ZarrV3ConsolidatedMetadata` accepts it: in the form a document writes it, with each entry a document or a node model.

    A node model is accepted when the scope of the consolidated metadata
    reads every one of its claims identically, or claims names the
    model's scope left unclaimed. In the second case the model's document
    is read again in that scope. A node model is refused, with a problem
    at its path, when the two scopes read a name differently, or when the
    scope of the consolidated metadata leaves a claimed name unclaimed.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3NodeMetadataInput]


class ZarrV3GroupMetadataUpdate(TypedDict, total=False, extra_items=ZarrV3ExtensionField | UNSET):
    """The members `ZarrV3GroupMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove it.

    `consolidated_metadata` is typed as the reference implementation
    writes it, or as any JSON object, such as another group's
    `consolidated_metadata`, since a group checks only its envelope. It
    can also be a `ZarrV3ConsolidatedMetadata`, whose document is written.
    A group does not read the documents its consolidated metadata lists,
    so build that model to check them before they are written.
    """

    attributes: Mapping[str, JSONValue] | UNSET
    consolidated_metadata: (
        ZarrV3ConsolidatedMetadataJSON
        | Mapping[str, JSONValue]
        | ZarrV3ConsolidatedMetadata
        | UNSET
    )


class ZarrV3GroupMetadata(Keyed):
    """A v3 group document, and the scope it was read in.

    Like `ZarrV3ArrayMetadata`, the model is this pair. `to_json` returns
    the document as written, refined, and `context` is the scope.
    `attributes` and `consolidated_metadata` are views of the values the
    read refined. Each member the spec does not define is an extra field,
    in `extra_fields`, as the scope read it. `CORE` reads a
    `consolidated_metadata` member as the envelope the core spec fixes,
    and no definition reads the documents the member lists: they follow a
    convention of the reference implementation, not the core spec. To
    read them, pass the member to `ZarrV3ConsolidatedMetadata` or
    `read_consolidated_metadata_v3`, or read one listed document with
    `read_node_metadata_v3`. A model is only built by reading: the
    constructor reads `document` in `context` and raises
    `MetadataValidationError` with every problem.
    """

    __slots__ = ("_claims", "_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final = 3
    node_type: Final = "group"

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded for each name the document writes (its extra fields), keyed by kind and the name the scope files it under. Computed on first access."""
        try:
            return self._claims
        except AttributeError:
            self._claims = MappingProxyType(claims_of(self._reading.fields()))
            return self._claims

    def __init__(self, document: object, context: ZarrV3Context | None = None) -> None:
        context = scoped(context, CORE_AND_EXTENSIONS)
        reading, members = read_group_v3(document, context)
        if members is None or len(reading.problems) != 0:
            raise MetadataValidationError(reading.problems)
        self._adopt(_document_of(members), context, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV3Context,
        reading: ZarrV3GroupMetadataReading,
        members: GroupMembersV3,
    ) -> ZarrV3GroupMetadata:
        """Build a model of a document that a read found no problems with, holding that reading, without reading it again. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, reading, members)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: ZarrV3Context,
        reading: ZarrV3GroupMetadataReading,
        members: GroupMembersV3,
    ) -> None:
        self._document = document
        self._context = context
        self._members = members
        if members.attributes is None:
            msg = "a group model holds members a read found nothing wrong with"
            raise TypeError(msg)
        # The members the model exposes, read-only at every level.
        self._shown = (
            frozen(members.attributes),
            frozen(members.extra_fields),
            UNSET if members.consolidated is UNSET else frozen(members.consolidated),
            MappingProxyType({key: held(extra) for key, extra in reading.extra_fields.items()}),
        )
        # The reading holds this model, no matter how the model was built.
        self._reading = dataclasses.replace(reading, metadata=self)

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> ZarrV3Context:
        """The scope the document was read in. `update` reads new members in this scope."""
        return self._context

    @property
    def reading(self) -> ZarrV3GroupMetadataReading:
        """The document as the scope read it: its extra fields, no problems, and this model."""
        return self._reading

    def to_json(self) -> ZarrV3GroupMetadataJSON:
        """The document as written, refined. The result shares no objects with the model."""
        return cast("ZarrV3GroupMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3GroupMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `NaN`, `Infinity` and `-Infinity` in `attributes`, or in the
        documents the consolidated metadata lists, are written as bare
        tokens, as zarr-python writes them. A strict JSON parser rejects
        these tokens.
        """
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for a v3 group model: its attributes as JSON text, and what each extra field means, as `extra_fields_key` computes it."""
        return (
            json_text(self._members.attributes),
            extra_fields_key(self._reading.extra_fields),
        )

    def __reduce__(self) -> tuple[type[ZarrV3GroupMetadata], tuple[object, ZarrV3Context]]:
        # Pickle the pair, and read it again on load.
        return type(self), (self._document, self._context)

    # --- typed views ------------------------------------------------------

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The attributes, read-only at every level; empty when the document writes none."""
        return self._shown[0]

    @property
    def extra_fields(
        self,
    ) -> Mapping[str, AcceptedField[GroupExtraFieldDefinition[Any]] | UnclaimedField]:
        """Each member the spec does not define, by key, as the scope read it: an `AcceptedField` when a definition in scope claims the key (as `CORE` claims `consolidated_metadata`), and an `UnclaimedField` otherwise."""
        return self._shown[3]

    @property
    def consolidated_metadata(self) -> Mapping[str, JSONValue] | UNSET:
        """The `consolidated_metadata` member as written, refined, and read-only at every level. `UNSET` when the document writes none, or writes one that is not an object, which only a scope that leaves the member unclaimed allows.

        The documents it lists are not read. Pass the member to
        `ZarrV3ConsolidatedMetadata` to read them all, or read one with
        `read_node_metadata_v3`.
        """
        return self._shown[2]

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """The extra fields, by key and as written, that no definition in scope reads and that a reader is obligated to understand.

        Each unclaimed member of `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578). The scope is what
        the reader recognizes, so a compliant reader MUST fail to open the
        group when this is not empty. The model only reports these fields;
        refusing to open is up to the reader.
        """
        return must_understand_subset(self.extra_fields, self._shown[1])

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV3GroupMetadataUpdate]) -> ZarrV3GroupMetadata:
        """Return this model with `members` (as JSON) replacing the document's members, read in this model's own scope. A member set to `UNSET` is removed.

        A `ZarrV3ConsolidatedMetadata` given as `consolidated_metadata` is
        written as its document. Raises `MetadataValidationError` when the
        resulting document has a problem.
        """
        document: dict[str, object] = {**self._document, **_written(members)}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV3Context | None = None) -> ZarrV3GroupMetadata:
        """Return this document read in `context`, whatever that changes. Raises `MetadataValidationError` when the document has a problem in `context`. The existing reading is reused when `context` reads every claim identically."""
        context = scoped(context, CORE_AND_EXTENSIONS)
        if context.disagreements(self.claims).agrees:
            return self._of(self._document, context, self._reading, self._members)
        return type(self)(self._document, context=context)

    def refined_in(self, context: ZarrV3Context | None = None) -> ZarrV3GroupMetadata:
        """Return this document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` when `context` reads an extra field with
        a different definition, or with no definition, naming each one and
        its key. Raises `MetadataValidationError` when a definition that
        `context` claims rejects what was written under its name.
        """
        context = scoped(context, CORE_AND_EXTENSIONS)
        found = context.disagreements(self.claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(context)

    def refines(self, other: ZarrV3GroupMetadata) -> bool:
        """Whether this group holds everything `other` holds: the same attributes, and the same extra fields, each refining its counterpart in the order `refines` defines for fields. False when `other` is not a v3 group."""
        if type(other) is not type(self):
            return False
        if json_text(self._members.attributes) != json_text(other._members.attributes):
            return False
        mine, theirs = self.extra_fields, other.extra_fields
        if mine.keys() != theirs.keys():
            return False
        return all(refines_field(mine[key], theirs[key]) for key in mine)

    # --- constructors -----------------------------------------------------

    @classmethod
    def create_default(
        cls,
        *,
        context: ZarrV3Context | None = None,
        **overrides: Unpack[ZarrV3GroupMetadataUpdate],
    ) -> ZarrV3GroupMetadata:
        """Return a group with no attributes, with any document members in `overrides` applied, read in `context`. `overrides` takes what `update` takes, including `UNSET` to drop a member and a `ZarrV3ConsolidatedMetadata` as `consolidated_metadata`. Raises `MetadataValidationError` when the resulting document has a problem."""
        given = {key: value for key, value in overrides.items() if value is not UNSET}
        return cls({"zarr_format": 3, "node_type": "group", **_written(given)}, context=context)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV3Context | None = None
    ) -> ZarrV3GroupMetadata:
        """Return the model of `data`, a v3 group document read in `context`.

        Raises `MetadataValidationError` with every problem the read finds.
        A `consolidated_metadata` of `null`, which zarr-python 3.0 and 3.1
        wrote, is a value the document wrote that is not an object. This is
        a problem, because the spec requires an object. Use
        `read_repaired_node_metadata_v3` to read such a store. A member the
        spec does not define is kept in `extra_fields`.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: ZarrV3Context | None = None
    ) -> ZarrV3GroupMetadata:
        """Return the model of the group document at `zarr.json` in `mapping`, read in `context`.

        Raises `MetadataValidationError` when the key is missing, its bytes
        are not JSON, or the document is not valid.
        """
        return cls(load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY), context=context)


def _written(members: Mapping[str, object]) -> dict[str, object]:
    """`members`, the members given to `update` or `create_default`, with a `ZarrV3ConsolidatedMetadata` given as `consolidated_metadata` replaced by its document."""
    given = members.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    if not isinstance(given, ZarrV3ConsolidatedMetadata):
        return dict(members)
    return {**members, ZARR_V3_CONSOLIDATED_METADATA_KEY: given.to_json()}


class ZarrV3ConsolidatedMetadata(Keyed):
    """A group's inline `consolidated_metadata` member with every document it lists read, and the scope they were read in.

    Models the reference-implementation convention where consolidated
    metadata is embedded in a group's `zarr.json`. A group's own read
    checks only the member's envelope. This model reads each document the
    member lists, once, as the array or group its `node_type` declares,
    and checks that the documents and the group form a hierarchy with the
    group as its root. Build it to check consolidated metadata before
    `ZarrV3GroupMetadata.update` writes it into a group. `metadata` maps
    each path to the model, in this scope, of the document at that path.
    Each document is stored at its node's path without the leading `/`:
    the node at `/a/b` is at `a/b`. `kind` is `inline` and
    `must_understand` is `False`, as the convention declares. A model is
    only built by reading: the constructor raises
    `MetadataValidationError` with every problem, located in the member.
    """

    __slots__ = ("_context", "_document", "_metadata")

    kind: Final = "inline"
    must_understand: Final = False

    def __init__(self, member: object, context: ZarrV3Context | None = None) -> None:
        context = scoped(context, CORE_AND_EXTENSIONS)
        # The member is always under a group's key, even when read on its
        # own, so nesting depth is counted from there, as in the group.
        readings, members, document, problems = _read_consolidated_v3(
            member, context, (ZARR_V3_CONSOLIDATED_METADATA_KEY,)
        )
        if len(problems) != 0 or document is None:
            raise MetadataValidationError(problems)
        self._adopt(document, context, _nested_models(context, readings, members))

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV3Context,
        metadata: dict[str, ZarrV3NodeMetadata],
    ) -> ZarrV3ConsolidatedMetadata:
        """Build the model of a member that a read found no problems with, holding the models that read built. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, metadata)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: ZarrV3Context,
        metadata: dict[str, ZarrV3NodeMetadata],
    ) -> None:
        self._document = document
        self._context = context
        self._metadata = metadata

    @property
    def reading(self) -> ZarrV3ConsolidatedMetadataReading:
        """The member as the scope read it: no problems, and this model."""
        return ZarrV3ConsolidatedMetadataReading(metadata=self)

    @property
    def context(self) -> ZarrV3Context:
        """The scope the documents were read in."""
        return self._context

    @property
    def metadata(self) -> Mapping[str, ZarrV3NodeMetadata]:
        """A read-only view of the model of each document, keyed by its path below the group."""
        return MappingProxyType(self._metadata)

    def to_json(self) -> ZarrV3ConsolidatedMetadataJSON:
        """The member as written, refined. The result shares no objects with the model."""
        return cast("ZarrV3ConsolidatedMetadataJSON", copied(self._document))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for consolidated metadata: each document's path and key, sorted by path."""
        return tuple(
            (path, node._model_key)
            for path, node in sorted(self.metadata.items(), key=lambda item: item[0])
        )

    def __reduce__(self) -> tuple[type[ZarrV3ConsolidatedMetadata], tuple[object, ZarrV3Context]]:
        return type(self), (self._document, self._context)

    def with_context(self, context: ZarrV3Context | None = None) -> ZarrV3ConsolidatedMetadata:
        """Return this member with every document read in `context`, whatever that changes. Raises `MetadataValidationError` when a document has a problem in `context`. The existing readings are reused when `context` reads every claim of every document identically."""
        context = scoped(context, CORE_AND_EXTENSIONS)
        if all(context.disagreements(node.claims).agrees for node in self._metadata.values()):
            moved = {path: node.with_context(context) for path, node in self._metadata.items()}
            return self._of(self._document, context, moved)
        return type(self)(self._document, context=context)

    def refined_in(self, context: ZarrV3Context | None = None) -> ZarrV3ConsolidatedMetadata:
        """Return this member with every document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` naming each conflict, located at the
        document's path. Raises `MetadataValidationError` when a gain
        exposes a problem.
        """
        context = scoped(context, CORE_AND_EXTENSIONS)
        conflicts = [
            dataclasses.replace(
                conflict, loc=("metadata", path, *(() if conflict.loc is None else conflict.loc))
            )
            for path, node in self._metadata.items()
            for conflict in located_conflicts(
                node.reading.fields(), context.disagreements(node.claims).conflicts
            )
        ]
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return self.with_context(context)

    def refines(self, other: ZarrV3ConsolidatedMetadata) -> bool:
        """Whether every document this holds refines the document `other` holds at the same path, and both hold the same paths. False when `other` is not consolidated metadata."""
        if type(other) is not type(self):
            return False
        if self._metadata.keys() != other._metadata.keys():
            return False
        return all(
            _node_refines(self._metadata[path], other._metadata[path]) for path in self._metadata
        )

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV3Context | None = None
    ) -> ZarrV3ConsolidatedMetadata:
        """Return the model of `data`, a group's `consolidated_metadata` member. Each document is read once in `context`, as the array or group its `node_type` declares. Raises `MetadataValidationError` with every problem found."""
        return cls(data, context=context)


def _node_refines(node: ZarrV3NodeMetadata, other: ZarrV3NodeMetadata) -> bool:
    """Whether `node` refines `other`. Models of the same kind use their own `refines`; an array never refines a group, or the reverse."""
    if isinstance(node, ZarrV3ArrayMetadata):
        return isinstance(other, ZarrV3ArrayMetadata) and node.refines(other)
    return isinstance(other, ZarrV3GroupMetadata) and node.refines(other)


@dataclass(frozen=True, slots=True)
class ZarrV3GroupMetadataReading:
    """A v3 group document as a scope read it: each extra field, every problem, and the model when there are no problems."""

    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV3GroupMetadata | None = None
    """The document's model when there is no problem; None otherwise."""
    extra_fields: Mapping[str, ResolvedField[GroupExtraFieldDefinition[Any]]] = dataclasses.field(
        default_factory=no_extra_fields
    )
    """Each member the spec does not define, as the scope read it, by key: an `UnclaimedField` when no definition in scope claims the key."""

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field in the document, as the scope read it, with its location: the extra fields, each at its key."""
        for key, extra in self.extra_fields.items():
            yield from fields_of(extra, (key,))

    def __reduce__(self) -> str | tuple[object, ...]:
        # Same as the other readings: pickle through the model, when there
        # is one. A read-only view does not pickle, so the extra fields go
        # as a dict.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return (_group_reading, (self.problems, dict(self.extra_fields)))


def _group_reading(
    problems: tuple[ValidationProblem, ...],
    extra_fields: dict[str, ResolvedField[GroupExtraFieldDefinition[Any]]],
) -> ZarrV3GroupMetadataReading:
    """Rebuild a group reading without a model from the values `ZarrV3GroupMetadataReading.__reduce__` returns, with its extra fields held read-only."""
    return ZarrV3GroupMetadataReading(
        problems=problems, extra_fields=MappingProxyType(extra_fields)
    )


@dataclass(frozen=True, slots=True)
class ZarrV3ConsolidatedMetadataReading:
    """A group's `consolidated_metadata` member as a scope read it: every problem, and the model when there are no problems."""

    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the member is not valid."""
    metadata: ZarrV3ConsolidatedMetadata | None = None
    """The member's model when there is no problem; None otherwise."""

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field of each document, as read, located in the member: under `metadata` and the document's path. Nothing when there is no model."""
        if self.metadata is None:
            return
        for path, node in self.metadata.metadata.items():
            for loc, field in node.reading.fields():
                yield ("metadata", path, *loc), field

    def __reduce__(self) -> str | tuple[object, ...]:
        # Same as the other readings: pickle through the model, when there is one.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)


def read_consolidated_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3ConsolidatedMetadataReading:
    """Read `value`, a group's `consolidated_metadata` member, in `context` (`CORE_AND_EXTENSIONS` when none is given), whatever it holds: every problem the read finds, and the model when there are none. This is the form of `ZarrV3ConsolidatedMetadata(value, context)` that returns problems instead of raising them."""
    context = scoped(context, CORE_AND_EXTENSIONS)
    readings, members, document, problems = _read_consolidated_v3(
        value, context, (ZARR_V3_CONSOLIDATED_METADATA_KEY,)
    )
    if len(problems) != 0 or document is None:
        return ZarrV3ConsolidatedMetadataReading(problems=problems)
    model = ZarrV3ConsolidatedMetadata._of(  # pyright: ignore[reportPrivateUsage]
        document, context, _nested_models(context, readings, members)
    )
    return model.reading


@dataclass(frozen=True, slots=True)
class ZarrV3UnknownNodeMetadataReading:
    """A v3 document with no node type the spec defines (its `node_type` is missing, or is neither `"array"` nor `"group"`), or a value that is not an object. Only its `zarr_format` is read, and the problems report it.

    This way, a document in another format is reported as such:
    zarr-python 2's draft of v3 wrote a root `zarr.json` whose
    `zarr_format` is a URL, and a v2 document names format 2.
    """

    problems: tuple[ValidationProblem, ...]
    """Why the document is not a node."""

    @property
    def metadata(self) -> None:
        """Its model: always None, since without a node type there is no model to choose."""
        return None

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Its fields as read: none, since no field is read."""
        return iter(())


ZarrV3NodeMetadataReading = TypeAliasType(
    "ZarrV3NodeMetadataReading",
    "ZarrV3ArrayMetadataReading | ZarrV3GroupMetadataReading | ZarrV3UnknownNodeMetadataReading",
)
"""A v3 `zarr.json` as `read_node_metadata_v3` reads it: as the array or group its `node_type` declares, or as neither."""


def read_node_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3NodeMetadataReading:
    """Read `value`, a v3 `zarr.json`, in `context` as the node type its `node_type` declares.

    The node type is the tag of a union, like pydantic's discriminator
    and zod's discriminated union. An array is read by
    `read_array_metadata_v3`, and a group by `read_group_metadata_v3`. A
    document that declares neither, or a value that is not an object,
    gives a `ZarrV3UnknownNodeMetadataReading` with the problems,
    including any problem with its `zarr_format`. So no caller has to
    read `node_type` from unvalidated JSON, and a document in another
    format is reported as not v3.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_metadata_v3(value, context=context)
    if node_type == "group":
        return read_group_metadata_v3(value, context=context)
    return ZarrV3UnknownNodeMetadataReading(problems)


ZarrV3NodeMetadata = TypeAliasType(
    "ZarrV3NodeMetadata", "ZarrV3ArrayMetadata | ZarrV3GroupMetadata"
)
"""The model of a v3 `zarr.json`: an array model or a group model, as its `node_type` declares."""


def node_metadata_from_json_v3(
    data: object, *, context: ZarrV3Context | None = None
) -> ZarrV3NodeMetadata:
    """Return the model of `data`, a v3 `zarr.json` read in `context`, as the node type its `node_type` declares.

    The result is what `ZarrV3ArrayMetadata.from_json` or
    `ZarrV3GroupMetadata.from_json` returns. This works like pydantic's
    `TypeAdapter` validating a discriminated union. Raises
    `MetadataValidationError` with every problem `read_node_metadata_v3`
    finds, including a `node_type` that is neither array nor group.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    reading = read_node_metadata_v3(data, context=context)
    if reading.metadata is None:
        raise MetadataValidationError(reading.problems)
    return reading.metadata


def node_metadata_from_key_value_v3(
    mapping: Mapping[StoreKey, bytes], *, context: ZarrV3Context | None = None
) -> ZarrV3NodeMetadata:
    """Return the model of the document at `zarr.json` in `mapping`, read in `context` as the node type its `node_type` declares, as `node_metadata_from_json_v3` reads it.

    Raises `MetadataValidationError` when the key is missing, its bytes
    are not JSON, or the document is not a valid array or group.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    # Array documents and group documents are both stored at `zarr.json`.
    document = load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY)
    return node_metadata_from_json_v3(document, context=context)


def validate_node_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 `zarr.json`: the problems `validate_array_metadata_v3` or `validate_group_metadata_v3` finds for the node type its `node_type` declares, or why it declares neither."""
    context = scoped(context, CORE_AND_EXTENSIONS)
    return _read_node_v3(value, context)[0].problems


def _read_node_v3(
    value: object, context: ZarrV3Context, at: tuple[str | int, ...] = ()
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """Read `value` as `read_node_metadata_v3` does, without building models, and return its refined members.

    `at` is the location of `value` within the input document, so nesting
    depth is counted from the root of the input document. A document
    nested too deep is not read. Its problem is the one `_refine` reports
    for a container at that depth.
    """
    past = nested_past_the_levels(value, at)
    if past is not None:
        return ZarrV3UnknownNodeMetadataReading(within((past,), at)), None
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_v3(value, context, at=at)
    if node_type == "group":
        return read_group_v3(value, context, at=at)
    return ZarrV3UnknownNodeMetadataReading(problems), None


_NODE_TYPES: Final = ("array", "group")
"""The node types the spec defines."""


def _node_type(value: object) -> tuple[str | None, tuple[ValidationProblem, ...]]:
    """The node type `value` declares, one of `_NODE_TYPES`. None, with the problems, when it declares none of them or is not an object.

    When a document declares no valid node type, its `zarr_format` is
    checked too, so a document in another format is reported as not v3.
    """
    if not is_object(value):
        return None, not_an_object(value)
    document = value
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
    """The refined members of a v3 group document, as the model holds them."""

    attributes: dict[str, JSONValue] | None
    """The attributes; None when they have a problem."""
    extra_fields: dict[str, JSONValue]
    """Each member the spec does not define that is JSON."""
    consolidated: dict[str, JSONValue] | UNSET
    """The `consolidated_metadata` member, refined, with the documents it lists not read; UNSET when the document writes none, or the member has a problem."""
    document: dict[str, JSONValue] | None
    """The refined document that the model holds: each member as the read refined it, in the order the document writes them; None when the read found a problem."""


def read_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3GroupMetadataReading:
    """Read `value`, a v3 group document, in `context`, whatever it holds: every problem `validate_group_metadata_v3` finds, and the model when there are none.

    The documents listed in its `consolidated_metadata` are not read; see
    `ZarrV3GroupMetadata`. This is the form of
    `ZarrV3GroupMetadata(value, context)` that returns problems instead
    of raising them.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    reading, members = read_group_v3(value, context)
    if members is None or members.document is None:
        return reading
    return ZarrV3GroupMetadata._of(members.document, context, reading, members).reading  # pyright: ignore[reportPrivateUsage]


def read_group_v3(
    value: object, context: ZarrV3Context, *, at: tuple[str | int, ...] = ()
) -> tuple[ZarrV3GroupMetadataReading, GroupMembersV3 | None]:
    """Read `value`, a v3 group document, in `context`, without building a model. Return the reading and the refined members; the members are None when `value` is not an object.

    `at` is the location of this document within the input document, as
    in `read_array_v3`, so nesting depth is counted from the outermost
    root. Each member the spec does not define is read as an extra field
    of the group, by the definition in `context` that claims its key.
    """
    if not is_object(value):
        return ZarrV3GroupMetadataReading(problems=not_an_object(value)), None
    doc = value
    found: list[ValidationProblem] = list(missing_keys(_GROUP_METADATA_REQUIRED_KEYS_V3, doc))
    past = members_past_the_levels(doc, at)
    found.extend(past.values())
    whole, doc = doc, {key: item for key, item in doc.items() if key not in past}
    extra_fields, extra, others = read_extra_fields_v3(
        doc, _GROUP_METADATA_STANDARD_KEYS_V3, GroupExtraFieldDefinition, context, at
    )
    found.extend(others)
    found.extend(check_literal(doc, "zarr_format", 3))
    found.extend(check_literal(doc, "node_type", "group"))
    attributes: dict[str, JSONValue] | None = {}
    if "attributes" in doc:
        attributes, problems = attributes_of(doc["attributes"], at)
        found.extend(problems)
    member = extra_fields.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    consolidated: dict[str, JSONValue] | UNSET = member if isinstance(member, dict) else UNSET
    reading = ZarrV3GroupMetadataReading(
        problems=with_input(found, whole), extra_fields=MappingProxyType(extra)
    )
    document: dict[str, JSONValue] | None = None
    if len(found) == 0 and attributes is not None:
        # The document the model holds: each member as refined above, in
        # the document's order, built from what was read instead of being
        # refined again.
        refined: dict[str, JSONValue] = {
            **extra_fields,
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        }
        document = {key: refined[key] for key in cast("Mapping[str, object]", doc)}
    return reading, GroupMembersV3(attributes, extra_fields, consolidated, document)


_CONSOLIDATED_MEMBERS: Final = ("kind", "must_understand", "metadata")
"""The members of an inline `consolidated_metadata`, in the order the convention declares them."""


_CONSOLIDATED_ENVELOPE: Final[dict[str, JSONValue]] = {"kind": "inline", "must_understand": False}
"""The fixed values the convention declares for the member."""


def _envelope_problems(member: Mapping[object, object]) -> list[ValidationProblem]:
    """Why `member`, a `consolidated_metadata` object, is not the envelope the core spec fixes: `kind` `"inline"`, `must_understand` false and an object at `metadata`, and nothing else. The entries of `metadata` are not checked. Each problem is located in the member."""
    # Missing members are reported in the order the envelope declares them.
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in _CONSOLIDATED_MEMBERS
        if key not in member
    ]
    problems.extend(unexpected_keys(frozenset(_CONSOLIDATED_MEMBERS), member))
    problems.extend(check_literal(member, "kind", "inline"))
    problems.extend(check_literal(member, "must_understand", False))
    if "metadata" in member and not is_object(member["metadata"]):
        problems.append(ValidationProblem(("metadata",), "expected an object", "invalid_type"))
    return problems


def _read_node_model(
    entry: ZarrV3ArrayMetadata | ZarrV3GroupMetadata, context: ZarrV3Context, at: Loc
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """Read `entry`, a node model given in place of a listed document, in `context`.

    Returns the model's own reading when `context` reads every one of its
    claims identically, and a new read of its document when `context`
    claims more. Returns problems when `context` reads a name with a
    different definition, or with none. A model may join consolidated
    metadata when its claims refine into the scope of the consolidated
    metadata. On a gain, the document is read again in that scope, so a
    problem found by a newly claiming definition is reported at its
    location.
    """
    document = entry.to_json()
    found = context.disagreements(entry.claims)
    if len(found.conflicts) != 0:
        # Read the document again in this scope, so the reading belongs to
        # it, like every other reading the consolidated metadata holds.
        # The conflicts become its problems, each at the field's location.
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
        # List the conflicts first: they are the cause of what the re-read finds.
        return _with_problems(reading, (*problems, *reading.problems)), None
    if found.agrees:
        # Use the model's own reading, unless the document is nested too
        # deep at its listed location. In that case, a read from that
        # location reports the problem.
        _, past = refine_user_data(document, at)
        if len(past) != 0:
            return _read_node_v3(document, context, at)
        moved = entry.with_context(context)
        return moved.reading, moved._members  # pyright: ignore[reportPrivateUsage]
    return _read_node_v3(document, context, at)


def _conflict_said(conflict: Conflict, written: str | None) -> str:
    """The message for a conflict between a model entry's scope and the scope of the consolidated metadata. It states the kind, the name as the document writes it, the definition that read it, and how the scope of the consolidated metadata reads it: with another definition (distinguished from the model's when the two print the same), or with none."""
    kind, filed = conflict.key
    name = filed if written is None else written
    claimed = definition_said(conflict.claimed)
    head = (
        "expected a document read in the scope of the consolidated metadata, got a model that "
        f"reads the {kind_name(kind)} {name!r}"
    )
    if conflict.found is None:
        return f"{head} by {claimed}, which that scope leaves unclaimed"
    found = definition_said(conflict.found)
    if claimed == found:
        return f"{head} by another definition than the one that scope reads it by, {found}"
    return f"{head} by {claimed}, which that scope reads by {found}"


def _with_problems(
    reading: ZarrV3NodeMetadataReading, problems: tuple[ValidationProblem, ...]
) -> ZarrV3NodeMetadataReading:
    """`reading`, a read of a document in the scope of the consolidated metadata, with `problems` as its problems and no model."""
    if isinstance(reading, (ZarrV3GroupMetadataReading, ZarrV3ArrayMetadataReading)):
        return dataclasses.replace(reading, problems=problems, metadata=None)
    return dataclasses.replace(reading, problems=problems)


@dataclass(frozen=True, slots=True)
class _Listing:
    """The documents a group listed in consolidated metadata lists in its own `consolidated_metadata`, as read: the reading of each, and the members of each that a model can be built from, keyed by path."""

    readings: Mapping[str, ZarrV3NodeMetadataReading]
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3]


def _read_consolidated_v3(
    value: object, context: ZarrV3Context, at: tuple[str | int, ...] = ()
) -> tuple[
    dict[str, ZarrV3NodeMetadataReading],
    dict[str, ArrayMembersV3 | GroupMembersV3],
    dict[str, JSONValue] | None,
    tuple[ValidationProblem, ...],
]:
    """Read an inline `consolidated_metadata` member in `context`, with every document it lists.

    Returns four values: the reading of each document it holds, read
    once as `read_node_metadata_v3` reads it, keyed by path; the members
    of each document that a model can be built from; the refined member
    (each document as its read refined it) when there are no problems,
    else None; and every problem, located in the member. A listed group
    that has its own listing has that listing read the same way, so that
    it can be checked against this one. `at` is the location of the
    member within the input document. Each container the reader descends
    into (the member, its `metadata`, and each document) is checked at
    its location, as `_refine` checks a container, so a chain of
    documents is bounded by the maximum nesting depth.
    """
    past = nested_past_the_levels(value, at)
    if past is not None:
        return {}, {}, None, within((past,), at)
    if not is_object(value):
        return {}, {}, None, (ValidationProblem((), "expected an object", "invalid_type"),)
    env = value
    problems = _envelope_problems(env)
    readings: dict[str, ZarrV3NodeMetadataReading] = {}
    members: dict[str, ArrayMembersV3 | GroupMembersV3] = {}
    listings: dict[str, _Listing] = {}
    node_types: dict[str, NodeType | None] = {}
    entries = env.get("metadata")
    past = nested_past_the_levels(entries, (*at, "metadata"))
    if is_object(entries) and past is not None:
        problems.extend(within((past,), at))
    elif is_object(entries):
        for key, entry in entries.items():
            if not isinstance(key, str):
                problems.append(
                    ValidationProblem(
                        ("metadata",), f"non-string key {shown_key(key)}", "invalid_type"
                    )
                )
                continue
            faults = _key_problems(key)
            problems.extend(faults)
            here = (*at, "metadata", key)
            if isinstance(entry, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
                readings[key], child = _read_node_model(entry, context, here)
            else:
                readings[key], child = _read_node_v3(entry, context, here)
            if child is not None:
                members[key] = child
            problems.extend(_prefix("metadata", _prefix(key, readings[key].problems)))
            if (
                isinstance(child, GroupMembersV3)
                and child.document is not None
                and child.consolidated is not UNSET
            ):
                # A listed group with no problem of its own, which lists
                # documents too: read its listing, located in its document.
                nested, nested_members, _, inside = _read_consolidated_v3(
                    child.consolidated, context, (*here, ZARR_V3_CONSOLIDATED_METADATA_KEY)
                )
                listings[key] = _Listing(nested, nested_members)
                problems.extend(
                    _prefix(
                        "metadata", _prefix(key, _prefix(ZARR_V3_CONSOLIDATED_METADATA_KEY, inside))
                    )
                )
            if len(faults) == 0:
                node_types[key] = _node_type_of(readings[key])
    problems.extend(_hierarchy_problems(node_types))
    for key, listing in listings.items():
        if key in node_types:
            problems.extend(_nested_listing_problems(key, listing, readings, members, node_types))
    refined: dict[str, JSONValue] | None = None
    if len(problems) == 0:
        # There are no problems, so every entry has members and a
        # document. The envelope has the declared values, in the member's
        # key order.
        documents = {
            path: _document_of(members[path]) for path in cast("Mapping[str, object]", entries)
        }
        held: dict[str, JSONValue] = {**_CONSOLIDATED_ENVELOPE, "metadata": documents}
        refined = {key: held[key] for key in cast("Mapping[str, object]", env)}
    return readings, members, refined, tuple(problems)


def _document_of(members: ArrayMembersV3 | GroupMembersV3) -> dict[str, JSONValue]:
    """The refined document held by `members`, the members of a node that a read found no problems with. Raises `TypeError` for a group's members with a problem, which such a read rules out."""
    if members.document is None:
        msg = "expected the members of a document a read found nothing wrong with"
        raise TypeError(msg)
    return members.document


def _key_problems(key: str) -> list[ValidationProblem]:
    """Why `key` is not a valid key for a document in consolidated metadata, as one problem at the key.

    Consolidated metadata holds the hierarchy below its group, with the
    group as the root. The reference implementation stores the document
    of each node at the node's path in that hierarchy, without the
    leading `/`: the node at `/a/b` is at the key `a/b`.
    """
    faults = _below_faults(key)
    if len(faults) == 0:
        return []
    message = f"expected the path of a node below the group, got {shown(key)}, which {said(faults)}"
    return [ValidationProblem(("metadata", key), message, "invalid_value")]


def _nested_listing_problems(
    key: str,
    listing: _Listing,
    readings: Mapping[str, ZarrV3NodeMetadataReading],
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3],
    node_types: Mapping[str, NodeType | None],
) -> list[ValidationProblem]:
    """Check `listing`, the own consolidated listing of the group at `key`, against the outer group's flat listing (`readings` and their `node_types`). Each problem is located at the nested entry.

    The reference implementation lists every node below the group in the
    group's own listing, as a flat mapping, and gives each group it lists
    an empty listing of its own. So every node that a listed group lists
    must also be listed by the outer group, at the joined key, with the
    same node type and the same document, as each is read. The reference
    reader would drop a node that only the nested listing has. A node
    listed with another type contradicts the tree. A node listed with a
    different document would give a reader two answers for one node.
    Only the entries of this listing are checked. Deeper listings are
    checked when the listing that holds them is read, and a document
    with its own problem is reported for that problem.
    """
    problems: list[ValidationProblem] = []
    for path, entry in listing.readings.items():
        if len(_below_faults(path)) != 0:
            # The nested listing's own read reports a key that is not a node path.
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
        flat, nested = node_types[joined], _node_type_of(entry)
        if flat is not None and nested is not None and flat != nested:
            message = (
                f"expected {_an(flat)}, as the group lists {shown(f'/{joined}')}, got {_an(nested)}"
            )
            problems.append(ValidationProblem(here, message, "invalid_value"))
            continue
        own = listing.members.get(path)
        flat_members = members.get(joined)
        if (
            own is not None
            and flat_members is not None
            and not _read_alike(readings[joined], flat_members, entry, own)
        ):
            message = (
                f"expected the document the group lists at {shown(f'/{joined}')}, got one "
                "that reads otherwise: one node, one document"
            )
            problems.append(ValidationProblem(here, message, "invalid_value"))
    return problems


def _read_alike(
    reading: ZarrV3NodeMetadataReading,
    members: ArrayMembersV3 | GroupMembersV3,
    other: ZarrV3NodeMetadataReading,
    other_members: ArrayMembersV3 | GroupMembersV3,
) -> bool:
    """Whether two documents of one node read the same. Arrays are compared by their model keys. Groups are compared by their attributes and extra fields only; each group's listing is checked separately, at its own location."""
    if isinstance(reading, ZarrV3ArrayMetadataReading) and isinstance(
        other, ZarrV3ArrayMetadataReading
    ):
        if not isinstance(members, ArrayMembersV3) or not isinstance(other_members, ArrayMembersV3):
            return True
        return array_key_of(reading, members) == array_key_of(other, other_members)
    if isinstance(members, GroupMembersV3) and isinstance(other_members, GroupMembersV3):
        return (json_text(members.attributes), json_text(_unlisted(members))) == (
            json_text(other_members.attributes),
            json_text(_unlisted(other_members)),
        )
    return True


def _unlisted(members: GroupMembersV3) -> dict[str, JSONValue]:
    """The extra fields of a group's members, without its own consolidated listing."""
    return {
        key: value
        for key, value in members.extra_fields.items()
        if key != ZARR_V3_CONSOLIDATED_METADATA_KEY
    }


def _an(node_type: NodeType) -> str:
    return "an array" if node_type == "array" else "a group"


def _hierarchy_problems(node_types: Mapping[str, NodeType | None]) -> list[ValidationProblem]:
    """Why the documents in consolidated metadata and their group do not form a valid hierarchy with the group as root, as `hierarchy_problems` checks it. Each problem is at the key it is about.

    `node_types` gives the node type of each document by its key, or None
    for a document with no node type the spec defines. It holds only keys
    that `_key_problems` finds no problems with.
    """
    nodes: dict[str, NodeType | None] = {"/": "group"}
    nodes.update((f"/{key}", node_type) for key, node_type in node_types.items())
    return [
        ValidationProblem(("metadata", cast("str", found.loc[0])[1:]), found.message, found.kind)
        for found in hierarchy_problems(nodes)
    ]


def _node_type_of(reading: ZarrV3NodeMetadataReading) -> NodeType | None:
    """The node type a document declares, as its reading shows; None when it declares none."""
    if isinstance(reading, ZarrV3ArrayMetadataReading):
        return "array"
    if isinstance(reading, ZarrV3GroupMetadataReading):
        return "group"
    return None


def _below_faults(path: str) -> list[str]:
    """Each reason `path` is not a valid path, relative to a group, of a node below that group."""
    if path == "":
        return ["is the group's own"]
    if path.startswith("/"):
        return ['starts with "/"']
    return path_faults(f"/{path}")


def _nested_models(
    context: ZarrV3Context,
    readings: Mapping[str, ZarrV3NodeMetadataReading],
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3],
) -> dict[str, ZarrV3NodeMetadata]:
    """The model, in `context`, of each document in `members` (the members of a `metadata` member) that a model can be built from. Reuses the model its reading already holds when that model is in `context`. Otherwise builds the model from its reading and members, which hold the refined document, without reading it again."""
    models: dict[str, ZarrV3NodeMetadata] = {}
    for path, child in members.items():
        reading = readings[path]
        held = reading.metadata
        if held is not None and held.context is context:
            models[path] = held
            continue
        if isinstance(reading, ZarrV3ArrayMetadataReading) and isinstance(child, ArrayMembersV3):
            models[path] = ZarrV3ArrayMetadata._of(  # pyright: ignore[reportPrivateUsage]
                child.document, context, reading, child
            )
        elif (
            isinstance(reading, ZarrV3GroupMetadataReading)
            and isinstance(child, GroupMembersV3)
            and child.document is not None
        ):
            models[path] = ZarrV3GroupMetadata._of(  # pyright: ignore[reportPrivateUsage]
                child.document, context, reading, child
            )
    return models


def validate_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 group document.

    Unknown top-level keys are allowed (they map to `extra_fields`). A
    reader must understand each one that does not declare
    `must_understand: false`; the model reports these as
    `must_understand_fields`. A `consolidated_metadata` member, if
    present, is checked against the envelope the core spec fixes. The
    documents it lists are not read; `read_consolidated_metadata_v3`
    reads them. The result equals the `problems` of
    `read_group_metadata_v3`.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    return read_group_v3(value, context)[0].problems


def is_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> TypeGuard[ZarrV3GroupMetadataJSON]:
    """Whether `value` is a v3 group document, written with tuples, that `validate_group_metadata_v3` finds no problems with."""
    context = scoped(context, CORE_AND_EXTENSIONS)
    return (
        is_canonical_json(value, finite=False)
        and isinstance(value, dict)
        and not validate_group_metadata_v3(value, context=context)
    )


def parse_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or raise `MetadataValidationError`."""
    context = scoped(context, CORE_AND_EXTENSIONS)
    problems = validate_group_metadata_v3(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(value))
