"""The Zarr v3 group, its consolidated metadata, and the reader of a `zarr.json` of either node type.

These three are one module because they are mutually recursive: a group
document holds its consolidated metadata, consolidated metadata holds the
documents of arrays and groups, and each of those is read as a node of
either type. Split into modules, they would import each other.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
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
    object_at,
    outside_of,
    refine_user_data,
    shown,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._json import prefixed as _prefix
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._context import Context, ZarrV3Context, scoped
from zarr_metadata.definition._scope import (
    Claims,
    Conflict,
    ScopeConflictError,
    claims_of,
    definition_said,
    kind_name,
)
from zarr_metadata.model._array_v3 import (
    ArrayMembersV3,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataReading,
    array_key_of,
    is_canonical_array_metadata_v3,
    members_past_the_levels,
    must_understand_subset,
    read_array_metadata_v3,
    read_array_v3,
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
    other_members,
    reading_of,
    unexpected_keys,
)
from zarr_metadata.v3._common import ZarrV3ExtensionField
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3._hierarchy import hierarchy_problems, path_faults, said
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY
from zarr_metadata.v3.group import ZARR_V3_GROUP_METADATA_STORE_KEY, ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._hierarchy import NodeType
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import ResolvedField
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.consolidated import ZarrV3ConsolidatedMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSONPartial, ZarrV3GroupMetadataStoreKey


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
"""An entry of consolidated metadata, at a path, as passed to a constructor or `update`: a document, or a model of one.

An `Input` type is what a constructor accepts. A `JSON` TypedDict is
what a document writes, and an `Update` type is what `update` replaces.
Of these three, only an `Input` may hold a model.
"""


class ZarrV3ConsolidatedMetadataInput(TypedDict, closed=True):
    """The `consolidated_metadata` member as a constructor or `update` accepts it: in the form a document writes it, with each entry a document or a node model.

    A node model is accepted when the group's scope reads every one of
    its claims identically, or claims names the model's scope left
    unclaimed. In the second case the model's document is read again in
    the group's scope. A node model is refused, with a problem at its
    path, when the two scopes read a name differently, or when the
    group's scope leaves a claimed name unclaimed.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3NodeMetadataInput]


class ZarrV3GroupMetadataUpdate(TypedDict, total=False, extra_items=ZarrV3ExtensionField | UNSET):
    """The members `ZarrV3GroupMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove it.

    `consolidated_metadata` can be given in the form a document writes
    it, with each entry a document or a node model, as described by
    `ZarrV3ConsolidatedMetadataInput`. It can also be another group's
    `ZarrV3ConsolidatedMetadata`, whose models are used. If it is not
    given, the document's existing member is read again as part of the
    whole document.
    """

    attributes: Mapping[str, JSONValue] | UNSET
    consolidated_metadata: ZarrV3ConsolidatedMetadataInput | ZarrV3ConsolidatedMetadata | UNSET


class ZarrV3GroupMetadata(Keyed):
    """A v3 group document, and the scope it was read in.

    Like `ZarrV3ArrayMetadata`, the model is this pair. `to_json` returns
    the document as written, refined, and `context` is the scope.
    `attributes` and `extra_fields` are views of the values the read
    refined. The `consolidated_metadata` member (a reference-implementation
    convention) is a `ZarrV3ConsolidatedMetadata` view of the same pair:
    each document it holds is a model in this scope, built from this
    single read. A model is only built by reading: the constructor reads
    `document` in `context` and raises `MetadataValidationError` with
    every problem. A problem in a nested document is located under
    `consolidated_metadata.metadata.<path>`.
    """

    __slots__ = (
        "_claims",
        "_consolidated",
        "_context",
        "_document",
        "_members",
        "_reading",
        "_shown",
    )

    zarr_format: Final = 3
    node_type: Final = "group"

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded for each name that the document and its consolidated documents write, keyed by kind and the name the scope files it under. Computed on first access."""
        try:
            return self._claims
        except AttributeError:
            self._claims = MappingProxyType(claims_of(self._reading.fields()))
            return self._claims

    def __init__(self, document: object, context: ZarrV3Context | None = None) -> None:
        scope = scoped(context, CORE_AND_EXTENSIONS)
        reading, members = read_group_v3(document, scope)
        if members is None or len(reading.problems) != 0:
            raise MetadataValidationError(reading.problems)
        self._adopt(_document_of(members), scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
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
        context: Context,
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
        )
        held: Mapping[str, ZarrV3NodeMetadataReading] = reading.consolidated
        if members.consolidated is UNSET:
            self._consolidated: ZarrV3ConsolidatedMetadata | UNSET = UNSET
        else:
            member = object_at(document, ZARR_V3_CONSOLIDATED_METADATA_KEY)
            models = _nested_models(context, reading.consolidated, members.consolidated)
            # One model per document, in this scope: each nested reading
            # holds the same model the group holds.
            held = MappingProxyType(
                {
                    path: (models[path].reading if path in models else nested)
                    for path, nested in reading.consolidated.items()
                }
            )
            self._consolidated = ZarrV3ConsolidatedMetadata._of(  # pyright: ignore[reportPrivateUsage]
                member, context, models
            )
        # The reading holds this model, no matter how the model was built.
        # Its nested document readings are read-only, like the model.
        self._reading = dataclasses.replace(
            reading, consolidated=MappingProxyType(dict(held)), metadata=self
        )

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
        """The scope the document was read in. `update` reads new members in this scope."""
        return self._context

    @property
    def reading(self) -> ZarrV3GroupMetadataReading:
        """The document as the scope read it, including the reading of each document in its consolidated metadata."""
        return self._reading

    def to_json(self) -> ZarrV3GroupMetadataJSON:
        """The document as written, refined. The result shares no objects with the model."""
        return cast("ZarrV3GroupMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3GroupMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `NaN`, `Infinity` and `-Infinity` in `attributes` are written as
        bare tokens, as zarr-python writes them. A strict JSON parser
        rejects these tokens.
        """
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for a v3 group model: its attributes and extra fields as JSON text, and the key of its consolidated metadata."""
        consolidated = self.consolidated_metadata
        members = self._members
        return (
            json_text(members.attributes),
            UNSET if consolidated is UNSET else consolidated._model_key,
            json_text(members.extra_fields),
        )

    def __reduce__(self) -> tuple[type[ZarrV3GroupMetadata], tuple[object, Context]]:
        # Pickle the pair, and read it again on load.
        return type(self), (self._document, self._context)

    # --- typed views ------------------------------------------------------

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The attributes, read-only at every level; empty when the document writes none."""
        return self._shown[0]

    @property
    def extra_fields(self) -> Mapping[str, ZarrV3ExtensionField]:
        """Each member the spec does not define, except `consolidated_metadata`, by name. Read-only at every level."""
        return self._shown[1]

    @property
    def consolidated_metadata(self) -> ZarrV3ConsolidatedMetadata | UNSET:
        """The `consolidated_metadata` member as a model in this scope; `UNSET` when the document writes none."""
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
        """Return this model with `members` (as JSON) replacing the document's members, read in this model's own scope. A member set to `UNSET` is removed.

        A given `consolidated_metadata` is read. If it is not given, the
        document's existing member is read again as part of the whole
        document. Raises `MetadataValidationError` when the resulting
        document has a problem.
        """
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV3Context | None = None) -> ZarrV3GroupMetadata:
        """Return this document read in `context`, whatever that changes. Raises `MetadataValidationError` when the document has a problem in `context`. The existing reading is reused when `context` reads every claim identically."""
        scope = scoped(context, CORE_AND_EXTENSIONS)
        if scope.disagreements(self.claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: ZarrV3Context | None = None) -> ZarrV3GroupMetadata:
        """Return this document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` when `context` reads a name with a
        different definition, or with no definition. The error names each
        such name and its location, including names in the documents of
        the consolidated metadata. Raises `MetadataValidationError` when a
        definition that `context` claims rejects what was written under
        its name.
        """
        scope = scoped(context, CORE_AND_EXTENSIONS)
        found = scope.disagreements(self.claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

    def refines(self, other: ZarrV3GroupMetadata) -> bool:
        """Whether this model holds everything `other` holds: the same attributes and extra fields, and consolidated metadata in which every document refines its counterpart."""
        if type(other) is not type(self):
            return False
        mine, theirs = self._members, other._members
        if json_text(mine.attributes) != json_text(theirs.attributes):
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
        context: ZarrV3Context | None = None,
        **members: Unpack[ZarrV3GroupMetadataJSONPartial],
    ) -> ZarrV3GroupMetadata:
        """Return a group with no attributes, with any document members in `members` applied, read in `context`. Raises `MetadataValidationError` when the resulting document has a problem."""
        return cls({"zarr_format": 3, "node_type": "group", **members}, context=context)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV3Context | None = None
    ) -> ZarrV3GroupMetadata:
        """Return the model of `data`, a v3 group document read in `context`, including each document in its consolidated metadata.

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


class ZarrV3ConsolidatedMetadata(Keyed):
    """A group's inline `consolidated_metadata` member, and the scope it was read in.

    Models the reference-implementation convention where consolidated
    metadata is embedded as an extension field on a group's `zarr.json`.
    `metadata` maps each path to the model, in this scope, of the
    complete array or group document at that path. When a group holds
    the member, it is a view of the group's pair, built from the group's
    single read. When the member is read on its own, it is a view of its
    own pair. `kind` is `inline` and `must_understand` is `False`, as
    the convention declares. The documents and the group form the
    hierarchy below the group, with the group as its root. Each document
    is stored at its node's path without the leading `/`: the node at
    `/a/b` is at `a/b`.
    """

    __slots__ = ("_context", "_document", "_metadata")

    kind: Final = "inline"
    must_understand: Final = False

    def __init__(self, member: object, context: ZarrV3Context | None = None) -> None:
        scope = scoped(context, CORE_AND_EXTENSIONS)
        # The member is always under a group's key, even when read on its
        # own, so nesting depth is counted from there, as in the group.
        readings, members, document, problems = _read_consolidated_v3(
            member, scope, (ZARR_V3_CONSOLIDATED_METADATA_KEY,)
        )
        if len(problems) != 0 or document is None:
            raise MetadataValidationError(problems)
        self._adopt(document, scope, _nested_models(scope, readings, members))

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        metadata: dict[str, ZarrV3NodeMetadata],
    ) -> ZarrV3ConsolidatedMetadata:
        """Build the member of a group that a read found no problems with, holding the models that read built. The readers in this package build models through this method; pyright reports these calls as private use."""
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
        # Not shared with the readings, which hold their own models; see `metadata`.

    @property
    def context(self) -> Context:
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

    def __reduce__(self) -> tuple[type[ZarrV3ConsolidatedMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

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


def _no_documents() -> Mapping[str, ZarrV3NodeMetadataReading]:
    """The empty mapping of nested readings, for a group that has no consolidated metadata or whose consolidated metadata holds no documents."""
    return {}


@dataclass(frozen=True, slots=True)
class ZarrV3GroupMetadataReading:
    """A v3 group document as a scope read it, whatever it holds: the reading of each document in its consolidated metadata, every problem, and the model when there are no problems."""

    consolidated: Mapping[str, ZarrV3NodeMetadataReading] = dataclasses.field(
        default_factory=_no_documents
    )
    """The reading of each document in the consolidated metadata, as `read_node_metadata_v3` reads it, keyed by path."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV3GroupMetadata | None = None
    """The document's model when there is no problem; None otherwise."""

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field of each document in the consolidated metadata, as read, with its location in this document."""
        for path, reading in self.consolidated.items():
            for loc, node in reading.fields():
                yield (ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", path, *loc), node

    def __reduce__(self) -> tuple[Callable[..., object], tuple[object, ...]]:
        # A reading that holds a model pickles and copies the same way the
        # model does, and is restored as that model's own reading, so there
        # is still one model per document. A reading without a model is
        # rebuilt from the dict behind its read-only view, because the dict
        # can be pickled and the view cannot.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return (_group_reading, (dict(self.consolidated), self.problems, None))


def _group_reading(
    consolidated: dict[str, ZarrV3NodeMetadataReading],
    problems: tuple[ValidationProblem, ...],
    metadata: ZarrV3GroupMetadata | None,
) -> ZarrV3GroupMetadataReading:
    """Rebuild a group reading from the values `ZarrV3GroupMetadataReading.__reduce__` returns, with its documents held read-only."""
    return ZarrV3GroupMetadataReading(MappingProxyType(consolidated), problems, metadata)


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
    scope = scoped(context, CORE_AND_EXTENSIONS)
    node_type, problems = _node_type(value)
    if node_type == "array":
        return read_array_metadata_v3(value, context=scope)
    if node_type == "group":
        return read_group_metadata_v3(value, context=scope)
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
    scope = scoped(context, CORE_AND_EXTENSIONS)
    reading = read_node_metadata_v3(data, context=scope)
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
    scope = scoped(context, CORE_AND_EXTENSIONS)
    # Array documents and group documents are both stored at `zarr.json`.
    document = load_store_json(mapping, ZARR_V3_GROUP_METADATA_STORE_KEY)
    return node_metadata_from_json_v3(document, context=scope)


def validate_node_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 `zarr.json`: the problems `validate_array_metadata_v3` or `validate_group_metadata_v3` finds for the node type its `node_type` declares, or why it declares neither."""
    scope = scoped(context, CORE_AND_EXTENSIONS)
    return _read_node_v3(value, scope)[0].problems


def _read_node_v3(
    value: object, context: Context, at: tuple[str | int, ...] = ()
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
    """The refined members of a v3 group document, as the models hold them: the group's own members, and the members of each document in its consolidated metadata that a model can be built from."""

    attributes: dict[str, JSONValue] | None
    """The attributes; None when they have a problem."""
    extra_fields: dict[str, JSONValue]
    """Each member the spec does not define that is JSON."""
    consolidated: Mapping[str, ArrayMembersV3 | GroupMembersV3] | UNSET
    """The members of each array in the consolidated metadata that has no problem, and of each group, keyed by path; UNSET when there is no consolidated metadata."""
    document: dict[str, JSONValue] | None
    """The refined document that the model holds: each member as the read refined it, in the order the document writes them, including each document in the consolidated metadata; None when the read found a problem."""


def read_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3GroupMetadataReading:
    """Read `value`, a v3 group document, in `context`, whatever it holds.

    The result holds everything the read finds: each document in the
    consolidated metadata, read once, as `read_array_metadata_v3` or
    this function reads it; every problem `validate_group_metadata_v3`
    finds; and, when there are no problems, the group's model. The
    model's consolidated metadata holds the models of those documents,
    and their readings hold the same models. A value that is not an
    object holds nothing.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    reading, members = read_group_v3(value, scope)
    if members is None:
        return reading
    return _with_models(reading, members, value, scope)


def read_group_v3(
    value: object, context: Context, *, at: tuple[str | int, ...] = ()
) -> tuple[ZarrV3GroupMetadataReading, GroupMembersV3 | None]:
    """Read `value`, a v3 group document, in `context`, without building models. Return the reading and the refined members; the members are None when `value` is not an object.

    Every value must be JSON. A field object built by hand, in a document
    held by `consolidated_metadata`, is not JSON, and is refused in the
    same way `read_array_v3` refuses one. `at` is the location of this
    document within the input document, as in `read_array_v3`. Each
    document in the consolidated metadata is read at its own location, so
    a chain of nested documents is bounded by the maximum nesting depth,
    counted from the outermost root.
    """
    if not is_object(value):
        return ZarrV3GroupMetadataReading(problems=not_an_object(value)), None
    doc = value
    found: list[ValidationProblem] = list(missing_keys(_GROUP_METADATA_REQUIRED_KEYS_V3, doc))
    past = members_past_the_levels(doc, at)
    found.extend(past.values())
    whole, doc = doc, {key: item for key, item in doc.items() if key not in past}
    extra_fields, others = other_members(
        doc,
        _GROUP_METADATA_STANDARD_KEYS_V3,
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
    consolidated: dict[str, ZarrV3NodeMetadataReading] = {}
    held: Mapping[str, ArrayMembersV3 | GroupMembersV3] | UNSET = UNSET
    member: dict[str, JSONValue] | None = None
    if ZARR_V3_CONSOLIDATED_METADATA_KEY in doc:
        # Read whatever is written there, including the `UNSET` sentinel.
        # A document cannot hold a sentinel, and a reader that treated one
        # as an absent key would write it as null.
        consolidated, held, member, inside = _read_consolidated_v3(
            doc[ZARR_V3_CONSOLIDATED_METADATA_KEY],
            context,
            (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY),
        )
        found.extend(_prefix(ZARR_V3_CONSOLIDATED_METADATA_KEY, inside))
    reading = ZarrV3GroupMetadataReading(MappingProxyType(consolidated), with_input(found, whole))
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
            ZARR_V3_CONSOLIDATED_METADATA_KEY: member,
        }
        document = {key: refined[key] for key in cast("Mapping[str, object]", doc)}
    return reading, GroupMembersV3(attributes, extra_fields, held, document)


_CONSOLIDATED_MEMBERS: Final = ("kind", "must_understand", "metadata")
"""The members of an inline `consolidated_metadata`, in the order the convention declares them."""


_CONSOLIDATED_ENVELOPE: Final[dict[str, JSONValue]] = {"kind": "inline", "must_understand": False}
"""The fixed values the convention declares for the member."""


def documents_for(value: object) -> object:
    """`value`, a v3 group document, with each node model in its consolidated metadata replaced by that model's document. A `ZarrV3ConsolidatedMetadata` given as the member is replaced by its JSON. Returns `value` itself when it holds no models.

    A group's own document is built from the result, so a document built
    from models is plain JSON like any other, with each child written as
    the child wrote it.
    """
    if not is_object(value):
        return value
    document = value
    given = document.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    member = _member_documents_for(given)
    if member is given:
        return value
    return {**document, ZARR_V3_CONSOLIDATED_METADATA_KEY: member}


def _member_documents_for(member: object) -> object:
    """A `consolidated_metadata` member with each node model it lists replaced by the model's document. Returns `member` itself when it lists no models."""
    if isinstance(member, ZarrV3ConsolidatedMetadata):
        return member.to_json()
    if not is_object(member):
        return member
    entries = member.get("metadata")
    if not is_object(entries):
        return member
    replaced: dict[object, object] = {}
    changed = False
    for path, entry in entries.items():
        if isinstance(entry, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
            replaced[path] = entry.to_json()
            changed = True
        else:
            # A document listed here may itself list models.
            replaced[path] = documents_for(entry)
            changed = changed or replaced[path] is not entry
    if not changed:
        return member
    return {**member, "metadata": replaced}


def _read_node_model(
    entry: ZarrV3ArrayMetadata | ZarrV3GroupMetadata, context: Context, at: Loc
) -> tuple[ZarrV3NodeMetadataReading, ArrayMembersV3 | GroupMembersV3 | None]:
    """Read `entry`, a node model given in place of a listed document, in `context`.

    Returns the model's own reading when `context` reads every one of its
    claims identically, and a new read of its document when `context`
    claims more. Returns problems when `context` reads a name with a
    different definition, or with none. A model may join a group when its
    claims refine into the group's scope. On a gain, the document is read
    again in that scope, so a problem found by a newly claiming
    definition is reported at its location.
    """
    document = entry.to_json()
    found = context.disagreements(entry.claims)
    if len(found.conflicts) != 0:
        # Read the document again in the group's scope, so the reading
        # belongs to that scope, like every other reading the group holds.
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
    """The message for a conflict between a model entry's scope and the group's scope. It states the kind, the name as the document writes it, the definition that read it, and how the group's scope reads it: with another definition (distinguished from the model's when the two print the same), or with none."""
    kind, filed = conflict.key
    name = filed if written is None else written
    claimed = definition_said(conflict.claimed)
    head = f"expected a document read in the group's scope, got a model that reads the {kind_name(kind)} {name!r}"
    if conflict.found is None:
        return f"{head} by {claimed}, which the group's scope leaves unclaimed"
    found = definition_said(conflict.found)
    if claimed == found:
        return f"{head} by another definition than the one the group's scope reads it by, {found}"
    return f"{head} by {claimed}, which the group's scope reads by {found}"


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
    dict[str, JSONValue] | None,
    tuple[ValidationProblem, ...],
]:
    """Read an inline `consolidated_metadata` member in `context`.

    Returns four values: the reading of each document it holds, read
    once as `read_node_metadata_v3` reads it, keyed by path; the members
    of each document that a model can be built from; the refined member
    (each document as its read refined it) when there are no problems,
    else None; and every problem, located in the member. `at` is the
    location of the member within the input document. Each container the
    reader descends into (the member, its `metadata`, and each document)
    is checked at its location, as `_refine` checks a container, so a
    chain of documents is bounded by the maximum nesting depth.
    """
    if isinstance(value, ZarrV3ConsolidatedMetadata):
        # Another group's whole member: use its models at their paths, in
        # the order its document writes them.
        value = {**value._document, "metadata": dict(value.metadata)}  # pyright: ignore[reportPrivateUsage]
    past = nested_past_the_levels(value, at)
    if past is not None:
        return {}, {}, None, within((past,), at)
    if not is_object(value):
        return {}, {}, None, (ValidationProblem((), "expected an object", "invalid_type"),)
    env = value
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
    if "metadata" in env and not is_object(entries):
        problems.append(ValidationProblem(("metadata",), "expected an object", "invalid_type"))
    elif is_object(entries) and past is not None:
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
                key, readings[key], members.get(key), readings, members, node_types
            )
        )
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
    listed: ZarrV3NodeMetadataReading,
    listed_members: ArrayMembersV3 | GroupMembersV3 | None,
    readings: Mapping[str, ZarrV3NodeMetadataReading],
    members: Mapping[str, ArrayMembersV3 | GroupMembersV3],
    node_types: Mapping[str, NodeType | None],
) -> list[ValidationProblem]:
    """Check the own consolidated listing of `listed`, the group at `key`, against the outer group's flat listing (`readings` and their `node_types`). Each problem is located at the nested entry.

    The reference implementation lists every node below the group in the
    group's own listing, as a flat mapping, and gives each group it lists
    an empty listing of its own. So every node that a listed group lists
    must also be listed by the outer group, at the joined key, with the
    same node type and the same document, as each is read. The reference
    reader would drop a node that only the nested listing has. A node
    listed with another type contradicts the tree. A node listed with a
    different document would give a reader two answers for one node.
    Only the entries of this listing are checked. Deeper listings are
    checked when the document of the group that holds them is read, and
    a document with its own problem is reported for that problem.
    """
    problems: list[ValidationProblem] = []
    for path, entry in _reading_listing(listed).items():
        if len(_below_faults(path)) != 0:
            # The nested group's own read reports a key that is not a node path.
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
        own = _own_listing_members(listed_members).get(path)
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


def _own_listing_members(
    listed: ArrayMembersV3 | GroupMembersV3 | None,
) -> Mapping[str, ArrayMembersV3 | GroupMembersV3]:
    """The members of each document in a listed group's own listing that a model can be built from. Empty for an array, a group that lists nothing, or a document with a problem."""
    if isinstance(listed, GroupMembersV3) and listed.consolidated is not UNSET:
        return listed.consolidated
    return {}


def _read_alike(
    reading: ZarrV3NodeMetadataReading,
    members: ArrayMembersV3 | GroupMembersV3,
    other: ZarrV3NodeMetadataReading,
    other_members: ArrayMembersV3 | GroupMembersV3,
) -> bool:
    """Whether two documents of one node read the same. Arrays are compared by their model keys. Groups are compared by their own members only; each group's listing is checked separately, at its own location."""
    if isinstance(reading, ZarrV3ArrayMetadataReading) and isinstance(
        other, ZarrV3ArrayMetadataReading
    ):
        if not isinstance(members, ArrayMembersV3) or not isinstance(other_members, ArrayMembersV3):
            return True
        return array_key_of(reading, members) == array_key_of(other, other_members)
    if isinstance(members, GroupMembersV3) and isinstance(other_members, GroupMembersV3):
        return (json_text(members.attributes), json_text(members.extra_fields)) == (
            json_text(other_members.attributes),
            json_text(other_members.extra_fields),
        )
    return True


def _an(node_type: NodeType) -> str:
    return "an array" if node_type == "array" else "a group"


def _reading_listing(reading: ZarrV3NodeMetadataReading) -> Mapping[str, ZarrV3NodeMetadataReading]:
    """The readings listed in a reading's own consolidated metadata. Empty for an array reading, or a reading with no node type."""
    if isinstance(reading, ZarrV3GroupMetadataReading):
        return reading.consolidated
    return {}


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


def _with_models(
    reading: ZarrV3GroupMetadataReading,
    members: GroupMembersV3,
    value: object,
    context: Context,
) -> ZarrV3GroupMetadataReading:
    """`reading`, with a model for each document in its consolidated metadata that has no problem, and its own model when it has no problem. Each model is built in `context` from the refined values in `members`.

    A document with a problem (a key that is not a string, a value that
    is not JSON, or nesting past the depth cap) has no model. Each
    document in the consolidated metadata was read on its own, so the
    documents without a problem still get models. A listed group with a
    problem of its own still holds models for the documents in its own
    listing. These are found by walking `value`, the document that was
    read, down to that group.
    """
    if len(reading.problems) == 0:
        model = ZarrV3GroupMetadata._of(_document_of(members), context, reading, members)  # pyright: ignore[reportPrivateUsage]
        return model.reading
    value = documents_for(value)
    if members.consolidated is UNSET or not is_object(value):
        return reading
    member = value.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    if not is_object(member):
        return reading
    entries = member.get("metadata")
    if not is_object(entries):
        return reading
    held_entries = entries
    models = _nested_models(context, reading.consolidated, members.consolidated)
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
            # A listed group with a problem of its own (one the reader
            # refused, or one it walked past) still holds, in its reading,
            # a model of each document in its own listing that has no
            # problem.
            held[path] = _with_models(nested, child, held_entries[path], context)
    return dataclasses.replace(reading, consolidated=MappingProxyType(held))


def _nested_models(
    context: Context,
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
    present, is validated too: its envelope, and each document it holds
    by path, with each array read as `validate_array_metadata_v3` reads
    it, in `context`. The result equals the `problems` of
    `read_group_metadata_v3`, which also holds what was read to find
    them.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    return read_group_v3(value, scope)[0].problems


def is_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> TypeGuard[ZarrV3GroupMetadataJSON]:
    """Whether `value` is a v3 group document, written with tuples, that `validate_group_metadata_v3` finds no problems with."""
    scope = scoped(context, CORE_AND_EXTENSIONS)
    return (
        is_canonical_json(value, finite=False)
        and _is_canonical_group_metadata_v3(value)
        and not validate_group_metadata_v3(value, context=scope)
    )


def _is_canonical_group_metadata_v3(value: object) -> bool:
    """Whether `value`, a group document, uses the container types `ZarrV3GroupMetadataJSON` declares. Each document listed in its consolidated metadata must match its own TypedDict: arrays as `is_canonical_array_metadata_v3` checks, and groups by this same check, recursively."""
    if not is_object(value) or not isinstance(value, dict):
        return False
    member = value.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    if not is_object(member):
        return True
    entries = member.get("metadata")
    if not is_object(entries):
        return True
    for entry in entries.values():
        if not is_object(entry):
            continue
        node_type = entry.get("node_type")
        if node_type == "array" and not is_canonical_array_metadata_v3(entry):
            return False
        if node_type == "group" and not _is_canonical_group_metadata_v3(entry):
            return False
    return True


def parse_group_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or raise `MetadataValidationError`."""
    scope = scoped(context, CORE_AND_EXTENSIONS)
    problems = validate_group_metadata_v3(value, context=scope)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(documents_for(value)))
