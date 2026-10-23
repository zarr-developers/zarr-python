"""The Zarr v2 group and its consolidated metadata."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypeGuard, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._hierarchy import NodeType, tree_problems
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    frozen,
    is_json_object,
    is_object,
    json_text,
    not_an_object,
    object_at,
    refine_json,
    refine_user_data,
    refined_object,
    shown,
    with_input,
)
from zarr_metadata._json import is_canonical_json as _is_canonical_json
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._context import Context, ZarrV2Context, scoped
from zarr_metadata.definition._scope import Claims, Conflict, ScopeConflictError
from zarr_metadata.model._array_v2 import (
    ARRAY_FILE_ATTRIBUTES,
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataReading,
    node_document_v2,
    read_array_metadata_v2,
    read_array_v2,
)
from zarr_metadata.model._keyed import Keyed
from zarr_metadata.model._shared import (
    StoreKey,
    check_literal,
    dump_store_json,
    load_store_json,
    located_conflicts,
    missing_keys,
    reading_of,
    unexpected_keys,
    validate_attributes,
)
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.consolidated import ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY, ZarrV2GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import ResolvedField
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v2.consolidated import ZarrV2ConsolidatedMetadataStoreKey
    from zarr_metadata.v2.group import ZarrV2GroupMetadataStoreKey


_GROUP_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__required_keys__
)


_GROUP_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__optional_keys__
)


_GROUP_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    _GROUP_METADATA_REQUIRED_KEYS_V2 | _GROUP_METADATA_OPTIONAL_KEYS_V2
)


def validate_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v2 group doc.

    Validates the in-memory merged form: the `.zgroup` fields plus an
    optional `attributes` mapping folded in from `.zattrs`. A group has no
    fields for a scope to read. `context` is accepted for consistency with
    every other v2 reader.
    """
    scoped(context, CORE_V2)
    if not is_object(value):
        return not_an_object(value)
    doc = value
    problems: list[ValidationProblem] = list(missing_keys(_GROUP_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(unexpected_keys(_GROUP_METADATA_STANDARD_KEYS_V2, doc))
    problems.extend(check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(validate_attributes(doc["attributes"]))
    return with_input(problems, doc)


def is_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> TypeGuard[ZarrV2GroupMetadataJSON]:
    """Whether `value` is a structurally-valid v2 group metadata document. `context` is accepted for consistency with every other v2 reader."""
    return _is_canonical_json(value, finite=False) and not validate_group_metadata_v2(
        value, context=context
    )


def parse_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV2GroupMetadataJSON`, or raise `MetadataValidationError`. `context` is accepted for consistency with every other v2 reader."""
    problems = validate_group_metadata_v2(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV2GroupMetadataJSON, arrays_to_tuples(value))


_GROUP_FILE_ATTRIBUTES: Final = ValidationProblem(
    ("attributes",), "unexpected key 'attributes'", "unknown_key"
)
"""The problem of a `.zgroup` that holds `attributes`, which `.zgroup` does not declare: a group's attributes belong in `.zattrs`."""


@dataclass(frozen=True, slots=True)
class ZarrV2GroupMetadataReading:
    """A v2 group document as a scope read it: every problem, and the model when there are no problems. A group has no fields for a scope to read."""

    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV2GroupMetadata | None = None
    """The document's model when there is no problem; None otherwise."""

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Its fields as read: none, since a group has no fields."""
        return iter(())

    def __reduce__(self) -> str | tuple[object, ...]:
        # Same as the other readings: pickle through the model, when there is one.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)


def read_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2GroupMetadataReading:
    """Read `value`, a v2 group document in the merged form (`.zgroup` plus `attributes` from `.zattrs`), in `context` (`CORE_V2` when none is given), whatever it holds: every problem `validate_group_metadata_v2` finds, and the model when there are none."""
    scope = scoped(context, CORE_V2)
    problems = validate_group_metadata_v2(value, context=scope)
    if len(problems) != 0:
        return ZarrV2GroupMetadataReading(problems=problems)
    return ZarrV2GroupMetadata(value, context=scope).reading


@dataclass(frozen=True, slots=True)
class ZarrV2UnknownNodeMetadataReading:
    """A store that holds no v2 node: neither a `.zarray` nor a `.zgroup`, or both. The problems say which."""

    problems: tuple[ValidationProblem, ...]
    """Why the store holds no node."""

    @property
    def metadata(self) -> None:
        """Its model: always None, since there is no node to model."""
        return None

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Its fields as read: none, since no document is read."""
        return iter(())


ZarrV2NodeMetadataReading = TypeAliasType(
    "ZarrV2NodeMetadataReading",
    "ZarrV2ArrayMetadataReading | ZarrV2GroupMetadataReading | ZarrV2UnknownNodeMetadataReading",
)
"""A v2 node as `read_node_metadata_v2` reads it from a store: as the array or group its files declare, or as neither."""


def read_node_metadata_v2(
    mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
) -> ZarrV2NodeMetadataReading:
    """Read the v2 node whose files are in `mapping`, in `context` (`CORE_V2` when none is given), as the node type its files declare.

    A v2 document does not name its node type; the file does. A
    `.zarray` makes the node an array, read as `read_array_metadata_v2`
    reads it, and a `.zgroup` makes it a group, read as
    `read_group_metadata_v2` reads it, with the attributes at `.zattrs`
    in either case. A store with neither file, or with both, gives a
    `ZarrV2UnknownNodeMetadataReading` with the problem. A file that
    cannot be read (bytes that are not JSON, or `attributes` written in
    the node file) gives the reading of its node type with the problem
    and no model. This is the v2 counterpart of `read_node_metadata_v3`,
    which reads one `zarr.json` and its `node_type`.
    """
    scope = scoped(context, CORE_V2)
    files = cast("Mapping[str, bytes]", mapping)
    is_array = ZARR_V2_ARRAY_METADATA_STORE_KEY in files
    is_group = ZARR_V2_GROUP_METADATA_STORE_KEY in files
    if is_array and is_group:
        both = ValidationProblem(
            (ZARR_V2_GROUP_METADATA_STORE_KEY,),
            f"a node is an array or a group, not both: a {ZARR_V2_ARRAY_METADATA_STORE_KEY} is beside it",
            "invalid_value",
        )
        return ZarrV2UnknownNodeMetadataReading((both,))
    if is_array:
        document, problems = node_document_v2(
            mapping, ZARR_V2_ARRAY_METADATA_STORE_KEY, ARRAY_FILE_ATTRIBUTES
        )
        if len(problems) != 0:
            return ZarrV2ArrayMetadataReading(problems=problems)
        return read_array_metadata_v2(document, context=scope)
    if is_group:
        document, problems = node_document_v2(
            mapping, ZARR_V2_GROUP_METADATA_STORE_KEY, _GROUP_FILE_ATTRIBUTES
        )
        if len(problems) != 0:
            return ZarrV2GroupMetadataReading(problems=problems)
        return read_group_metadata_v2(document, context=scope)
    neither = ValidationProblem(
        (),
        f"expected a {ZARR_V2_ARRAY_METADATA_STORE_KEY} or a {ZARR_V2_GROUP_METADATA_STORE_KEY}",
        "missing_key",
    )
    return ZarrV2UnknownNodeMetadataReading((neither,))


def node_metadata_from_key_value_v2(
    mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
) -> ZarrV2NodeMetadata:
    """Return the model of the v2 node whose files are in `mapping`, read in `context` as `read_node_metadata_v2` reads it. Raises `MetadataValidationError` with every problem the read finds."""
    reading = read_node_metadata_v2(mapping, context=context)
    if reading.metadata is None:
        raise MetadataValidationError(reading.problems)
    return reading.metadata


def validate_node_metadata_v2(
    mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason the files in `mapping` are not a valid v2 node, as `read_node_metadata_v2` reads them."""
    return read_node_metadata_v2(mapping, context=context).problems


class ZarrV2GroupMetadataUpdate(TypedDict, total=False):
    """The members `ZarrV2GroupMetadata.update` replaces: `attributes` in the form a `.zattrs` writes them, or `UNSET` for no `.zattrs`."""

    attributes: Mapping[str, JSONValue] | UNSET


class ZarrV2GroupMetadata(Keyed):
    """A v2 group document, and the scope it was read in.

    Like the v3 models, this model is a pair. `to_json` returns the
    merged document (`.zgroup`, plus `attributes` when a `.zattrs` holds
    them) as written, refined. `attributes` is `UNSET` when no `.zattrs`
    exists, which is distinct from an empty one. A group has no fields
    for a scope to read, so the scope is held only for consistency with
    the other models: `update` reads new attributes in it, and no other
    scope conflicts with the reading. A model is only built by reading:
    the constructor raises `MetadataValidationError` with every problem.
    Two groups are equal when their attributes are written the same way,
    as `_key_of` defines.
    """

    __slots__ = ("_attributes", "_context", "_document", "_shown")

    _attributes: dict[str, JSONValue] | UNSET
    _context: Context
    _document: dict[str, JSONValue]
    _shown: Mapping[str, JSONValue] | UNSET

    zarr_format: Final = 2

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded: none, since a group has no fields."""
        return MappingProxyType({})

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        scope = scoped(context, CORE_V2)
        problems = validate_group_metadata_v2(document, context=scope)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        # Refine once. The attributes the model holds are taken from the
        # refined document, which shares no objects with the input.
        refined = refined_object(document)
        attributes: dict[str, JSONValue] | UNSET = (
            object_at(refined, "attributes") if "attributes" in refined else UNSET
        )
        self._adopt(refined, scope, attributes)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        attributes: dict[str, JSONValue] | UNSET,
    ) -> ZarrV2GroupMetadata:
        """Build a model of a document that a read found no problems with, without reading it again. The readers in this package build models through this method; pyright reports these calls as private use."""
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

    @property
    def reading(self) -> ZarrV2GroupMetadataReading:
        """The document as the scope read it: no problems, and this model."""
        return ZarrV2GroupMetadataReading(metadata=self)

    @property
    def context(self) -> Context:
        """The scope the document was read in. `update` reads new attributes in this scope."""
        return self._context

    @property
    def attributes(self) -> Mapping[str, JSONValue] | UNSET:
        """The user attributes a `.zattrs` holds, read-only at every level; `UNSET` when there is no `.zattrs`."""
        return self._shown

    def to_json(self) -> ZarrV2GroupMetadataJSON:
        """The merged document as written, refined. The result shares no objects with the model.

        `attributes` is included when set, even if empty. This is not the
        on-disk `.zgroup`, which excludes them: `to_key_value` splits the
        document as a store holds it
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313; https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        return cast("ZarrV2GroupMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2GroupMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zgroup` without the attributes, and `.zattrs` with the attributes when they are set, even if empty."""
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

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for a v2 group model: its attributes as JSON text, or `UNSET` when there is no `.zattrs`."""
        attributes = self._attributes
        return (UNSET if attributes is UNSET else json_text(attributes),)

    def __reduce__(self) -> tuple[type[ZarrV2GroupMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

    def update(self, **members: Unpack[ZarrV2GroupMetadataUpdate]) -> ZarrV2GroupMetadata:
        """Return this model with `attributes` replacing the document's attributes, read in this model's own scope. `UNSET` removes them. Raises `MetadataValidationError` when the resulting document has a problem."""
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2GroupMetadata:
        """Return this document read in `context`: the same group, holding that scope."""
        scope = scoped(context, CORE_V2)
        return self._of(self._document, scope, self._attributes)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2GroupMetadata:
        """Return this document read in `context`. A group has no fields, so no scope conflicts with its reading, and this is the same as `with_context`."""
        return self.with_context(context)

    def refines(self, other: ZarrV2GroupMetadata) -> bool:
        """Whether this group holds everything `other` holds, which means its attributes are written the same way. False when `other` is not a v2 group."""
        return type(other) is type(self) and self._model_key == other._model_key

    @classmethod
    def create_default(
        cls, *, context: ZarrV2Context | None = None, **overrides: Unpack[ZarrV2GroupMetadataUpdate]
    ) -> ZarrV2GroupMetadata:
        """Return a group with no `.zattrs`, with `overrides` applied, read in `context`. Raises `MetadataValidationError` when the resulting document has a problem."""
        given = {key: value for key, value in overrides.items() if value is not UNSET}
        return cls({"zarr_format": 2, **given}, context=context)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV2Context | None = None
    ) -> ZarrV2GroupMetadata:
        """Return the model of `data`, a v2 group document with its attributes under `attributes`, read in `context`. Raises `MetadataValidationError` with every problem."""
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
    ) -> ZarrV2GroupMetadata:
        """Return the model of the group at `.zgroup` in `mapping`, with the attributes at `.zattrs` when that key exists, read in `context`.

        Raises `MetadataValidationError` when `.zgroup` is missing, bytes
        are not JSON, `.zgroup` holds `attributes`, or the document is not
        valid.
        """
        document, problems = node_document_v2(
            mapping, ZARR_V2_GROUP_METADATA_STORE_KEY, _GROUP_FILE_ATTRIBUTES
        )
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        return cls(document, context=context)


ZarrV2NodeMetadata: TypeAlias = "ZarrV2ArrayMetadata | ZarrV2GroupMetadata"
"""The model of one node in a v2 `.zmetadata` document: an array or a group."""


class ZarrV2ConsolidatedMetadata(Keyed):
    """A v2 `.zmetadata` document, and the scope its nodes were read in.

    `metadata` holds the flat entries keyed by file (`"path/.zarray"`,
    `"path/.zattrs"`, ...) as written, refined. This keeps the
    information about which nodes had a `.zattrs` at all. `nodes` holds
    each `.zarray` or `.zgroup` entry, merged with its sibling `.zattrs`,
    as a model in this scope, keyed by the node's path (`""` for the
    root). A `.zattrs` with no sibling is kept but does not form a node.
    Any other entry must be JSON, and is kept. A model is only built by
    reading: the constructor raises `MetadataValidationError` with every
    problem, each located under its entry. Two documents are equal when
    each node means the same thing and the other entries are written the
    same way, as the model key defines. `refines`, `with_context` and
    `refined_in` work through the nodes.
    """

    __slots__ = ("_context", "_document", "_nodes", "_shown")

    _context: Context
    _document: dict[str, JSONValue]
    _nodes: dict[str, ZarrV2NodeMetadata]
    _shown: Mapping[str, JSONValue]

    zarr_consolidated_format: Final = 1

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        scope = scoped(context, CORE_V2)
        entries, nodes, problems = _read_consolidated_v2(document, scope)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        self._adopt({"zarr_consolidated_format": 1, "metadata": entries}, scope, nodes)

    @classmethod
    def _of(
        cls, document: dict[str, JSONValue], context: Context, nodes: dict[str, ZarrV2NodeMetadata]
    ) -> ZarrV2ConsolidatedMetadata:
        """Build a model of a document that a read found no problems with, holding the nodes that read built. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, nodes)
        return model

    def _adopt(
        self, document: dict[str, JSONValue], context: Context, nodes: dict[str, ZarrV2NodeMetadata]
    ) -> None:
        self._document = document
        self._context = context
        self._nodes = nodes
        self._shown = frozen(object_at(document, "metadata"))

    @property
    def context(self) -> Context:
        """The scope the nodes were read in."""
        return self._context

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        """The entries as written, refined, keyed by store key. Read-only at every level."""
        return self._shown

    @property
    def nodes(self) -> Mapping[str, ZarrV2NodeMetadata]:
        """A read-only view of the model of each node, keyed by its path below the root (`""` for the root)."""
        return MappingProxyType(self._nodes)

    def to_json(self) -> dict[str, JSONValue]:
        """The `.zmetadata` document as written, refined. The result shares no objects with the model."""
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

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for v2 consolidated metadata: each node's path and key, and every other entry as JSON text."""
        nodes = self._nodes
        return (
            tuple(sorted((path, node._model_key) for path, node in nodes.items())),
            self._other_entries_text(),
        )

    def _other_entries_text(self) -> str:
        """The entries that no node is read from (an orphan `.zattrs`, or any other key) as JSON text, which `==` compares."""
        entries = object_at(self._document, "metadata")
        consumed: set[str] = set()
        for path, names in _entries_by_path(entries)[0].items():
            if path in self._nodes:
                consumed.update(names.values())
        return json_text({key: value for key, value in entries.items() if key not in consumed})

    def __reduce__(self) -> tuple[type[ZarrV2ConsolidatedMetadata], tuple[object, Context]]:
        return type(self), (self._document, self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """Return this document with every node read in `context`, whatever that changes. Raises `MetadataValidationError` when a node has a problem in `context`."""
        scope = scoped(context, CORE_V2)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """Return this document with every node read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` naming each conflict, located at the
        node's entry. Raises `MetadataValidationError` when a gain exposes
        a problem.
        """
        scope = scoped(context, CORE_V2)
        conflicts: list[Conflict] = []
        entries = object_at(self._document, "metadata")
        by_path, _ = _entries_by_path(entries)
        for path, node in self._nodes.items():
            if not isinstance(node, ZarrV2ArrayMetadata):
                continue
            found = scope.disagreements(node.claims)
            key = by_path[path][ZARR_V2_ARRAY_METADATA_STORE_KEY]
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
        """Whether every node this holds refines the node `other` holds at the same path, both hold the same paths, and the other entries are written the same way. False when `other` is not v2 consolidated metadata."""
        if type(other) is not type(self):
            return False
        if self._nodes.keys() != other._nodes.keys():
            return False
        if self._other_entries_text() != other._other_entries_text():
            return False
        return all(_v2_node_refines(self._nodes[path], other._nodes[path]) for path in self._nodes)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV2Context | None = None
    ) -> ZarrV2ConsolidatedMetadata:
        """Return the model of `data`, a `.zmetadata` document, with its nodes read in `context`. Raises `MetadataValidationError` with every problem."""
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
    ) -> ZarrV2ConsolidatedMetadata:
        """Return the model of the document at `.zmetadata` in `mapping`, read in `context`.

        Raises `MetadataValidationError` when the key is missing, its bytes
        are not JSON, or the document is not valid.
        """
        return cls(
            load_store_json(mapping, ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY), context=context
        )


def _v2_node_refines(node: ZarrV2NodeMetadata, other: ZarrV2NodeMetadata) -> bool:
    """Whether `node` refines `other`. Models of the same kind use their own `refines`; an array never refines a group, or the reverse."""
    if isinstance(node, ZarrV2ArrayMetadata):
        return isinstance(other, ZarrV2ArrayMetadata) and node.refines(other)
    return isinstance(other, ZarrV2GroupMetadata) and node.refines(other)


_NODE_FILES: Final = (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    ZARR_V2_ATTRIBUTES_STORE_KEY,
)


def _entries_by_path(
    entries: Mapping[str, JSONValue],
) -> tuple[dict[str, dict[str, str]], list[ValidationProblem]]:
    """The store key of each `.zarray`, `.zgroup` and `.zattrs` entry, by node path and then by file name. Also a problem for each extra key that names a file of a node already seen, since that entry would otherwise not be read."""
    by_path: dict[str, dict[str, str]] = {}
    problems: list[ValidationProblem] = []
    for key in entries:
        path, _, name = key.rpartition("/")
        if name not in _NODE_FILES:
            continue
        # A node's path, as a store names it: segments joined by a single `/`.
        segments = [segment for segment in path.split("/") if segment != ""]
        if any(segment in (".", "..") for segment in segments):
            problems.append(
                ValidationProblem(
                    ("metadata", key),
                    f"expected the path of a node, got {key!r}, which has a '.' or '..' segment",
                    "invalid_value",
                )
            )
            continue
        path = "/".join(segments)
        files = by_path.setdefault(path, {})
        if name in files:
            problems.append(
                ValidationProblem(
                    ("metadata", key),
                    f"a second {name} for the node at {path!r}, which {files[name]!r} is",
                    "invalid_value",
                )
            )
            continue
        files[name] = key
    return by_path, problems


def _read_consolidated_v2(
    data: object, context: Context
) -> tuple[dict[str, JSONValue], dict[str, ZarrV2NodeMetadata], tuple[ValidationProblem, ...]]:
    """Read `data`, a `.zmetadata` document. Return each entry as read, the model of each node read in `context`, and every problem, located in the document.

    Each entry is read as the kind of document its key names. A `.zattrs`
    entry is user data. Any other entry must be JSON by RFC 8259. A
    `.zarray` or `.zgroup` entry is read as that document, merged with
    its sibling `.zattrs`. Its problems are located under its own entry,
    and problems with the attributes under the `.zattrs` entry. The
    nodes are empty when there is any problem.
    """
    if not is_object(data):
        return {}, {}, not_an_object(data)
    doc = data
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
        if not is_object(entries) or not all(isinstance(k, str) for k in entries):
            problems.append(
                ValidationProblem(
                    ("metadata",), "expected an object with string keys", "invalid_type"
                )
            )
        else:
            for key, value in entries.items():
                if not isinstance(key, str):
                    continue
                refine = (
                    refine_user_data
                    if key.rsplit("/", 1)[-1] == ZARR_V2_ATTRIBUTES_STORE_KEY
                    else refine_json
                )
                entry, found = refine(value, ("metadata", key))
                problems.extend(found)
                refined[key] = entry
    nodes: dict[str, ZarrV2NodeMetadata] = {}
    by_path: dict[str, dict[str, str]] = {}
    if len(problems) == 0:
        # A second key for one file is reported along with the node
        # problems. Every node is still read, so a user sees all problems
        # at once.
        by_path, doubled = _entries_by_path(refined)
        problems.extend(doubled)
        for path, names in by_path.items():
            node, found = _read_node_v2(path, names, refined, context)
            problems.extend(found)
            if node is not None:
                nodes[path] = node
        problems.extend(_hierarchy_problems_v2(by_path))
    if len(problems) != 0:
        nodes = {}
    return refined, nodes, with_input(problems, doc)


def _hierarchy_problems_v2(by_path: Mapping[str, Mapping[str, str]]) -> list[ValidationProblem]:
    """Why the nodes in a `.zmetadata` do not form a valid hierarchy, as `tree_problems` checks it; the paths are already checked against the v2 rules. A node below an array is reported at its own entry. A missing group above a node is reported at the entry that group would have. An orphan `.zattrs` is not a node."""
    # The root is a group unless an entry says otherwise: a `.zmetadata`
    # does not have to list its own `.zgroup`.
    nodes: dict[str, NodeType] = {"/": "group"}
    for path, names in by_path.items():
        if ZARR_V2_ARRAY_METADATA_STORE_KEY in names:
            nodes[f"/{path}"] = "array"
        elif ZARR_V2_GROUP_METADATA_STORE_KEY in names:
            nodes[f"/{path}"] = "group"
    problems: list[ValidationProblem] = []
    for found in tree_problems(nodes):
        at = found.loc[0]
        path = at[1:] if isinstance(at, str) else ""
        names = by_path.get(path, {})
        key = names.get(
            ZARR_V2_ARRAY_METADATA_STORE_KEY,
            names.get(
                ZARR_V2_GROUP_METADATA_STORE_KEY,
                f"{path}/{ZARR_V2_GROUP_METADATA_STORE_KEY}"
                if path != ""
                else ZARR_V2_GROUP_METADATA_STORE_KEY,
            ),
        )
        problems.append(ValidationProblem(("metadata", key), found.message, found.kind))
    return problems


def _read_node_v2(
    path: str, names: Mapping[str, str], entries: Mapping[str, JSONValue], context: Context
) -> tuple[ZarrV2NodeMetadata | None, list[ValidationProblem]]:
    """Read the node at `path` from its `.zarray` or `.zgroup` entry merged with its `.zattrs`. Return the node and every problem, located under the entries. Returns None and no problems when there is only a `.zattrs`."""
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
    if not is_json_object(document):
        return None, [
            ValidationProblem(
                ("metadata", key), f"expected an object, got {shown(document)}", "invalid_type"
            )
        ]
    merged = dict(document)
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
        return ZarrV2ArrayMetadata._of(members.document, context, reading, members), []  # pyright: ignore[reportPrivateUsage]
    found = validate_group_metadata_v2(merged, context=context)
    if len(found) != 0:
        return None, located(found)
    document = refined_object(merged)
    attributes: dict[str, JSONValue] | UNSET = (
        object_at(document, "attributes") if "attributes" in document else UNSET
    )
    return ZarrV2GroupMetadata._of(document, context, attributes), []  # pyright: ignore[reportPrivateUsage]
