"""The Zarr v2 group and its consolidated metadata."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from typing_extensions import TypeAliasType, TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._hierarchy import NodeType, said, tree_problems
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
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
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._context import disagreements, scoped
from zarr_metadata.definition._scope import Conflict, ScopeConflictError, claims_of
from zarr_metadata.model._array_v2 import (
    ARRAY_FILE_ATTRIBUTES,
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataReading,
    decoded_node_files_v2,
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
    member_changes,
    missing_keys,
    reading_of,
    unexpected_keys,
    validate_attributes,
)

# A runtime import: `get_type_hints` resolves the public signatures.
from zarr_metadata.v2._context import ZarrV2Context  # noqa: TC001
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.consolidated import ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY, ZarrV2GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import ResolvedField, UnclaimedField
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


def _group_problems_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v2 group document in the merged form: the `.zgroup` members, plus an optional `attributes` object from `.zattrs`. A group has no fields for a scope to read, so no scope is needed."""
    if not is_object(value):
        return not_an_object(value)
    doc = value
    problems: list[ValidationProblem] = list(missing_keys(_GROUP_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(unexpected_keys(_GROUP_METADATA_STANDARD_KEYS_V2, doc))
    problems.extend(check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(validate_attributes(doc["attributes"]))
    return with_input(problems, doc)


_GROUP_FILE_ATTRIBUTES: Final = ValidationProblem(
    ("attributes",),
    f"unexpected key 'attributes' in {ZARR_V2_GROUP_METADATA_STORE_KEY}: a group's attributes "
    f"belong in {ZARR_V2_ATTRIBUTES_STORE_KEY}",
    "invalid_value",
)
"""The problem of a `.zgroup` that holds `attributes`, which belong in `.zattrs`. The kind is the one `.zarray` gives the same member: the key is not unknown, but taken by the attributes, and a caller who tolerates an unknown key would drop the attributes the member holds."""


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
    """Read `value`, a v2 group document in the merged form (`.zgroup` plus `attributes` from `.zattrs`), in `context` (`CORE_V2` when none is given), whatever it holds: every problem, and the model when there are none. A group has no fields for a scope to read; `context` is taken for consistency with every other v2 reader."""
    context = scoped(context, CORE_V2)
    problems = _group_problems_v2(value)
    if len(problems) != 0:
        return ZarrV2GroupMetadataReading(problems=problems)
    return ZarrV2GroupMetadata(value, context=context).reading


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
    ZarrV2ArrayMetadataReading | ZarrV2GroupMetadataReading | ZarrV2UnknownNodeMetadataReading,
)
"""A v2 node as `read_node_metadata_v2` reads it from a store: as the array or group its files declare, or as neither."""


def read_node_metadata_v2(
    files: Mapping[str, object], *, context: ZarrV2Context | None = None
) -> ZarrV2NodeMetadataReading:
    """Read the v2 node described by `files`, in `context` (`CORE_V2` when none is given), as the node type its files declare.

    `files` maps each of the node's files to its decoded JSON: `.zarray`
    or `.zgroup`, and `.zattrs` when there is one. They can come from a
    store, decoded (as `node_metadata_from_key_value_v2` does), or from the
    entries of a `.zmetadata`, which hold the same JSON. A v2 document
    does not name its node type; the file does. A `.zarray` makes the node
    an array, read as `read_array_metadata_v2` reads it, and a `.zgroup`
    makes it a group, read as `read_group_metadata_v2` reads it, with the
    attributes at `.zattrs` in either case. `files` with neither, or with
    both, give a `ZarrV2UnknownNodeMetadataReading` with the problem. A
    `.zarray` or `.zgroup` that writes `attributes` itself gives the
    reading of its node type with no model, and with that problem
    followed by the problems of the rest of the file. This is the
    v2 counterpart of `read_node_metadata_v3`, which reads one `zarr.json`
    and its `node_type`.
    """
    context = scoped(context, CORE_V2)
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
            files, ZARR_V2_ARRAY_METADATA_STORE_KEY, ARRAY_FILE_ATTRIBUTES
        )
        reading = read_array_metadata_v2(document, context=context)
        if len(problems) != 0:
            return dataclasses.replace(
                reading, problems=(*problems, *reading.problems), metadata=None
            )
        return reading
    if is_group:
        document, problems = node_document_v2(
            files, ZARR_V2_GROUP_METADATA_STORE_KEY, _GROUP_FILE_ATTRIBUTES
        )
        reading = read_group_metadata_v2(document, context=context)
        if len(problems) != 0:
            return ZarrV2GroupMetadataReading(problems=(*problems, *reading.problems))
        return reading
    neither = ValidationProblem(
        (),
        f"expected a {ZARR_V2_ARRAY_METADATA_STORE_KEY} or a {ZARR_V2_GROUP_METADATA_STORE_KEY}",
        "missing_key",
    )
    return ZarrV2UnknownNodeMetadataReading((neither,))


def node_metadata_from_key_value_v2(
    mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
) -> ZarrV2NodeMetadata:
    """Return the model of the v2 node whose files are in `mapping`, a store's keys and bytes: each file is decoded, then read as `read_node_metadata_v2` reads it. Raises `MetadataValidationError` with every problem, including a file whose bytes are not JSON, and `TypeError` for a scope of another format before any bytes are decoded."""
    context = scoped(context, CORE_V2)
    files, problems = decoded_node_files_v2(mapping)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    reading = read_node_metadata_v2(files, context=context)
    if reading.metadata is None:
        raise MetadataValidationError(reading.problems)
    return reading.metadata


class ZarrV2GroupMetadataUpdate(TypedDict, total=False):
    """The members `ZarrV2GroupMetadata.update` replaces: `attributes` in the form a `.zattrs` writes them, or `UNSET` for no `.zattrs`."""

    attributes: Mapping[str, JSONValue] | UNSET


class ZarrV2GroupMetadata(Keyed):
    """A v2 group document, and the scope it was read in.

    Like the v3 models, this model is a pair. `to_json` returns the
    merged document (`.zgroup`, plus `attributes` when a `.zattrs` holds
    them) as written, refined. `attributes` is empty when no `.zattrs`
    exists, as in a v3 model. The document still tells the two apart:
    `to_json` and `to_key_value` write `attributes` and `.zattrs` only when
    a `.zattrs` exists, even an empty one. A group has no fields
    for a scope to read, so the scope is held only for consistency with
    the other models: `update` reads new attributes in it, and no other
    scope conflicts with the reading. A model is only built by reading:
    the constructor raises `MetadataValidationError` with every problem.
    Two groups are equal when their attributes are written the same way,
    as `_key_of` defines; no `.zattrs` and an empty one are equal.
    """

    __slots__ = ("_attributes", "_context", "_document", "_shown")

    _attributes: dict[str, JSONValue] | UNSET
    _context: ZarrV2Context
    _document: dict[str, JSONValue]
    _shown: Mapping[str, JSONValue]

    zarr_format: Final[Literal[2]] = 2  # noqa: PYI064 - a bare Final does not narrow a union of models
    node_type: Final[Literal["group"]] = "group"  # noqa: PYI064 - as above

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        context = scoped(context, CORE_V2)
        problems = _group_problems_v2(document)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        # Refine once. The attributes the model holds are taken from the
        # refined document, which shares no objects with the input.
        refined = refined_object(document)
        attributes: dict[str, JSONValue] | UNSET = (
            object_at(refined, "attributes") if "attributes" in refined else UNSET
        )
        self._adopt(refined, context, attributes)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        attributes: dict[str, JSONValue] | UNSET,
    ) -> ZarrV2GroupMetadata:
        """Build a model of a document that a read found no problems with, without reading it again. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, attributes)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        attributes: dict[str, JSONValue] | UNSET,
    ) -> None:
        self._document = document
        self._context = context
        self._attributes = attributes
        self._shown = frozen({} if attributes is UNSET else attributes)

    @property
    def reading(self) -> ZarrV2GroupMetadataReading:
        """The document as the scope read it: no problems, and this model."""
        return ZarrV2GroupMetadataReading(metadata=self)

    @property
    def context(self) -> ZarrV2Context:
        """The scope the document was read in. `update` reads new attributes in this scope."""
        return self._context

    @property
    def unclaimed_fields(self) -> dict[Loc, UnclaimedField]:
        """Every field nothing in scope claims. A v2 group holds no fields, so this is always empty; it is here so a reader can ask any node the same question."""
        return {}

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The user attributes a `.zattrs` holds, read-only at every level; empty when there is no `.zattrs`."""
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
        """The key `==` and `hash` compare for a v2 group model: its attributes as JSON text. No `.zattrs` counts as empty attributes, since both mean that the group has none, as in v3."""
        attributes = self._attributes
        return (json_text({} if attributes is UNSET else attributes),)

    def __reduce__(self) -> tuple[type[ZarrV2GroupMetadata], tuple[object, ZarrV2Context]]:
        return type(self), (self._document, self._context)

    def update(self, **members: Unpack[ZarrV2GroupMetadataUpdate]) -> ZarrV2GroupMetadata:
        """Return this model with `attributes` replacing the document's attributes, read in this model's own scope. `UNSET` removes them. Raises `MetadataValidationError` when the resulting document has a problem."""
        changes = member_changes(members, ZarrV2GroupMetadataUpdate, frozenset())
        document: dict[str, object] = {**self._document, **changes}
        for key, value in changes.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2GroupMetadata:
        """Return this document read in `context`: the same group, holding that scope."""
        context = scoped(context, CORE_V2)
        return self._of(self._document, context, self._attributes)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2GroupMetadata:
        """Return this document read in `context`. A group has no fields, so no scope conflicts with its reading, and this is the same as `with_context`."""
        return self.with_context(context)

    def refines(self, other: ZarrV2GroupMetadata) -> bool:
        """Whether this group holds everything `other` holds, which means its attributes are written the same way, no `.zattrs` counting as empty attributes. False when `other` is not a v2 group."""
        return type(other) is type(self) and self._model_key == other._model_key

    @classmethod
    def create_default(
        cls, *, context: ZarrV2Context | None = None, **overrides: Unpack[ZarrV2GroupMetadataUpdate]
    ) -> ZarrV2GroupMetadata:
        """Return a group with no `.zattrs`, with `overrides` applied, read in `context`. Raises `MetadataValidationError` when the resulting document has a problem."""
        changes = member_changes(overrides, ZarrV2GroupMetadataUpdate, frozenset())
        given = {key: value for key, value in changes.items() if value is not UNSET}
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
        valid, with every problem the read finds. Raises `TypeError` for a
        scope of another format before any bytes are decoded.
        """
        context = scoped(context, CORE_V2)
        files, problems = decoded_node_files_v2(
            mapping, (ZARR_V2_GROUP_METADATA_STORE_KEY, ZARR_V2_ATTRIBUTES_STORE_KEY)
        )
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        document, problems = node_document_v2(
            files, ZARR_V2_GROUP_METADATA_STORE_KEY, _GROUP_FILE_ATTRIBUTES
        )
        if len(problems) != 0:
            if document is not None:
                problems = (*problems, *_group_problems_v2(document))
            raise MetadataValidationError(problems)
        return cls(document, context=context)


ZarrV2NodeMetadata = TypeAliasType("ZarrV2NodeMetadata", ZarrV2ArrayMetadata | ZarrV2GroupMetadata)
"""The model of one v2 node, as `node_metadata_from_key_value_v2` returns it and a `.zmetadata` holds it: an array or a group."""


@dataclass(frozen=True, slots=True)
class ZarrV2ConsolidatedMetadataReading:
    """A v2 `.zmetadata` document as a scope read it: every problem, and the model when there are no problems."""

    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV2ConsolidatedMetadata | None = None
    """The document's model when there is no problem; None otherwise."""

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field of each array node, as read, located in the `.zmetadata`: under the node's `.zarray` entry, at the key the document writes for it. Nothing when there is no model, and nothing for a group, which has no fields."""
        if self.metadata is None:
            return
        yield from self.metadata._located_fields()  # pyright: ignore[reportPrivateUsage]

    def __reduce__(self) -> str | tuple[object, ...]:
        # Same as the other readings: pickle through the model, when there is one.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)


def read_consolidated_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2ConsolidatedMetadataReading:
    """Read `value`, a v2 `.zmetadata` document, in `context` (`CORE_V2` when none is given), whatever it holds: every problem the read finds, and the model when there are none. This is the form of `ZarrV2ConsolidatedMetadata(value, context)` that returns problems instead of raising them."""
    context = scoped(context, CORE_V2)
    document, nodes, files, problems = _read_consolidated_v2(value, context)
    if len(problems) != 0:
        return ZarrV2ConsolidatedMetadataReading(problems=problems)
    model = ZarrV2ConsolidatedMetadata._of(document, context, nodes, files)  # pyright: ignore[reportPrivateUsage]
    return model.reading


class ZarrV2ConsolidatedMetadata(Keyed):
    """A v2 `.zmetadata` document, and the scope its nodes were read in.

    `metadata` maps the path of each node to its model in this scope, as
    `ZarrV3ConsolidatedMetadata.metadata` does. A path is written without
    the leading `/`, and the root's path is `""`: the node at `/a/b` is at
    `a/b`. Each node is a `.zarray` or `.zgroup` entry, merged with its
    sibling `.zattrs`. A `.zattrs` with no sibling is kept but does not
    form a node. Any other entry must be JSON, and is kept. `to_json`
    returns the entries keyed by file (`"a/.zarray"`, `"a/.zattrs"`, ...)
    as written, refined, so the document still tells which nodes have a
    `.zattrs` at all. A model is only built by reading: the constructor
    raises `MetadataValidationError` with every problem, each located
    under its entry. Two documents are equal when each node means the same
    thing and the other entries are written the same way, as the model
    key defines. `refines`, `with_context` and `refined_in` work through
    the nodes.
    """

    __slots__ = ("_context", "_document", "_files", "_metadata")

    _context: ZarrV2Context
    _document: dict[str, JSONValue]
    _files: dict[str, dict[str, str]]
    _metadata: dict[str, ZarrV2NodeMetadata]

    zarr_consolidated_format: Final = 1

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        context = scoped(context, CORE_V2)
        read, nodes, files, problems = _read_consolidated_v2(document, context)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        self._adopt(read, context, nodes, files)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        metadata: dict[str, ZarrV2NodeMetadata],
        files: dict[str, dict[str, str]],
    ) -> ZarrV2ConsolidatedMetadata:
        """Build a model of a document that a read found no problems with, holding the nodes that read built and the entry key of each node's files. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, metadata, files)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        metadata: dict[str, ZarrV2NodeMetadata],
        files: dict[str, dict[str, str]],
    ) -> None:
        self._document = document
        self._context = context
        self._metadata = metadata
        # The entry key of each file of each node, by node path and then by
        # file name, as the document writes it.
        self._files = files

    @property
    def reading(self) -> ZarrV2ConsolidatedMetadataReading:
        """The document as the scope read it: no problems, and this model."""
        return ZarrV2ConsolidatedMetadataReading(metadata=self)

    @property
    def context(self) -> ZarrV2Context:
        """The scope the nodes were read in."""
        return self._context

    @property
    def metadata(self) -> Mapping[str, ZarrV2NodeMetadata]:
        """A read-only view of the model of each node, keyed by its path without the leading `/` (`""` for the root)."""
        return MappingProxyType(self._metadata)

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
        return (
            tuple(sorted((path, node._model_key) for path, node in self._metadata.items())),
            self._other_entries_text(),
        )

    def _other_entries_text(self) -> str:
        """The entries that no node is read from (an orphan `.zattrs`, or any other key) as JSON text, which `==` compares."""
        entries = object_at(self._document, "metadata")
        consumed = {key for names in self._files.values() for key in names.values()}
        return json_text({key: value for key, value in entries.items() if key not in consumed})

    def _located_fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field of each array node, as read, located in the document: under the node's `.zarray` entry, at the key the document writes for it."""
        for path, node in self._metadata.items():
            if isinstance(node, ZarrV2ArrayMetadata):
                key = self._files[path][ZARR_V2_ARRAY_METADATA_STORE_KEY]
                for loc, field in node.reading.fields():
                    yield ("metadata", key, *loc), field

    def __reduce__(self) -> tuple[type[ZarrV2ConsolidatedMetadata], tuple[object, ZarrV2Context]]:
        return type(self), (self._document, self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """Return this document with every node read in `context`, whatever that changes. Raises `MetadataValidationError` when a node has a problem in `context`."""
        context = scoped(context, CORE_V2)
        return type(self)(self._document, context=context)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2ConsolidatedMetadata:
        """Return this document with every node read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` naming each conflict, located at the
        node's entry. Raises `MetadataValidationError` when a gain exposes
        a problem.
        """
        context = scoped(context, CORE_V2)
        conflicts: list[Conflict] = []
        for path, node in self._metadata.items():
            if not isinstance(node, ZarrV2ArrayMetadata):
                continue
            found = disagreements(context, claims_of(node.reading.fields()))
            key = self._files[path][ZARR_V2_ARRAY_METADATA_STORE_KEY]
            conflicts.extend(
                dataclasses.replace(
                    conflict,
                    loc=("metadata", key, *(() if conflict.loc is None else conflict.loc)),
                )
                for conflict in located_conflicts(node.reading.fields(), found.conflicts)
            )
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return self.with_context(context)

    def refines(self, other: ZarrV2ConsolidatedMetadata) -> bool:
        """Whether every node this holds refines the node `other` holds at the same path, both hold the same paths, and the other entries are written the same way. False when `other` is not v2 consolidated metadata."""
        if type(other) is not type(self):
            return False
        if self._metadata.keys() != other._metadata.keys():
            return False
        if self._other_entries_text() != other._other_entries_text():
            return False
        return all(
            _v2_node_refines(self._metadata[path], other._metadata[path]) for path in self._metadata
        )

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
        are not JSON, or the document is not valid. Raises `TypeError` for
        a scope of another format before the bytes are decoded.
        """
        context = scoped(context, CORE_V2)
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
    """The store key of each `.zarray`, `.zgroup` and `.zattrs` entry, by node path and then by file name. Also a problem for each such key that does not name its file by the node's path, since a reader would not find that entry."""
    by_path: dict[str, dict[str, str]] = {}
    problems: list[ValidationProblem] = []
    for key in entries:
        path, _, name = key.rpartition("/")
        if name not in _NODE_FILES:
            continue
        faults = _key_faults(key)
        if len(faults) != 0:
            problems.append(
                ValidationProblem(
                    ("metadata", key),
                    f"expected the path of a node, got {key!r}, which {said(faults)}",
                    "invalid_value",
                )
            )
            continue
        by_path.setdefault(path, {})[name] = key
    return by_path, problems


def _key_faults(key: str) -> list[str]:
    """Each reason `key`, the key of a node's file in a `.zmetadata`, does not name the file by its node's path.

    A store names a node's file by the node's path: names joined by a
    single `/`, with no leading `/`. zarr-python 2.x looks an entry up by
    that key, so it never finds an entry written otherwise. Such a key is
    refused, as a v3 listing refuses it.
    """
    names = key.split("/")[:-1]
    faults: list[str] = []
    if key.startswith("/"):
        faults.append('starts with "/"')
        names = names[1:]
    if "" in names:
        faults.append('holds an empty name between two "/"')
    if any(name in (".", "..") for name in names):
        faults.append("has a '.' or '..' segment")
    return faults


def _read_consolidated_v2(
    data: object, context: ZarrV2Context
) -> tuple[
    dict[str, JSONValue],
    dict[str, ZarrV2NodeMetadata],
    dict[str, dict[str, str]],
    tuple[ValidationProblem, ...],
]:
    """Read `data`, a `.zmetadata` document. Return the document as read, with its keys in the order it writes them and each entry as read; the model of each node read in `context`, by path; the entry key of each node's files, by path and then by file name; and every problem, located in the document.

    Each entry is read as the kind of document its key names. An entry
    for a node's file is refined as user data, and its reader judges each
    value the spec interprets: a `.zattrs` entry is user data, and a
    `.zarray` or `.zgroup` reader refuses a non-finite number in a member
    the spec defines, as it does in a store. Any other entry must be JSON
    by RFC 8259. A `.zarray` or `.zgroup` entry is read as that document,
    merged with its sibling `.zattrs`. Its problems are located under its own entry,
    and problems with the attributes under the `.zattrs` entry. The
    nodes and their files are empty when there is any problem.
    """
    if not is_object(data):
        return {}, {}, {}, not_an_object(data)
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
                refine = refine_user_data if key.rsplit("/", 1)[-1] in _NODE_FILES else refine_json
                entry, found = refine(value, ("metadata", key))
                problems.extend(found)
                refined[key] = entry
    nodes: dict[str, ZarrV2NodeMetadata] = {}
    by_path: dict[str, dict[str, str]] = {}
    if len(problems) == 0:
        # A key that does not name its node's path is reported along with
        # the node problems. Every other node is still read, so a user sees
        # all problems at once.
        by_path, misnamed = _entries_by_path(refined)
        problems.extend(misnamed)
        for path, names in by_path.items():
            node, found = _read_node_v2(path, names, refined, context)
            problems.extend(found)
            if node is not None:
                nodes[path] = node
        problems.extend(_hierarchy_problems_v2(by_path))
    if len(problems) != 0:
        nodes = {}
    files = {path: by_path[path] for path in nodes}
    members: dict[str, JSONValue] = {"zarr_consolidated_format": 1, "metadata": refined}
    # The keys in the order the document writes them, as the v3 models keep them.
    document = {key: members[key] for key in doc if isinstance(key, str) and key in members}
    return document, nodes, files, with_input(problems, doc)


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
    path: str, names: Mapping[str, str], entries: Mapping[str, JSONValue], context: ZarrV2Context
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
    beside: list[ValidationProblem] = []
    if "attributes" in merged:
        # The same problem the node reader reports for the file, at the
        # entry. The rest of the entry is still read, for its own problems.
        found = ARRAY_FILE_ATTRIBUTES if zarray is not None else _GROUP_FILE_ATTRIBUTES
        beside.append(dataclasses.replace(found, loc=("metadata", key, *found.loc)))
        del merged["attributes"]
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
        if members is None or len(beside) != 0:
            return None, [*beside, *located(reading.problems)]
        return ZarrV2ArrayMetadata._of(members.document, context, reading, members), []  # pyright: ignore[reportPrivateUsage]
    found = _group_problems_v2(merged)
    if len(found) != 0 or len(beside) != 0:
        return None, [*beside, *located(found)]
    document = refined_object(merged)
    attributes: dict[str, JSONValue] | UNSET = (
        object_at(document, "attributes") if "attributes" in document else UNSET
    )
    return ZarrV2GroupMetadata._of(document, context, attributes), []  # pyright: ignore[reportPrivateUsage]
