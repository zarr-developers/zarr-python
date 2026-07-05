"""In-memory models for Zarr group and consolidated metadata documents."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._array import (
    ATTRIBUTES_STORE_KEY_V2,
    ArrayMetadataModelV3,
    must_understand_subset,
)
from zarr_metadata.model._sentinel import UNSET, UnsetType
from zarr_metadata.model._validation import (
    GROUP_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    load_store_json,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_consolidated_metadata_v3,
)

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v2.group import GroupMetadataV2
    from zarr_metadata.v3.array import ExtensionFieldV3
    from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3
    from zarr_metadata.v3.group import GroupMetadataV3

GroupMetadataStoreKeyV3 = Literal["zarr.json"]
GROUP_METADATA_STORE_KEY_V3: Final[GroupMetadataStoreKeyV3] = "zarr.json"

GroupMetadataStoreKeyV2 = Literal[".zgroup"]
GROUP_METADATA_STORE_KEY_V2: Final[GroupMetadataStoreKeyV2] = ".zgroup"

ConsolidatedMetadataStoreKeyV2 = Literal[".zmetadata"]
CONSOLIDATED_METADATA_STORE_KEY_V2: Final[ConsolidatedMetadataStoreKeyV2] = ".zmetadata"

# The key under which consolidated metadata is embedded in a v3 group document.
# This is a reference-implementation convention (not a spec artifact), stored
# as an extension field on the group's `zarr.json`.
CONSOLIDATED_METADATA_KEY_V3: Final = "consolidated_metadata"


class GroupMetadataModelV3Partial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `GroupMetadataModelV3`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `GroupMetadataModelV3.update` and
    `create_default`. The `init=False` fields `zarr_format` and `node_type`
    are intentionally excluded, since they cannot be passed to
    `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_group.py::test_group_partial_keys_match_settable_model_fields`.
    """

    attributes: dict[str, JSONValue]
    consolidated_metadata: ConsolidatedMetadataModelV3 | UnsetType
    extra_fields: dict[str, ExtensionFieldV3]


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupMetadataModelV3:
    """In-memory model of a v3 group metadata document.

    A canonical, lossless representation of the `zarr.json` content for a
    group. The `consolidated_metadata` reference-implementation convention is
    modeled as a typed field holding thin child models; every other unknown
    top-level key lands in `extra_fields` verbatim.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["group"] = field(default="group", init=False)
    attributes: dict[str, JSONValue]
    consolidated_metadata: ConsolidatedMetadataModelV3 | UnsetType
    extra_fields: dict[str, ExtensionFieldV3]

    def __post_init__(self) -> None:
        reserved = GROUP_METADATA_STANDARD_KEYS_V3 | {CONSOLIDATED_METADATA_KEY_V3}
        if set(self.extra_fields.keys()).intersection(reserved):
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("extra_fields",),
                        "Extra fields cannot overlap with standard GroupMetadataV3 fields",
                        "invalid_value",
                    )
                ]
            )

    @classmethod
    def create_default(
        cls, **overrides: Unpack[GroupMetadataModelV3Partial]
    ) -> GroupMetadataModelV3:
        """
        Create a default (empty) v3 group metadata model, with optional overrides.

        The default is a structurally-valid group with no attributes — the group
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`).
        """
        default = cls(attributes={}, consolidated_metadata=UNSET, extra_fields={})
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[GroupMetadataModelV3Partial]) -> GroupMetadataModelV3:
        """
        Return a new `GroupMetadataModelV3` with the given fields updated.

        Only the constructor-settable fields listed in
        `GroupMetadataModelV3Partial` can be updated; the fixed `zarr_format` /
        `node_type` are rejected at the type level. Each given field fully
        replaces its previous value, including `extra_fields`.
        """
        return dataclasses.replace(self, **kwargs)

    def to_json(self) -> GroupMetadataV3:
        out: GroupMetadataV3 = {
            "zarr_format": self.zarr_format,
            "node_type": self.node_type,
        }
        if len(self.attributes) > 0:
            out["attributes"] = self.attributes
        if self.consolidated_metadata is not UNSET:
            # The consolidated-metadata shape ({kind, must_understand, metadata},
            # no `name`) predates the strict v3.1 extension-field rules, so it is
            # not assignable to `ExtensionFieldV3`; see the discussion on
            # `zarr_metadata.v3.consolidated`.
            out[CONSOLIDATED_METADATA_KEY_V3] = cast(
                "ExtensionFieldV3", self.consolidated_metadata.to_json()
            )
        for key, value in self.extra_fields.items():
            out[key] = value
        return out

    @classmethod
    def from_json(cls, data: object) -> GroupMetadataModelV3:
        parsed = parse_group_metadata_v3(arrays_to_tuples(data))
        # Cast to object: the TypedDict's extra_items type does not admit null,
        # but wild documents (historical zarr-python) contain it.
        consolidated_raw = cast("object", parsed.get(CONSOLIDATED_METADATA_KEY_V3, UNSET))
        consolidated: ConsolidatedMetadataModelV3 | UnsetType
        if consolidated_raw is UNSET or consolidated_raw is None:
            # consolidated_metadata: null was written by a historical
            # zarr-python bug; it gets no model representation. It is read as
            # absence and never written back — repaired, not preserved.
            consolidated = UNSET
        else:
            consolidated = ConsolidatedMetadataModelV3.from_json(consolidated_raw)
        # Sound cast: the TypedDict types all non-standard keys as its
        # `extra_items` (`ExtensionFieldV3`); the comprehension's inferred value
        # type is the union over ALL keys because the key filter cannot narrow it.
        extra_fields = cast(
            "dict[str, ExtensionFieldV3]",
            {
                k: v
                for k, v in parsed.items()
                if k not in GROUP_METADATA_STANDARD_KEYS_V3 and k != CONSOLIDATED_METADATA_KEY_V3
            },
        )
        return cls(
            attributes=dict(parsed.get("attributes", {})),
            consolidated_metadata=consolidated,
            extra_fields=extra_fields,
        )

    @property
    def must_understand_fields(self) -> dict[str, ExtensionFieldV3]:
        """Extra fields the reader is obligated to understand.

        Everything in `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule). A compliant
        reader MUST fail to open the group if this contains any field it does
        not recognize; the model layer only partitions by obligation, since
        recognition is reader-specific.
        """
        return must_understand_subset(self.extra_fields)

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> GroupMetadataModelV3:
        return cls.from_json(load_store_json(mapping, GROUP_METADATA_STORE_KEY_V3))

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        return {
            GROUP_METADATA_STORE_KEY_V3: json.dumps(self.to_json(), indent=indent).encode("utf-8")
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidatedMetadataModelV3:
    """In-memory model of v3 inline consolidated metadata.

    Models the reference-implementation convention where consolidated metadata
    is embedded as an extension field on a group's `zarr.json`. Each entry in
    `metadata` is a complete child document, held as a thin array or group
    model. `must_understand` is typed permissively as `bool` to mirror the
    document shape, but only `False` is valid; this is enforced at runtime.
    """

    kind: Literal["inline"] = field(default="inline", init=False)
    must_understand: bool = False
    metadata: dict[str, ArrayMetadataModelV3 | GroupMetadataModelV3]

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

    def to_json(self) -> ConsolidatedMetadataV3:
        # `must_understand` is emitted as the literal False: the field is typed
        # permissively as `bool`, but `__post_init__` guarantees the value.
        return {
            "kind": self.kind,
            "must_understand": False,
            "metadata": {key: node.to_json() for key, node in self.metadata.items()},
        }

    @classmethod
    def from_json(cls, data: object) -> ConsolidatedMetadataModelV3:
        problems = validate_consolidated_metadata_v3(data)
        if problems:
            raise MetadataValidationError(problems)
        env = cast("Mapping[str, object]", data)
        entries: dict[str, ArrayMetadataModelV3 | GroupMetadataModelV3] = {}
        for key, entry in cast("Mapping[str, object]", env["metadata"]).items():
            node_type = cast("Mapping[str, object]", entry).get("node_type")
            if node_type == "array":
                entries[key] = ArrayMetadataModelV3.from_json(entry)
            else:
                entries[key] = GroupMetadataModelV3.from_json(entry)
        return cls(metadata=entries)


class GroupMetadataModelV2Partial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `GroupMetadataModelV2`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `GroupMetadataModelV2.update` and
    `create_default`. The `init=False` field `zarr_format` is intentionally
    excluded, since it cannot be passed to `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_group.py::test_group_partial_keys_match_settable_model_fields`.
    """

    attributes: dict[str, JSONValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupMetadataModelV2:
    """In-memory model of a v2 group metadata document.

    A canonical, lossless representation of the `.zgroup` content plus the
    sibling `.zattrs` attributes, folded into a single in-memory value
    (mirroring the merged `GroupMetadataV2` document form).
    """

    zarr_format: Literal[2] = field(default=2, init=False)
    attributes: dict[str, JSONValue]

    @classmethod
    def create_default(
        cls, **overrides: Unpack[GroupMetadataModelV2Partial]
    ) -> GroupMetadataModelV2:
        """
        Create a default (empty) v2 group metadata model, with optional overrides.

        The default is a structurally-valid group with no attributes — the group
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`).
        """
        default = cls(attributes={})
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[GroupMetadataModelV2Partial]) -> GroupMetadataModelV2:
        """
        Return a new `GroupMetadataModelV2` with the given fields updated.

        Only the constructor-settable fields listed in
        `GroupMetadataModelV2Partial` can be updated; the fixed `zarr_format`
        is rejected at the type level. Each given field fully replaces its
        previous value.
        """
        return dataclasses.replace(self, **kwargs)

    def to_json(self) -> GroupMetadataV2:
        """Return the merged in-memory document form, INCLUDING `attributes`.

        This is not the on-disk `.zgroup` content: a conforming `.zgroup` must
        exclude `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage.
        """
        out: GroupMetadataV2 = {"zarr_format": self.zarr_format}
        if len(self.attributes) > 0:
            out["attributes"] = self.attributes
        return out

    @classmethod
    def from_json(cls, data: object) -> GroupMetadataModelV2:
        parsed = parse_group_metadata_v2(arrays_to_tuples(data))
        return cls(attributes=dict(parsed.get("attributes", {})))

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> GroupMetadataModelV2:
        zgroup = load_store_json(mapping, GROUP_METADATA_STORE_KEY_V2)
        zattrs: dict[str, JSONValue] = (
            load_store_json(mapping, ATTRIBUTES_STORE_KEY_V2)
            if ATTRIBUTES_STORE_KEY_V2 in mapping
            else {}
        )
        return cls.from_json({**zgroup, "attributes": zattrs})

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        # Attributes live only in the sibling `.zattrs` file; the `.zgroup`
        # document must exclude them.
        zgroup = {k: v for k, v in self.to_json().items() if k != "attributes"}
        return {
            GROUP_METADATA_STORE_KEY_V2: json.dumps(zgroup, indent=indent).encode("utf-8"),
            ATTRIBUTES_STORE_KEY_V2: json.dumps(self.attributes, indent=indent).encode("utf-8"),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidatedMetadataModelV2:
    """In-memory model of a v2 `.zmetadata` document.

    The `metadata` map holds the flat file-keyed entries (`"path/.zarray"`,
    `"path/.zattrs"`, ...) verbatim, preserving byte-faithful round-tripping.
    Entries are deliberately NOT merged into per-node models: which nodes had
    a `.zattrs` file at all is information the canonical representation must
    keep. Interpreting entries into node models is consumer work.
    """

    zarr_consolidated_format: Literal[1] = field(default=1, init=False)
    metadata: dict[str, JSONValue]

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "zarr_consolidated_format": self.zarr_consolidated_format,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: object) -> ConsolidatedMetadataModelV2:
        if not isinstance(data, Mapping):
            raise MetadataValidationError(
                [ValidationProblem((), "expected a mapping", "invalid_type")]
            )
        doc = cast("Mapping[str, object]", data)
        problems: list[ValidationProblem] = [
            ValidationProblem((key,), "missing required key", "missing_key")
            for key in ("zarr_consolidated_format", "metadata")
            if key not in doc
        ]
        if "metadata" in doc:
            entries = doc["metadata"]
            if not isinstance(entries, Mapping) or not all(
                isinstance(k, str) for k in cast("Mapping[object, object]", entries)
            ):
                problems.append(
                    ValidationProblem(
                        ("metadata",), "expected a mapping with string keys", "invalid_type"
                    )
                )
        if problems:
            raise MetadataValidationError(problems)
        entries_tupled = cast(
            "dict[str, JSONValue]",
            arrays_to_tuples(dict(cast("Mapping[str, object]", doc["metadata"]))),
        )
        return cls(metadata=entries_tupled)

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> ConsolidatedMetadataModelV2:
        return cls.from_json(load_store_json(mapping, CONSOLIDATED_METADATA_STORE_KEY_V2))

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        return {
            CONSOLIDATED_METADATA_STORE_KEY_V2: json.dumps(self.to_json(), indent=indent).encode(
                "utf-8"
            )
        }
