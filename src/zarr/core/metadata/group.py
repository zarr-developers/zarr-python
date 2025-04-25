from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, assert_never, cast, get_args

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ZGROUP_JSON,
    ZMETADATA_V2_JSON,
    NodeType,
    ZarrFormat,
)
from zarr.core.config import config
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.common import parse_attributes
from zarr.core.metadata.v3 import Any, Literal, V3JsonEncoder, _replace_special_floats

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer, BufferPrototype


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    """
    Metadata for a Group.
    """

    attributes: dict[str, Any] = field(default_factory=dict)
    zarr_format: ZarrFormat = 3
    consolidated_metadata: ConsolidatedMetadata | None = None
    node_type: Literal[group] = field(default="group", init=False)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        json_indent = config.get("json_indent")
        if self.zarr_format == 3:
            return {
                ZARR_JSON: prototype.buffer.from_bytes(
                    json.dumps(_replace_special_floats(self.to_dict()), cls=V3JsonEncoder).encode()
                )
            }
        else:
            items = {
                ZGROUP_JSON: prototype.buffer.from_bytes(
                    json.dumps({"zarr_format": self.zarr_format}, indent=json_indent).encode()
                ),
                ZATTRS_JSON: prototype.buffer.from_bytes(
                    json.dumps(self.attributes, indent=json_indent).encode()
                ),
            }
            if self.consolidated_metadata:
                d = {
                    ZGROUP_JSON: {"zarr_format": self.zarr_format},
                    ZATTRS_JSON: self.attributes,
                }
                consolidated_metadata = self.consolidated_metadata.to_dict()["metadata"]
                assert isinstance(consolidated_metadata, dict)
                for k, v in consolidated_metadata.items():
                    attrs = v.pop("attributes", None)
                    d[f"{k}/{ZATTRS_JSON}"] = _replace_special_floats(attrs)
                    if "shape" in v:
                        # it's an array
                        d[f"{k}/{ZARRAY_JSON}"] = _replace_special_floats(v)
                    else:
                        d[f"{k}/{ZGROUP_JSON}"] = {
                            "zarr_format": self.zarr_format,
                            "consolidated_metadata": {
                                "metadata": {},
                                "must_understand": False,
                                "kind": "inline",
                            },
                        }

                items[ZMETADATA_V2_JSON] = prototype.buffer.from_bytes(
                    json.dumps(
                        {"metadata": d, "zarr_consolidated_format": 1},
                        cls=V3JsonEncoder,
                    ).encode()
                )

            return items

    def __init__(
        self,
        attributes: dict[str, Any] | None = None,
        zarr_format: ZarrFormat = 3,
        consolidated_metadata: ConsolidatedMetadata | None = None,
    ) -> None:
        attributes_parsed = parse_attributes(attributes)
        zarr_format_parsed = parse_zarr_format(zarr_format)

        object.__setattr__(self, "attributes", attributes_parsed)
        object.__setattr__(self, "zarr_format", zarr_format_parsed)
        object.__setattr__(self, "consolidated_metadata", consolidated_metadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupMetadata:
        data = dict(data)
        assert data.pop("node_type", None) in ("group", None)
        consolidated_metadata = data.pop("consolidated_metadata", None)
        if consolidated_metadata:
            data["consolidated_metadata"] = ConsolidatedMetadata.from_dict(consolidated_metadata)

        zarr_format = data.get("zarr_format")
        if zarr_format == 2 or zarr_format is None:
            # zarr v2 allowed arbitrary keys here.
            # We don't want the GroupMetadata constructor to fail just because someone put an
            # extra key in the metadata.
            expected = {x.name for x in fields(cls)}
            data = {k: v for k, v in data.items() if k in expected}

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(replace(self, consolidated_metadata=None))
        if self.consolidated_metadata:
            result["consolidated_metadata"] = self.consolidated_metadata.to_dict()
        return result


@dataclass(frozen=True)
class ConsolidatedMetadata:
    """
    Consolidated Metadata for this Group.

    This stores the metadata of child nodes below this group. Any child groups
    will have their consolidated metadata set appropriately.
    """

    metadata: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata]
    kind: Literal[inline] = "inline"
    must_understand: Literal[False] = False

    def to_dict(self) -> dict[str, JSON]:
        return {
            "kind": self.kind,
            "must_understand": self.must_understand,
            "metadata": {k: v.to_dict() for k, v in self.flattened_metadata.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> ConsolidatedMetadata:
        data = dict(data)

        kind = data.get("kind")
        if kind != "inline":
            raise ValueError(f"Consolidated metadata kind='{kind}' is not supported.")

        raw_metadata = data.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise TypeError(f"Unexpected type for 'metadata': {type(raw_metadata)}")

        metadata: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {}
        if raw_metadata:
            for k, v in raw_metadata.items():
                if not isinstance(v, dict):
                    raise TypeError(
                        f"Invalid value for metadata items. key='{k}', type='{type(v).__name__}'"
                    )

                # zarr_format is present in v2 and v3.
                zarr_format = parse_zarr_format(v["zarr_format"])

                if zarr_format == 3:
                    node_type = parse_node_type(v.get("node_type", None))
                    if node_type == "group":
                        metadata[k] = GroupMetadata.from_dict(v)
                    elif node_type == "array":
                        metadata[k] = ArrayV3Metadata.from_dict(v)
                    else:
                        assert_never(node_type)
                elif zarr_format == 2:
                    if "shape" in v:
                        metadata[k] = ArrayV2Metadata.from_dict(v)
                    else:
                        metadata[k] = GroupMetadata.from_dict(v)
                else:
                    assert_never(zarr_format)

            cls._flat_to_nested(metadata)

        return cls(metadata=metadata)

    @staticmethod
    def _flat_to_nested(
        metadata: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata],
    ) -> None:
        """
        Convert a flat metadata representation to a nested one.

        Notes
        -----
        Flat metadata is used when persisting the consolidated metadata. The keys
        include the full path, not just the node name. The key prefixes can be
        used to determine which nodes are children of which other nodes.

        Nested metadata is used in-memory. The outermost level will only have the
        *immediate* children of the Group. All nested child groups will be stored
        under the consolidated metadata of their immediate parent.
        """
        # We have a flat mapping from {k: v} where the keys include the *full*
        # path segment:
        #  {
        #    "/a/b": { group_metadata },
        #    "/a/b/array-0": { array_metadata },
        #    "/a/b/array-1": { array_metadata },
        #  }
        #
        # We want to reorganize the metadata such that each Group contains the
        # array metadata of its immediate children.
        # In the example, the group at `/a/b` will have consolidated metadata
        # for its children `array-0` and `array-1`.
        #
        # metadata = dict(metadata)

        keys = sorted(metadata, key=lambda k: k.count("/"))
        grouped = {
            k: list(v) for k, v in itertools.groupby(keys, key=lambda k: k.rsplit("/", 1)[0])
        }

        # we go top down and directly manipulate metadata.
        for key, children_keys in grouped.items():
            # key is a key like "a", "a/b", "a/b/c"
            # The basic idea is to find the immediate parent (so "", "a", or "a/b")
            # and update that node's consolidated metadata to include the metadata
            # in children_keys
            *prefixes, name = key.split("/")
            parent = metadata

            while prefixes:
                # e.g. a/b/c has a parent "a/b". Walk through to get
                # metadata["a"]["b"]
                part = prefixes.pop(0)
                # we can assume that parent[part] here is a group
                # otherwise we wouldn't have a node with this `part` prefix.
                # We can also assume that the parent node will have consolidated metadata,
                # because we're walking top to bottom.
                parent = parent[part].consolidated_metadata.metadata  # type: ignore[union-attr]

            node = parent[name]
            children_keys = list(children_keys)

            if isinstance(node, ArrayV2Metadata | ArrayV3Metadata):
                # These are already present, either thanks to being an array in the
                # root, or by being collected as a child in the else clause
                continue
            children_keys = list(children_keys)
            # We pop from metadata, since we're *moving* this under group
            children = {
                child_key.split("/")[-1]: metadata.pop(child_key)
                for child_key in children_keys
                if child_key != key
            }
            parent[name] = replace(
                node, consolidated_metadata=ConsolidatedMetadata(metadata=children)
            )

    @property
    def flattened_metadata(self) -> dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata]:
        """
        Return the flattened representation of Consolidated Metadata.

        The returned dictionary will have a key for each child node in the hierarchy
        under this group. Under the default (nested) representation available through
        ``self.metadata``, the dictionary only contains keys for immediate children.

        The keys of the dictionary will include the full path to a child node from
        the current group, where segments are joined by ``/``.

        Examples
        --------
        >>> cm = ConsolidatedMetadata(
        ...     metadata={
        ...         "group-0": GroupMetadata(
        ...             consolidated_metadata=ConsolidatedMetadata(
        ...                 {
        ...                     "group-0-0": GroupMetadata(),
        ...                 }
        ...             )
        ...         ),
        ...         "group-1": GroupMetadata(),
        ...     }
        ... )
        {'group-0': GroupMetadata(attributes={}, zarr_format=3, consolidated_metadata=None, node_type='group'),
         'group-0/group-0-0': GroupMetadata(attributes={}, zarr_format=3, consolidated_metadata=None, node_type='group'),
         'group-1': GroupMetadata(attributes={}, zarr_format=3, consolidated_metadata=None, node_type='group')}
        """
        metadata = {}

        def flatten(
            key: str, group: GroupMetadata | ArrayV2Metadata | ArrayV3Metadata
        ) -> dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata]:
            children: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {}
            if isinstance(group, ArrayV2Metadata | ArrayV3Metadata):
                children[key] = group
            else:
                if group.consolidated_metadata and group.consolidated_metadata.metadata is not None:
                    children[key] = replace(
                        group, consolidated_metadata=ConsolidatedMetadata(metadata={})
                    )
                    for name, val in group.consolidated_metadata.metadata.items():
                        full_key = f"{key}/{name}"
                        if isinstance(val, GroupMetadata):
                            children.update(flatten(full_key, val))
                        else:
                            children[full_key] = val
                else:
                    children[key] = replace(group, consolidated_metadata=None)
            return children

        for k, v in self.metadata.items():
            metadata.update(flatten(k, v))

        return metadata


@dataclass(frozen=True)
class ImplicitGroupMarker(GroupMetadata):
    """
    Marker for an implicit group. Instances of this class are only used in the context of group
    creation as a placeholder to represent groups that should only be created if they do not
    already exist in storage
    """


def parse_zarr_format(data: object) -> ZarrFormat:
    """Parse the zarr_format field from metadata."""
    if data in get_args(ZarrFormat):
        return cast(ZarrFormat, data)
    msg = f"Invalid zarr_format. Expected one of 2 or 3. Got {data!r}."
    raise ValueError(msg)


def parse_node_type(data: object) -> NodeType:
    """Parse the node_type field from metadata."""
    if data in get_args(NodeType):
        return cast(NodeType, data)
    msg = f"Invalid node_type. Expected 'array' or 'group'. Got {data!r}."
    raise ValueError(msg)
