from __future__ import annotations

import asyncio
import itertools
import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Literal, TypeVar, assert_never, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

import zarr.api.asynchronous as async_api
from zarr._compat import _deprecate_positional_args
from zarr.abc.metadata import Metadata
from zarr.abc.store import Store, set_or_delete
from zarr.core._info import GroupInfo
from zarr.core.array import (
    Array,
    AsyncArray,
    CompressorLike,
    CompressorsLike,
    FiltersLike,
    SerializerLike,
    ShardsLike,
    _build_parents,
    _parse_deprecated_compressor,
    create_array,
)
from zarr.core.attributes import Attributes
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ZGROUP_JSON,
    ZMETADATA_V2_JSON,
    ChunkCoords,
    NodeType,
    ShapeLike,
    ZarrFormat,
    parse_shapelike,
)
from zarr.core.config import config
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import V3JsonEncoder
from zarr.core.sync import SyncMixin, sync
from zarr.errors import MetadataValidationError
from zarr.storage import StoreLike, StorePath
from zarr.storage._common import ensure_no_existing_node, make_store_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable, Iterator
    from typing import Any

    from zarr.core.array_spec import ArrayConfig, ArrayConfigLike
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding, ChunkKeyEncodingLike
    from zarr.core.common import MemoryOrder

logger = logging.getLogger("zarr.group")

DefaultT = TypeVar("DefaultT")


def parse_zarr_format(data: Any) -> ZarrFormat:
    """Parse the zarr_format field from metadata."""
    if data in (2, 3):
        return cast(ZarrFormat, data)
    msg = f"Invalid zarr_format. Expected one of 2 or 3. Got {data}."
    raise ValueError(msg)


def parse_node_type(data: Any) -> NodeType:
    """Parse the node_type field from metadata."""
    if data in ("array", "group"):
        return cast(Literal["array", "group"], data)
    raise MetadataValidationError("node_type", "array or group", data)


# todo: convert None to empty dict
def parse_attributes(data: Any) -> dict[str, Any]:
    """Parse the attributes field from metadata."""
    if data is None:
        return {}
    elif isinstance(data, dict) and all(isinstance(k, str) for k in data):
        return data
    msg = f"Expected dict with string keys. Got {type(data)} instead."
    raise TypeError(msg)


@overload
def _parse_async_node(node: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]) -> Array: ...


@overload
def _parse_async_node(node: AsyncGroup) -> Group: ...


def _parse_async_node(
    node: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup,
) -> Array | Group:
    """Wrap an AsyncArray in an Array, or an AsyncGroup in a Group."""
    if isinstance(node, AsyncArray):
        return Array(node)
    elif isinstance(node, AsyncGroup):
        return Group(node)
    else:
        raise TypeError(f"Unknown node type, got {type(node)}")


@dataclass(frozen=True)
class ConsolidatedMetadata:
    """
    Consolidated Metadata for this Group.

    This stores the metadata of child nodes below this group. Any child groups
    will have their consolidated metadata set appropriately.
    """

    metadata: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata]
    kind: Literal["inline"] = "inline"
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
class GroupMetadata(Metadata):
    """
    Metadata for a Group.
    """

    attributes: dict[str, Any] = field(default_factory=dict)
    zarr_format: ZarrFormat = 3
    consolidated_metadata: ConsolidatedMetadata | None = None
    node_type: Literal["group"] = field(default="group", init=False)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        json_indent = config.get("json_indent")
        if self.zarr_format == 3:
            return {
                ZARR_JSON: prototype.buffer.from_bytes(
                    json.dumps(self.to_dict(), cls=V3JsonEncoder).encode()
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
                    d[f"{k}/{ZATTRS_JSON}"] = attrs
                    if "shape" in v:
                        # it's an array
                        d[f"{k}/{ZARRAY_JSON}"] = v
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
class AsyncGroup:
    """
    Asynchronous Group object.
    """

    metadata: GroupMetadata
    store_path: StorePath

    @classmethod
    async def from_store(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] | None = None,
        overwrite: bool = False,
        zarr_format: ZarrFormat = 3,
    ) -> AsyncGroup:
        store_path = await make_store_path(store)

        if overwrite:
            if store_path.store.supports_deletes:
                await store_path.delete_dir()
            else:
                await ensure_no_existing_node(store_path, zarr_format=zarr_format)
        else:
            await ensure_no_existing_node(store_path, zarr_format=zarr_format)
        attributes = attributes or {}
        group = cls(
            metadata=GroupMetadata(attributes=attributes, zarr_format=zarr_format),
            store_path=store_path,
        )
        await group._save_metadata(ensure_parents=True)
        return group

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
        use_consolidated: bool | str | None = None,
    ) -> AsyncGroup:
        """Open a new AsyncGroup

        Parameters
        ----------
        store : StoreLike
        zarr_format : {2, 3}, optional
        use_consolidated : bool or str, default None
            Whether to use consolidated metadata.

            By default, consolidated metadata is used if it's present in the
            store (in the ``zarr.json`` for Zarr format 3 and in the ``.zmetadata`` file
            for Zarr format 2).

            To explicitly require consolidated metadata, set ``use_consolidated=True``,
            which will raise an exception if consolidated metadata is not found.

            To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
            which will fall back to using the regular, non consolidated metadata.

            Zarr format 2 allowed configuring the key storing the consolidated metadata
            (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
            to load consolidated metadata from a non-default key.
        """
        store_path = await make_store_path(store)

        consolidated_key = ZMETADATA_V2_JSON

        if (zarr_format == 2 or zarr_format is None) and isinstance(use_consolidated, str):
            consolidated_key = use_consolidated

        if zarr_format == 2:
            paths = [store_path / ZGROUP_JSON, store_path / ZATTRS_JSON]
            if use_consolidated or use_consolidated is None:
                paths.append(store_path / consolidated_key)

            zgroup_bytes, zattrs_bytes, *rest = await asyncio.gather(
                *[path.get() for path in paths]
            )
            if zgroup_bytes is None:
                raise FileNotFoundError(store_path)

            if use_consolidated or use_consolidated is None:
                maybe_consolidated_metadata_bytes = rest[0]

            else:
                maybe_consolidated_metadata_bytes = None

        elif zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise FileNotFoundError(store_path)
        elif zarr_format is None:
            (
                zarr_json_bytes,
                zgroup_bytes,
                zattrs_bytes,
                maybe_consolidated_metadata_bytes,
            ) = await asyncio.gather(
                (store_path / ZARR_JSON).get(),
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
                (store_path / str(consolidated_key)).get(),
            )
            if zarr_json_bytes is not None and zgroup_bytes is not None:
                # warn and favor v3
                msg = f"Both zarr.json (Zarr format 3) and .zgroup (Zarr format 2) metadata objects exist at {store_path}. Zarr format 3 will be used."
                warnings.warn(msg, stacklevel=1)
            if zarr_json_bytes is None and zgroup_bytes is None:
                raise FileNotFoundError(
                    f"could not find zarr.json or .zgroup objects in {store_path}"
                )
            # set zarr_format based on which keys were found
            if zarr_json_bytes is not None:
                zarr_format = 3
            else:
                zarr_format = 2
        else:
            raise MetadataValidationError("zarr_format", "2, 3, or None", zarr_format)

        if zarr_format == 2:
            # this is checked above, asserting here for mypy
            assert zgroup_bytes is not None

            if use_consolidated and maybe_consolidated_metadata_bytes is None:
                # the user requested consolidated metadata, but it was missing
                raise ValueError(consolidated_key)

            elif use_consolidated is False:
                # the user explicitly opted out of consolidated_metadata.
                # Discard anything we might have read.
                maybe_consolidated_metadata_bytes = None

            return cls._from_bytes_v2(
                store_path, zgroup_bytes, zattrs_bytes, maybe_consolidated_metadata_bytes
            )
        else:
            # V3 groups are comprised of a zarr.json object
            assert zarr_json_bytes is not None
            if not isinstance(use_consolidated, bool | None):
                raise TypeError("use_consolidated must be a bool or None for Zarr format 3.")

            return cls._from_bytes_v3(
                store_path,
                zarr_json_bytes,
                use_consolidated=use_consolidated,
            )

    @classmethod
    def _from_bytes_v2(
        cls,
        store_path: StorePath,
        zgroup_bytes: Buffer,
        zattrs_bytes: Buffer | None,
        consolidated_metadata_bytes: Buffer | None,
    ) -> AsyncGroup:
        # V2 groups are comprised of a .zgroup and .zattrs objects
        zgroup = json.loads(zgroup_bytes.to_bytes())
        zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}
        group_metadata = {**zgroup, "attributes": zattrs}

        if consolidated_metadata_bytes is not None:
            v2_consolidated_metadata = json.loads(consolidated_metadata_bytes.to_bytes())
            v2_consolidated_metadata = v2_consolidated_metadata["metadata"]
            # We already read zattrs and zgroup. Should we ignore these?
            v2_consolidated_metadata.pop(".zattrs", None)
            v2_consolidated_metadata.pop(".zgroup", None)

            consolidated_metadata: defaultdict[str, dict[str, Any]] = defaultdict(dict)

            # keys like air/.zarray, air/.zattrs
            for k, v in v2_consolidated_metadata.items():
                path, kind = k.rsplit("/.", 1)

                if kind == "zarray":
                    consolidated_metadata[path].update(v)
                elif kind == "zattrs":
                    consolidated_metadata[path]["attributes"] = v
                elif kind == "zgroup":
                    consolidated_metadata[path].update(v)
                else:
                    raise ValueError(f"Invalid file type '{kind}' at path '{path}")
            group_metadata["consolidated_metadata"] = {
                "metadata": dict(consolidated_metadata),
                "kind": "inline",
                "must_understand": False,
            }

        return cls.from_dict(store_path, group_metadata)

    @classmethod
    def _from_bytes_v3(
        cls,
        store_path: StorePath,
        zarr_json_bytes: Buffer,
        use_consolidated: bool | None,
    ) -> AsyncGroup:
        group_metadata = json.loads(zarr_json_bytes.to_bytes())
        if use_consolidated and group_metadata.get("consolidated_metadata") is None:
            msg = f"Consolidated metadata requested with 'use_consolidated=True' but not found in '{store_path.path}'."
            raise ValueError(msg)

        elif use_consolidated is False:
            # Drop consolidated metadata if it's there.
            group_metadata.pop("consolidated_metadata", None)

        return cls.from_dict(store_path, group_metadata)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, Any],
    ) -> AsyncGroup:
        return cls(
            metadata=GroupMetadata.from_dict(data),
            store_path=store_path,
        )

    async def setitem(self, key: str, value: Any) -> None:
        """
        Fastpath for creating a new array
        New arrays will be created with default array settings for the array type.

        Parameters
        ----------
        key : str
            Array name
        value : array-like
            Array data
        """
        path = self.store_path / key
        await async_api.save_array(
            store=path, arr=value, zarr_format=self.metadata.zarr_format, overwrite=True
        )

    async def getitem(
        self,
        key: str,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup:
        """
        Get a subarray or subgroup from the group.

        Parameters
        ----------
        key : str
            Array or group name

        Returns
        -------
        AsyncArray or AsyncGroup
        """
        store_path = self.store_path / key
        logger.debug("key=%s, store_path=%s", key, store_path)
        metadata: ArrayV2Metadata | ArrayV3Metadata | GroupMetadata

        # Consolidated metadata lets us avoid some I/O operations so try that first.
        if self.metadata.consolidated_metadata is not None:
            return self._getitem_consolidated(store_path, key, prefix=self.name)

        # Note:
        # in zarr-python v2, we first check if `key` references an Array, else if `key` references
        # a group,using standalone `contains_array` and `contains_group` functions. These functions
        # are reusable, but for v3 they would perform redundant I/O operations.
        # Not clear how much of that strategy we want to keep here.
        elif self.metadata.zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise KeyError(key)
            else:
                zarr_json = json.loads(zarr_json_bytes.to_bytes())
                metadata = _build_metadata_v3(zarr_json)
                return _build_node_v3(metadata, store_path)

        elif self.metadata.zarr_format == 2:
            # Q: how do we like optimistically fetching .zgroup, .zarray, and .zattrs?
            # This guarantees that we will always make at least one extra request to the store
            zgroup_bytes, zarray_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZARRAY_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
            )

            if zgroup_bytes is None and zarray_bytes is None:
                raise KeyError(key)

            # unpack the zarray, if this is None then we must be opening a group
            zarray = json.loads(zarray_bytes.to_bytes()) if zarray_bytes else None
            zgroup = json.loads(zgroup_bytes.to_bytes()) if zgroup_bytes else None
            # unpack the zattrs, this can be None if no attrs were written
            zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}

            if zarray is not None:
                metadata = _build_metadata_v2(zarray, zattrs)
                return _build_node_v2(metadata=metadata, store_path=store_path)
            else:
                # this is just for mypy
                if TYPE_CHECKING:
                    assert zgroup is not None
                metadata = _build_metadata_v2(zgroup, zattrs)
                return _build_node_v2(metadata=metadata, store_path=store_path)
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

    def _getitem_consolidated(
        self, store_path: StorePath, key: str, prefix: str
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup:
        # getitem, in the special case where we have consolidated metadata.
        # Note that this is a regular def (non async) function.
        # This shouldn't do any additional I/O.

        # the caller needs to verify this!
        assert self.metadata.consolidated_metadata is not None

        # we support nested getitems like group/subgroup/array
        indexers = key.split("/")
        indexers.reverse()
        metadata: ArrayV2Metadata | ArrayV3Metadata | GroupMetadata = self.metadata

        while indexers:
            indexer = indexers.pop()
            if isinstance(metadata, ArrayV2Metadata | ArrayV3Metadata):
                # we've indexed into an array with group["array/subarray"]. Invalid.
                raise KeyError(key)
            if metadata.consolidated_metadata is None:
                # we've indexed into a group without consolidated metadata.
                # This isn't normal; typically, consolidated metadata
                # will include explicit markers for when there are no child
                # nodes as metadata={}.
                # We have some freedom in exactly how we interpret this case.
                # For now, we treat None as the same as {}, i.e. we don't
                # have any children.
                raise KeyError(key)
            try:
                metadata = metadata.consolidated_metadata.metadata[indexer]
            except KeyError as e:
                # The Group Metadata has consolidated metadata, but the key
                # isn't present. We trust this to mean that the key isn't in
                # the hierarchy, and *don't* fall back to checking the store.
                msg = f"'{key}' not found in consolidated metadata."
                raise KeyError(msg) from e

        # update store_path to ensure that AsyncArray/Group.name is correct
        if prefix != "/":
            key = "/".join([prefix.lstrip("/"), key])
        store_path = StorePath(store=store_path.store, path=key)

        if isinstance(metadata, GroupMetadata):
            return AsyncGroup(metadata=metadata, store_path=store_path)
        else:
            return AsyncArray(metadata=metadata, store_path=store_path)

    async def delitem(self, key: str) -> None:
        """Delete a group member.

        Parameters
        ----------
        key : str
            Array or group name
        """
        store_path = self.store_path / key

        await store_path.delete_dir()
        if self.metadata.consolidated_metadata:
            self.metadata.consolidated_metadata.metadata.pop(key, None)
            await self._save_metadata()

    async def get(
        self, key: str, default: DefaultT | None = None
    ) -> AsyncArray[Any] | AsyncGroup | DefaultT | None:
        """Obtain a group member, returning default if not found.

        Parameters
        ----------
        key : str
            Group member name.
        default : object
            Default value to return if key is not found (default: None).

        Returns
        -------
        object
            Group member (AsyncArray or AsyncGroup) or default if not found.
        """
        try:
            return await self.getitem(key)
        except KeyError:
            return default

    async def _save_metadata(self, ensure_parents: bool = False) -> None:
        to_save = self.metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]

        if ensure_parents:
            parents = _build_parents(self)
            for parent in parents:
                awaitables.extend(
                    [
                        (parent.store_path / key).set_if_not_exists(value)
                        for key, value in parent.metadata.to_buffer_dict(
                            default_buffer_prototype()
                        ).items()
                    ]
                )

        await asyncio.gather(*awaitables)

    @property
    def path(self) -> str:
        """Storage path."""
        return self.store_path.path

    @property
    def name(self) -> str:
        """Group name following h5py convention."""
        if self.path:
            # follow h5py convention: add leading slash
            name = self.path
            if name[0] != "/":
                name = "/" + name
            return name
        return "/"

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self.name.split("/")[-1]

    @property
    def attrs(self) -> dict[str, Any]:
        return self.metadata.attributes

    @property
    def info(self) -> Any:
        """
        Return a visual representation of the statically known information about a group.

        Note that this doesn't include dynamic information, like the number of child
        Groups or Arrays.

        Returns
        -------
        GroupInfo

        See Also
        --------
        AsyncGroup.info_complete
            All information about a group, including dynamic information
        """

        if self.metadata.consolidated_metadata:
            members = list(self.metadata.consolidated_metadata.flattened_metadata.values())
        else:
            members = None
        return self._info(members=members)

    async def info_complete(self) -> Any:
        """
        Return all the information for a group.

        This includes dynamic information like the number
        of child Groups or Arrays. If this group doesn't contain consolidated
        metadata then this will need to read from the backing Store.

        Returns
        -------
        GroupInfo

        See Also
        --------
        AsyncGroup.info
        """
        members = [x[1].metadata async for x in self.members(max_depth=None)]
        return self._info(members=members)

    def _info(
        self, members: list[ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] | None = None
    ) -> Any:
        kwargs = {}
        if members is not None:
            kwargs["_count_members"] = len(members)
            count_arrays = 0
            count_groups = 0
            for member in members:
                if isinstance(member, GroupMetadata):
                    count_groups += 1
                else:
                    count_arrays += 1
            kwargs["_count_arrays"] = count_arrays
            kwargs["_count_groups"] = count_groups

        return GroupInfo(
            _name=self.store_path.path,
            _read_only=self.read_only,
            _store_type=type(self.store_path.store).__name__,
            _zarr_format=self.metadata.zarr_format,
            # maybe do a typeddict
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def store(self) -> Store:
        return self.store_path.store

    @property
    def read_only(self) -> bool:
        # Backwards compatibility for 2.x
        return self.store_path.read_only

    @property
    def synchronizer(self) -> None:
        # Backwards compatibility for 2.x
        # Not implemented in 3.x yet.
        return None

    async def create_group(
        self,
        name: str,
        *,
        overwrite: bool = False,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGroup:
        """Create a sub-group.

        Parameters
        ----------
        name : str
            Group name.
        overwrite : bool, optional
            If True, do not raise an error if the group already exists.
        attributes : dict, optional
            Group attributes.

        Returns
        -------
        g : AsyncGroup
        """
        attributes = attributes or {}
        return await type(self).from_store(
            self.store_path / name,
            attributes=attributes,
            overwrite=overwrite,
            zarr_format=self.metadata.zarr_format,
        )

    async def require_group(self, name: str, overwrite: bool = False) -> AsyncGroup:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : str
            Group name.
        overwrite : bool, optional
            Overwrite any existing group with given `name` if present.

        Returns
        -------
        g : AsyncGroup
        """
        if overwrite:
            # TODO: check that overwrite=True errors if an array exists where the group is being created
            grp = await self.create_group(name, overwrite=True)
        else:
            try:
                item: (
                    AsyncGroup | AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]
                ) = await self.getitem(name)
                if not isinstance(item, AsyncGroup):
                    raise TypeError(
                        f"Incompatible object ({item.__class__.__name__}) already exists"
                    )
                assert isinstance(item, AsyncGroup)  # make mypy happy
                grp = item
            except KeyError:
                grp = await self.create_group(name)
        return grp

    async def require_groups(self, *names: str) -> tuple[AsyncGroup, ...]:
        """Convenience method to require multiple groups in a single call.

        Parameters
        ----------
        *names : str
            Group names.

        Returns
        -------
        Tuple[AsyncGroup, ...]
        """
        if not names:
            return ()
        return tuple(await asyncio.gather(*(self.require_group(name) for name in names)))

    async def create_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords | Literal["auto"] = "auto",
        shards: ShardsLike | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = "auto",
        serializer: SerializerLike = "auto",
        fill_value: Any | None = 0,
        order: MemoryOrder | None = None,
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array within this group.

        This method lightly wraps :func:`zarr.core.array.create_array`.

        Parameters
        ----------
        name : str
            The name of the array relative to the group. If ``path`` is ``None``, the array will be located
            at the root of the store.
        shape : ChunkCoords
            Shape of the array.
        dtype : npt.DTypeLike
            Data type of the array.
        chunks : ChunkCoords, optional
            Chunk shape of the array.
            If not specified, default are guessed based on the shape and dtype.
        shards : ChunkCoords, optional
            Shard shape of the array. The default value of ``None`` results in no sharding at all.
        filters : Iterable[Codec], optional
            Iterable of filters to apply to each chunk of the array, in order, before serializing that
            chunk to bytes.

            For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
            and these values must be instances of ``ArrayArrayCodec``, or dict representations
            of ``ArrayArrayCodec``.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.

            For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
            the order if your filters is consistent with the behavior of each filter.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.
        compressors : Iterable[Codec], optional
            List of compressors to apply to the array. Compressors are applied in order, and after any
            filters are applied (if any are specified) and the data is serialized into bytes.

            For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
            returns another bytestream. Multiple compressors my be provided for Zarr format 3.
            If no ``compressors`` are provided, a default set of compressors will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_compressors``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default compressors.

            For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
            be provided for Zarr format 2.
            If no ``compressor`` is provided, a default compressor will be used.
            in :mod:`zarr.core.config`.
            Use ``None`` to omit the default compressor.
        compressor : Codec, optional
            Deprecated in favor of ``compressors``.
        serializer : dict[str, JSON] | ArrayBytesCodec, optional
            Array-to-bytes codec to use for encoding the array data.
            Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
            If no ``serializer`` is provided, a default serializer will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_serializer``
            in :mod:`zarr.core.config`.
        fill_value : Any, optional
            Fill value for the array.
        order : {"C", "F"}, optional
            The memory of the array (default is "C").
            For Zarr format 2, this parameter sets the memory order of the array.
            For Zarr format 3, this parameter is deprecated, because memory order
            is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
            order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
            If no ``order`` is provided, a default order will be used.
            This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
        attributes : dict, optional
            Attributes for the array.
        chunk_key_encoding : ChunkKeyEncoding, optional
            A specification of how the chunk keys are represented in storage.
            For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
            For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        storage_options : dict, optional
            If using an fsspec URL to create the store, these will be passed to the backend implementation.
            Ignored otherwise.
        overwrite : bool, default False
            Whether to overwrite an array with the same name in the store, if one exists.
        config : ArrayConfig or ArrayConfigLike, optional
            Runtime configuration for the array.

        Returns
        -------
        AsyncArray

        """
        compressors = _parse_deprecated_compressor(
            compressor, compressors, zarr_format=self.metadata.zarr_format
        )
        return await create_array(
            store=self.store_path,
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            fill_value=fill_value,
            order=order,
            zarr_format=self.metadata.zarr_format,
            attributes=attributes,
            chunk_key_encoding=chunk_key_encoding,
            dimension_names=dimension_names,
            storage_options=storage_options,
            overwrite=overwrite,
            config=config,
        )

    @deprecated("Use AsyncGroup.create_array instead.")
    async def create_dataset(
        self, name: str, *, shape: ShapeLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.create_array` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.require_dataset` method.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs : dict
            Additional arguments passed to :func:`zarr.AsyncGroup.create_array`.

        Returns
        -------
        a : AsyncArray
        """
        data = kwargs.pop("data", None)
        # create_dataset in zarr 2.x requires shape but not dtype if data is
        # provided. Allow this configuration by inferring dtype from data if
        # necessary and passing it to create_array
        if "dtype" not in kwargs and data is not None:
            kwargs["dtype"] = data.dtype
        array = await self.create_array(name, shape=shape, **kwargs)
        if data is not None:
            await array.setitem(slice(None), data)
        return array

    @deprecated("Use AsyncGroup.require_array instead.")
    async def require_dataset(
        self,
        name: str,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike = None,
        exact: bool = False,
        **kwargs: Any,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Obtain an array, creating if it doesn't exist.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.require_dataset` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : str or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : AsyncArray
        """
        return await self.require_array(name, shape=shape, dtype=dtype, exact=exact, **kwargs)

    async def require_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike = None,
        exact: bool = False,
        **kwargs: Any,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Obtain an array, creating if it doesn't exist.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : str or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : AsyncArray
        """
        try:
            ds = await self.getitem(name)
            if not isinstance(ds, AsyncArray):
                raise TypeError(f"Incompatible object ({ds.__class__.__name__}) already exists")

            shape = parse_shapelike(shape)
            if shape != ds.shape:
                raise TypeError(f"Incompatible shape ({ds.shape} vs {shape})")

            dtype = np.dtype(dtype)
            if exact:
                if ds.dtype != dtype:
                    raise TypeError(f"Incompatible dtype ({ds.dtype} vs {dtype})")
            else:
                if not np.can_cast(ds.dtype, dtype):
                    raise TypeError(f"Incompatible dtype ({ds.dtype} vs {dtype})")
        except KeyError:
            ds = await self.create_array(name, shape=shape, dtype=dtype, **kwargs)

        return ds

    async def update_attributes(self, new_attributes: dict[str, Any]) -> AsyncGroup:
        """Update group attributes.

        Parameters
        ----------
        new_attributes : dict
            New attributes to set on the group.

        Returns
        -------
        self : AsyncGroup
        """
        # metadata.attributes is "frozen" so we simply clear and update the dict
        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        await self._save_metadata()

        return self

    def __repr__(self) -> str:
        return f"<AsyncGroup {self.store_path}>"

    async def nmembers(
        self,
        max_depth: int | None = 0,
    ) -> int:
        """Count the number of members in this group.

        Parameters
        ----------
        max_depth : int, default 0
            The maximum number of levels of the hierarchy to include. By
            default, (``max_depth=0``) only immediate children are included. Set
            ``max_depth=None`` to include all nodes, and some positive integer
            to consider children within that many levels of the root Group.

        Returns
        -------
        count : int
        """
        # check if we can use consolidated metadata, which requires that we have non-None
        # consolidated metadata at all points in the hierarchy.
        if self.metadata.consolidated_metadata is not None:
            return len(self.metadata.consolidated_metadata.flattened_metadata)
        # TODO: consider using aioitertools.builtins.sum for this
        # return await aioitertools.builtins.sum((1 async for _ in self.members()), start=0)
        n = 0
        async for _ in self.members(max_depth=max_depth):
            n += 1
        return n

    async def members(
        self,
        max_depth: int | None = 0,
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup],
        None,
    ]:
        """
        Returns an AsyncGenerator over the arrays and groups contained in this group.
        This method requires that `store_path.store` supports directory listing.

        The results are not guaranteed to be ordered.

        Parameters
        ----------
        max_depth : int, default 0
            The maximum number of levels of the hierarchy to include. By
            default, (``max_depth=0``) only immediate children are included. Set
            ``max_depth=None`` to include all nodes, and some positive integer
            to consider children within that many levels of the root Group.

        Returns
        -------
        path:
            A string giving the path to the target, relative to the Group ``self``.
        value: AsyncArray or AsyncGroup
            The AsyncArray or AsyncGroup that is a child of ``self``.
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be None or >= 0. Got '{max_depth}' instead")
        async for item in self._members(max_depth=max_depth):
            yield item

    def _members_consolidated(
        self, max_depth: int | None, prefix: str = ""
    ) -> Generator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup],
        None,
    ]:
        consolidated_metadata = self.metadata.consolidated_metadata

        do_recursion = max_depth is None or max_depth > 0

        # we kind of just want the top-level keys.
        if consolidated_metadata is not None:
            for key in consolidated_metadata.metadata:
                obj = self._getitem_consolidated(
                    self.store_path, key, prefix=self.name
                )  # Metadata -> Group/Array
                key = f"{prefix}/{key}".lstrip("/")
                yield key, obj

                if do_recursion and isinstance(obj, AsyncGroup):
                    if max_depth is None:
                        new_depth = None
                    else:
                        new_depth = max_depth - 1
                    yield from obj._members_consolidated(new_depth, prefix=key)

    async def _members(
        self, max_depth: int | None
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
    ]:
        skip_keys: tuple[str, ...]
        if self.metadata.zarr_format == 2:
            skip_keys = (".zattrs", ".zgroup", ".zarray", ".zmetadata")
        elif self.metadata.zarr_format == 3:
            skip_keys = ("zarr.json",)
        else:
            raise ValueError(f"Unknown Zarr format: {self.metadata.zarr_format}")

        if self.metadata.consolidated_metadata is not None:
            members = self._members_consolidated(max_depth=max_depth)
            for member in members:
                yield member
            return

        if not self.store_path.store.supports_listing:
            msg = (
                f"The store associated with this group ({type(self.store_path.store)}) "
                "does not support listing, "
                "specifically via the `list_dir` method. "
                "This function requires a store that supports listing."
            )

            raise ValueError(msg)
        # enforce a concurrency limit by passing a semaphore to all the recursive functions
        semaphore = asyncio.Semaphore(config.get("async.concurrency"))
        async for member in _iter_members_deep(
            self, max_depth=max_depth, skip_keys=skip_keys, semaphore=semaphore
        ):
            yield member

    async def keys(self) -> AsyncGenerator[str, None]:
        """Iterate over member names."""
        async for key, _ in self.members():
            yield key

    async def contains(self, member: str) -> bool:
        """Check if a member exists in the group.

        Parameters
        ----------
        member : str
            Member name.

        Returns
        -------
        bool
        """
        # TODO: this can be made more efficient.
        try:
            await self.getitem(member)
        except KeyError:
            return False
        else:
            return True

    async def groups(self) -> AsyncGenerator[tuple[str, AsyncGroup], None]:
        """Iterate over subgroups."""
        async for name, value in self.members():
            if isinstance(value, AsyncGroup):
                yield name, value

    async def group_keys(self) -> AsyncGenerator[str, None]:
        """Iterate over group names."""
        async for key, _ in self.groups():
            yield key

    async def group_values(self) -> AsyncGenerator[AsyncGroup, None]:
        """Iterate over group values."""
        async for _, group in self.groups():
            yield group

    async def arrays(
        self,
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]], None
    ]:
        """Iterate over arrays."""
        async for key, value in self.members():
            if isinstance(value, AsyncArray):
                yield key, value

    async def array_keys(self) -> AsyncGenerator[str, None]:
        """Iterate over array names."""
        async for key, _ in self.arrays():
            yield key

    async def array_values(
        self,
    ) -> AsyncGenerator[AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata], None]:
        """Iterate over array values."""
        async for _, array in self.arrays():
            yield array

    async def tree(self, expand: bool | None = None, level: int | None = None) -> Any:
        """
        Return a tree-like representation of a hierarchy.

        This requires the optional ``rich`` dependency.

        Parameters
        ----------
        expand : bool, optional
            This keyword is not yet supported. A NotImplementedError is raised if
            it's used.
        level : int, optional
            The maximum depth below this Group to display in the tree.

        Returns
        -------
        TreeRepr
            A pretty-printable object displaying the hierarchy.
        """
        from zarr.core._tree import group_tree_async

        if expand is not None:
            raise NotImplementedError("'expand' is not yet implemented.")
        return await group_tree_async(self, max_depth=level)

    async def empty(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an empty array with the specified shape in this Group. The contents will
        be filled with the array's fill value or zeros if no fill value is provided.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Notes
        -----
        The contents of an empty Zarr array are not defined. On attempting to
        retrieve data from an empty Zarr array, any values may be returned,
        and these are not guaranteed to be stable from one access to the next.
        """
        return await async_api.empty(shape=shape, store=self.store_path, path=name, **kwargs)

    async def zeros(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with zero being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.zeros(shape=shape, store=self.store_path, path=name, **kwargs)

    async def ones(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with one being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.ones(shape=shape, store=self.store_path, path=name, **kwargs)

    async def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with "fill_value" being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        fill_value : scalar
            Value to fill the array with.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.full(
            shape=shape,
            fill_value=fill_value,
            store=self.store_path,
            path=name,
            **kwargs,
        )

    async def empty_like(
        self, *, name: str, data: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an empty sub-array like `data`. The contents will be filled with
        the array's fill value or zeros if no fill value is provided.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create an empty array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.empty_like(a=data, store=self.store_path, path=name, **kwargs)

    async def zeros_like(
        self, *, name: str, data: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array of zeros like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.zeros_like(a=data, store=self.store_path, path=name, **kwargs)

    async def ones_like(
        self, *, name: str, data: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array of ones like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.ones_like(a=data, store=self.store_path, path=name, **kwargs)

    async def full_like(
        self, *, name: str, data: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array like `data` filled with the `fill_value` of `data` .

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await async_api.full_like(a=data, store=self.store_path, path=name, **kwargs)

    async def move(self, source: str, dest: str) -> None:
        """Move a sub-group or sub-array from one path to another.

        Notes
        -----
        Not implemented
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Group(SyncMixin):
    _async_group: AsyncGroup

    @classmethod
    def from_store(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] | None = None,
        zarr_format: ZarrFormat = 3,
        overwrite: bool = False,
    ) -> Group:
        """Instantiate a group from an initialized store.

        Parameters
        ----------
        store : StoreLike
            StoreLike containing the Group.
        attributes : dict, optional
            A dictionary of JSON-serializable values with user-defined attributes.
        zarr_format : {2, 3}, optional
            Zarr storage format version.
        overwrite : bool, optional
            If True, do not raise an error if the group already exists.

        Returns
        -------
        Group
            Group instantiated from the store.

        Raises
        ------
        ContainsArrayError, ContainsGroupError, ContainsArrayAndGroupError
        """
        attributes = attributes or {}
        obj = sync(
            AsyncGroup.from_store(
                store,
                attributes=attributes,
                overwrite=overwrite,
                zarr_format=zarr_format,
            ),
        )

        return cls(obj)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
    ) -> Group:
        """Open a group from an initialized store.

        Parameters
        ----------
        store : StoreLike
            Store containing the Group.
        zarr_format : {2, 3, None}, optional
            Zarr storage format version.

        Returns
        -------
        Group
            Group instantiated from the store.
        """
        obj = sync(AsyncGroup.open(store, zarr_format=zarr_format))
        return cls(obj)

    def __getitem__(self, path: str) -> Array | Group:
        """Obtain a group member.

        Parameters
        ----------
        path : str
            Group member name.

        Returns
        -------
        Array | Group
            Group member (Array or Group) at the specified key

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group.create_group(name="subgroup").create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group["subarray"]
        <Array memory://132270269438272/subarray shape=(10,) dtype=float64>
        >>> group["subgroup"]
        <Group memory://132270269438272/subgroup>
        >>> group["subgroup"]["subarray"]
        <Array memory://132270269438272/subgroup/subarray shape=(10,) dtype=float64>

        """
        obj = self._sync(self._async_group.getitem(path))
        if isinstance(obj, AsyncArray):
            return Array(obj)
        else:
            return Group(obj)

    def get(self, path: str, default: DefaultT | None = None) -> Array | Group | DefaultT | None:
        """Obtain a group member, returning default if not found.

        Parameters
        ----------
        path : str
            Group member name.
        default : object
            Default value to return if key is not found (default: None).

        Returns
        -------
        object
            Group member (Array or Group) or default if not found.

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group.create_group(name="subgroup")
        >>> group.get("subarray")
        <Array memory://132270269438272/subarray shape=(10,) dtype=float64>
        >>> group.get("subgroup")
        <Group memory://132270269438272/subgroup>
        >>> group.get("nonexistent", None)

        """
        try:
            return self[path]
        except KeyError:
            return default

    def __delitem__(self, key: str) -> None:
        """Delete a group member.

        Parameters
        ----------
        key : str
            Group member name.

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> del group["subarray"]
        >>> "subarray" in group
        False
        """
        self._sync(self._async_group.delitem(key))

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over group member names.
        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_array('baz', shape=(10,), chunks=(10,))
        >>> d2 = g1.create_array('quux', shape=(10,), chunks=(10,))
        >>> for name in g1:
        ...     print(name)
        baz
        bar
        foo
        quux
        """
        yield from self.keys()

    def __len__(self) -> int:
        """Number of members."""
        return self.nmembers()

    def __setitem__(self, key: str, value: Any) -> None:
        """Fastpath for creating a new array.

        New arrays will be created using default settings for the array type.
        If you need to create an array with custom settings, use the `create_array` method.

        Parameters
        ----------
        key : str
            Array name.
        value : Any
            Array data.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group["foo"] = zarr.zeros((10,))
        >>> group["foo"]
        <Array memory://132270269438272/foo shape=(10,) dtype=float64>
        """
        self._sync(self._async_group.setitem(key, value))

    def __repr__(self) -> str:
        return f"<Group {self.store_path}>"

    async def update_attributes_async(self, new_attributes: dict[str, Any]) -> Group:
        """Update the attributes of this group.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> await group.update_attributes_async({"foo": "bar"})
        >>> group.attrs.asdict()
        {'foo': 'bar'}
        """
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        to_save = new_metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
        await asyncio.gather(*awaitables)

        async_group = replace(self._async_group, metadata=new_metadata)
        return replace(self, _async_group=async_group)

    @property
    def store_path(self) -> StorePath:
        """Path-like interface for the Store."""
        return self._async_group.store_path

    @property
    def metadata(self) -> GroupMetadata:
        """Group metadata."""
        return self._async_group.metadata

    @property
    def path(self) -> str:
        """Storage path."""
        return self._async_group.path

    @property
    def name(self) -> str:
        """Group name following h5py convention."""
        return self._async_group.name

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self._async_group.basename

    @property
    def attrs(self) -> Attributes:
        """Attributes of this Group"""
        return Attributes(self)

    @property
    def info(self) -> Any:
        """
        Return the statically known information for a group.

        Returns
        -------
        GroupInfo

        See Also
        --------
        Group.info_complete
            All information about a group, including dynamic information
            like the children members.
        """
        return self._async_group.info

    def info_complete(self) -> Any:
        """
        Return information for a group.

        If this group doesn't contain consolidated metadata then
        this will need to read from the backing Store.

        Returns
        -------
        GroupInfo

        See Also
        --------
        Group.info
        """
        return self._sync(self._async_group.info_complete())

    @property
    def store(self) -> Store:
        # Backwards compatibility for 2.x
        return self._async_group.store

    @property
    def read_only(self) -> bool:
        # Backwards compatibility for 2.x
        return self._async_group.read_only

    @property
    def synchronizer(self) -> None:
        # Backwards compatibility for 2.x
        # Not implemented in 3.x yet.
        return self._async_group.synchronizer

    def update_attributes(self, new_attributes: dict[str, Any]) -> Group:
        """Update the attributes of this group.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.update_attributes({"foo": "bar"})
        >>> group.attrs.asdict()
        {'foo': 'bar'}
        """
        self._sync(self._async_group.update_attributes(new_attributes))
        return self

    def nmembers(self, max_depth: int | None = 0) -> int:
        """Count the number of members in this group.

        Parameters
        ----------
        max_depth : int, default 0
            The maximum number of levels of the hierarchy to include. By
            default, (``max_depth=0``) only immediate children are included. Set
            ``max_depth=None`` to include all nodes, and some positive integer
            to consider children within that many levels of the root Group.

        Returns
        -------
        count : int
        """

        return self._sync(self._async_group.nmembers(max_depth=max_depth))

    def members(self, max_depth: int | None = 0) -> tuple[tuple[str, Array | Group], ...]:
        """
        Return the sub-arrays and sub-groups of this group as a tuple of (name, array | group)
        pairs
        """
        _members = self._sync_iter(self._async_group.members(max_depth=max_depth))

        return tuple((kv[0], _parse_async_node(kv[1])) for kv in _members)

    def keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_array('baz', shape=(10,), chunks=(10,))
        >>> d2 = g1.create_array('quux', shape=(10,), chunks=(10,))
        >>> for name in g1.keys():
        ...     print(name)
        baz
        bar
        foo
        quux
        """
        yield from self._sync_iter(self._async_group.keys())

    def __contains__(self, member: str) -> bool:
        """Test for group membership.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> d1 = g1.create_array('bar', shape=(10,), chunks=(10,))
        >>> 'foo' in g1
        True
        >>> 'bar' in g1
        True
        >>> 'baz' in g1
        False

        """
        return self._sync(self._async_group.contains(member))

    def groups(self) -> Generator[tuple[str, Group], None]:
        """Return the sub-groups of this group as a generator of (name, group) pairs.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for name, subgroup in group.groups():
        ...     print(name, subgroup)
        subgroup <Group memory://132270269438272/subgroup>
        """
        for name, async_group in self._sync_iter(self._async_group.groups()):
            yield name, Group(async_group)

    def group_keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for name in group.group_keys():
        ...     print(name)
        subgroup
        """
        for name, _ in self.groups():
            yield name

    def group_values(self) -> Generator[Group, None]:
        """Return an iterator over group members.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for subgroup in group.group_values():
        ...     print(subgroup)
        <Group memory://132270269438272/subgroup>
        """
        for _, group in self.groups():
            yield group

    def arrays(self) -> Generator[tuple[str, Array], None]:
        """Return the sub-arrays of this group as a generator of (name, array) pairs

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for name, subarray in group.arrays():
        ...     print(name, subarray)
        subarray <Array memory://140198565357056/subarray shape=(10,) dtype=float64>
        """
        for name, async_array in self._sync_iter(self._async_group.arrays()):
            yield name, Array(async_array)

    def array_keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for name in group.array_keys():
        ...     print(name)
        subarray
        """

        for name, _ in self.arrays():
            yield name

    def array_values(self) -> Generator[Array, None]:
        """Return an iterator over group members.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for subarray in group.array_values():
        ...     print(subarray)
        <Array memory://140198565357056/subarray shape=(10,) dtype=float64>
        """
        for _, array in self.arrays():
            yield array

    def tree(self, expand: bool | None = None, level: int | None = None) -> Any:
        """
        Return a tree-like representation of a hierarchy.

        This requires the optional ``rich`` dependency.

        Parameters
        ----------
        expand : bool, optional
            This keyword is not yet supported. A NotImplementedError is raised if
            it's used.
        level : int, optional
            The maximum depth below this Group to display in the tree.

        Returns
        -------
        TreeRepr
            A pretty-printable object displaying the hierarchy.
        """
        return self._sync(self._async_group.tree(expand=expand, level=level))

    def create_group(self, name: str, **kwargs: Any) -> Group:
        """Create a sub-group.

        Parameters
        ----------
        name : str
            Name of the new subgroup.

        Returns
        -------
        Group

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> subgroup = group.create_group("subgroup")
        >>> subgroup
        <Group memory://132270269438272/subgroup>
        """
        return Group(self._sync(self._async_group.create_group(name, **kwargs)))

    def require_group(self, name: str, **kwargs: Any) -> Group:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : str
            Group name.

        Returns
        -------
        g : Group
        """
        return Group(self._sync(self._async_group.require_group(name, **kwargs)))

    def require_groups(self, *names: str) -> tuple[Group, ...]:
        """Convenience method to require multiple groups in a single call.

        Parameters
        ----------
        *names : str
            Group names.

        Returns
        -------
        groups : tuple of Groups
        """
        return tuple(map(Group, self._sync(self._async_group.require_groups(*names))))

    def create(self, *args: Any, **kwargs: Any) -> Array:
        # Backwards compatibility for 2.x
        return self.create_array(*args, **kwargs)

    @_deprecate_positional_args
    def create_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords | Literal["auto"] = "auto",
        shards: ShardsLike | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = "auto",
        serializer: SerializerLike = "auto",
        fill_value: Any | None = 0,
        order: MemoryOrder | None = "C",
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
    ) -> Array:
        """Create an array within this group.

        This method lightly wraps :func:`zarr.core.array.create_array`.

        Parameters
        ----------
        name : str
            The name of the array relative to the group. If ``path`` is ``None``, the array will be located
            at the root of the store.
        shape : ChunkCoords
            Shape of the array.
        dtype : npt.DTypeLike
            Data type of the array.
        chunks : ChunkCoords, optional
            Chunk shape of the array.
            If not specified, default are guessed based on the shape and dtype.
        shards : ChunkCoords, optional
            Shard shape of the array. The default value of ``None`` results in no sharding at all.
        filters : Iterable[Codec], optional
            Iterable of filters to apply to each chunk of the array, in order, before serializing that
            chunk to bytes.

            For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
            and these values must be instances of ``ArrayArrayCodec``, or dict representations
            of ``ArrayArrayCodec``.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.

            For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
            the order if your filters is consistent with the behavior of each filter.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.
        compressors : Iterable[Codec], optional
            List of compressors to apply to the array. Compressors are applied in order, and after any
            filters are applied (if any are specified) and the data is serialized into bytes.

            For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
            returns another bytestream. Multiple compressors my be provided for Zarr format 3.
            If no ``compressors`` are provided, a default set of compressors will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_compressors``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default compressors.

            For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
            be provided for Zarr format 2.
            If no ``compressor`` is provided, a default compressor will be used.
            in :mod:`zarr.core.config`.
            Use ``None`` to omit the default compressor.
        compressor : Codec, optional
            Deprecated in favor of ``compressors``.
        serializer : dict[str, JSON] | ArrayBytesCodec, optional
            Array-to-bytes codec to use for encoding the array data.
            Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
            If no ``serializer`` is provided, a default serializer will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_serializer``
            in :mod:`zarr.core.config`.
        fill_value : Any, optional
            Fill value for the array.
        order : {"C", "F"}, optional
            The memory of the array (default is "C").
            For Zarr format 2, this parameter sets the memory order of the array.
            For Zarr format 3, this parameter is deprecated, because memory order
            is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
            order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
            If no ``order`` is provided, a default order will be used.
            This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
        attributes : dict, optional
            Attributes for the array.
        chunk_key_encoding : ChunkKeyEncoding, optional
            A specification of how the chunk keys are represented in storage.
            For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
            For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        storage_options : dict, optional
            If using an fsspec URL to create the store, these will be passed to the backend implementation.
            Ignored otherwise.
        overwrite : bool, default False
            Whether to overwrite an array with the same name in the store, if one exists.
        config : ArrayConfig or ArrayConfigLike, optional
            Runtime configuration for the array.

        Returns
        -------
        AsyncArray
        """
        compressors = _parse_deprecated_compressor(
            compressor, compressors, zarr_format=self.metadata.zarr_format
        )
        return Array(
            self._sync(
                self._async_group.create_array(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    shards=shards,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_key_encoding=chunk_key_encoding,
                    compressors=compressors,
                    serializer=serializer,
                    dimension_names=dimension_names,
                    order=order,
                    filters=filters,
                    overwrite=overwrite,
                    storage_options=storage_options,
                    config=config,
                )
            )
        )

    @deprecated("Use Group.create_array instead.")
    def create_dataset(self, name: str, **kwargs: Any) -> Array:
        """Create an array.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.create_array` instead.


        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.require_dataset` method.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs : dict
            Additional arguments passed to :func:`zarr.Group.create_array`

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.create_dataset(name, **kwargs)))

    @deprecated("Use Group.require_array instead.")
    def require_dataset(self, name: str, *, shape: ShapeLike, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.require_array` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.Group.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs :
            See :func:`zarr.Group.create_dataset`.

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.require_array(name, shape=shape, **kwargs)))

    def require_array(self, name: str, *, shape: ShapeLike, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.

        Other `kwargs` are as per :func:`zarr.Group.create_array`.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs :
            See :func:`zarr.Group.create_array`.

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.require_array(name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def empty(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        """Create an empty array with the specified shape in this Group. The contents will be filled with
        the array's fill value or zeros if no fill value is provided.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Notes
        -----
        The contents of an empty Zarr array are not defined. On attempting to
        retrieve data from an empty Zarr array, any values may be returned,
        and these are not guaranteed to be stable from one access to the next.
        """
        return Array(self._sync(self._async_group.empty(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def zeros(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        """Create an array, with zero being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.zeros(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def ones(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        """Create an array, with one being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.ones(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> Array:
        """Create an array, with "fill_value" being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        fill_value : scalar
            Value to fill the array with.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """
        return Array(
            self._sync(
                self._async_group.full(name=name, shape=shape, fill_value=fill_value, **kwargs)
            )
        )

    @_deprecate_positional_args
    def empty_like(self, *, name: str, data: async_api.ArrayLike, **kwargs: Any) -> Array:
        """Create an empty sub-array like `data`. The contents will be filled
        with the array's fill value or zeros if no fill value is provided.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create an empty array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.

        Notes
        -----
        The contents of an empty Zarr array are not defined. On attempting to
        retrieve data from an empty Zarr array, any values may be returned,
        and these are not guaranteed to be stable from one access to the next.
        """
        return Array(self._sync(self._async_group.empty_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def zeros_like(self, *, name: str, data: async_api.ArrayLike, **kwargs: Any) -> Array:
        """Create a sub-array of zeros like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """

        return Array(self._sync(self._async_group.zeros_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def ones_like(self, *, name: str, data: async_api.ArrayLike, **kwargs: Any) -> Array:
        """Create a sub-array of ones like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.ones_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def full_like(self, *, name: str, data: async_api.ArrayLike, **kwargs: Any) -> Array:
        """Create a sub-array like `data` filled with the `fill_value` of `data` .

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.full_like(name=name, data=data, **kwargs)))

    def move(self, source: str, dest: str) -> None:
        """Move a sub-group or sub-array from one path to another.

        Notes
        -----
        Not implemented
        """
        return self._sync(self._async_group.move(source, dest))

    @deprecated("Use Group.create_array instead.")
    @_deprecate_positional_args
    def array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords | Literal["auto"] = "auto",
        shards: ChunkCoords | Literal["auto"] | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = None,
        serializer: SerializerLike = "auto",
        fill_value: Any | None = 0,
        order: MemoryOrder | None = "C",
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
        data: npt.ArrayLike | None = None,
    ) -> Array:
        """Create an array within this group.

        .. deprecated:: 3.0.0
            Use `Group.create_array` instead.

        This method lightly wraps :func:`zarr.core.array.create_array`.

        Parameters
        ----------
        name : str
            The name of the array relative to the group. If ``path`` is ``None``, the array will be located
            at the root of the store.
        shape : ChunkCoords
            Shape of the array.
        dtype : npt.DTypeLike
            Data type of the array.
        chunks : ChunkCoords, optional
            Chunk shape of the array.
            If not specified, default are guessed based on the shape and dtype.
        shards : ChunkCoords, optional
            Shard shape of the array. The default value of ``None`` results in no sharding at all.
        filters : Iterable[Codec], optional
            Iterable of filters to apply to each chunk of the array, in order, before serializing that
            chunk to bytes.

            For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
            and these values must be instances of ``ArrayArrayCodec``, or dict representations
            of ``ArrayArrayCodec``.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.

            For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
            the order if your filters is consistent with the behavior of each filter.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.
        compressors : Iterable[Codec], optional
            List of compressors to apply to the array. Compressors are applied in order, and after any
            filters are applied (if any are specified) and the data is serialized into bytes.

            For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
            returns another bytestream. Multiple compressors my be provided for Zarr format 3.
            If no ``compressors`` are provided, a default set of compressors will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_compressors``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default compressors.

            For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
            be provided for Zarr format 2.
            If no ``compressor`` is provided, a default compressor will be used.
            in :mod:`zarr.core.config`.
            Use ``None`` to omit the default compressor.
        compressor : Codec, optional
            Deprecated in favor of ``compressors``.
        serializer : dict[str, JSON] | ArrayBytesCodec, optional
            Array-to-bytes codec to use for encoding the array data.
            Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
            If no ``serializer`` is provided, a default serializer will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_serializer``
            in :mod:`zarr.core.config`.
        fill_value : Any, optional
            Fill value for the array.
        order : {"C", "F"}, optional
            The memory of the array (default is "C").
            For Zarr format 2, this parameter sets the memory order of the array.
            For Zarr format 3, this parameter is deprecated, because memory order
            is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
            order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
            If no ``order`` is provided, a default order will be used.
            This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
        attributes : dict, optional
            Attributes for the array.
        chunk_key_encoding : ChunkKeyEncoding, optional
            A specification of how the chunk keys are represented in storage.
            For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
            For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        storage_options : dict, optional
            If using an fsspec URL to create the store, these will be passed to the backend implementation.
            Ignored otherwise.
        overwrite : bool, default False
            Whether to overwrite an array with the same name in the store, if one exists.
        config : ArrayConfig or ArrayConfigLike, optional
            Runtime configuration for the array.
        data : array_like
            The data to fill the array with.

        Returns
        -------
        AsyncArray
        """
        compressors = _parse_deprecated_compressor(compressor, compressors)
        return Array(
            self._sync(
                self._async_group.create_dataset(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    shards=shards,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_key_encoding=chunk_key_encoding,
                    compressors=compressors,
                    serializer=serializer,
                    dimension_names=dimension_names,
                    order=order,
                    filters=filters,
                    overwrite=overwrite,
                    storage_options=storage_options,
                    config=config,
                    data=data,
                )
            )
        )


async def _getitem_semaphore(
    node: AsyncGroup, key: str, semaphore: asyncio.Semaphore | None
) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup:
    """
    Combine node.getitem with an optional semaphore. If the semaphore parameter is an
    asyncio.Semaphore instance, then the getitem operation is performed inside an async context
    manager provided by that semaphore. If the semaphore parameter is None, then getitem is invoked
    without a context manager.
    """
    if semaphore is not None:
        async with semaphore:
            return await node.getitem(key)
    else:
        return await node.getitem(key)


async def _iter_members(
    node: AsyncGroup,
    skip_keys: tuple[str, ...],
    semaphore: asyncio.Semaphore | None,
) -> AsyncGenerator[
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
]:
    """
    Iterate over the arrays and groups contained in a group.

    Parameters
    ----------
    node : AsyncGroup
        The group to traverse.
    skip_keys : tuple[str, ...]
        A tuple of keys to skip when iterating over the possible members of the group.
    semaphore : asyncio.Semaphore | None
        An optional semaphore to use for concurrency control.

    Yields
    ------
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup]
    """

    # retrieve keys from storage
    keys = [key async for key in node.store.list_dir(node.path)]
    keys_filtered = tuple(filter(lambda v: v not in skip_keys, keys))

    node_tasks = tuple(
        asyncio.create_task(_getitem_semaphore(node, key, semaphore), name=key)
        for key in keys_filtered
    )

    for fetched_node_coro in asyncio.as_completed(node_tasks):
        try:
            fetched_node = await fetched_node_coro
        except KeyError as e:
            # keyerror is raised when `key` names an object (in the object storage sense),
            # as opposed to a prefix, in the store under the prefix associated with this group
            # in which case `key` cannot be the name of a sub-array or sub-group.
            warnings.warn(
                f"Object at {e.args[0]} is not recognized as a component of a Zarr hierarchy.",
                UserWarning,
                stacklevel=1,
            )
            continue
        match fetched_node:
            case AsyncArray() | AsyncGroup():
                yield fetched_node.basename, fetched_node
            case _:
                raise ValueError(f"Unexpected type: {type(fetched_node)}")


async def _iter_members_deep(
    group: AsyncGroup,
    *,
    max_depth: int | None,
    skip_keys: tuple[str, ...],
    semaphore: asyncio.Semaphore | None = None,
) -> AsyncGenerator[
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
]:
    """
    Iterate over the arrays and groups contained in a group, and optionally the
    arrays and groups contained in those groups.

    Parameters
    ----------
    group : AsyncGroup
        The group to traverse.
    max_depth : int | None
        The maximum depth of recursion.
    skip_keys : tuple[str, ...]
        A tuple of keys to skip when iterating over the possible members of the group.
    semaphore : asyncio.Semaphore | None
        An optional semaphore to use for concurrency control.

    Yields
    ------
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup]
    """

    to_recurse = {}
    do_recursion = max_depth is None or max_depth > 0

    if max_depth is None:
        new_depth = None
    else:
        new_depth = max_depth - 1
    async for name, node in _iter_members(group, skip_keys=skip_keys, semaphore=semaphore):
        yield name, node
        if isinstance(node, AsyncGroup) and do_recursion:
            to_recurse[name] = _iter_members_deep(
                node, max_depth=new_depth, skip_keys=skip_keys, semaphore=semaphore
            )

    for prefix, subgroup_iter in to_recurse.items():
        async for name, node in subgroup_iter:
            key = f"{prefix}/{name}".lstrip("/")
            yield key, node


def _resolve_metadata_v2(
    blobs: tuple[str | bytes | bytearray, str | bytes | bytearray],
) -> ArrayV2Metadata | GroupMetadata:
    zarr_metadata = json.loads(blobs[0])
    attrs = json.loads(blobs[1])
    if "shape" in zarr_metadata:
        return ArrayV2Metadata.from_dict(zarr_metadata | {"attrs": attrs})
    else:
        return GroupMetadata.from_dict(zarr_metadata | {"attrs": attrs})


def _build_metadata_v3(zarr_json: dict[str, Any]) -> ArrayV3Metadata | GroupMetadata:
    """
    Take a dict and convert it into the correct metadata type.
    """
    if "node_type" not in zarr_json:
        raise KeyError("missing `node_type` key in metadata document.")
    match zarr_json:
        case {"node_type": "array"}:
            return ArrayV3Metadata.from_dict(zarr_json)
        case {"node_type": "group"}:
            return GroupMetadata.from_dict(zarr_json)
        case _:
            raise ValueError("invalid value for `node_type` key in metadata document")


def _build_metadata_v2(
    zarr_json: dict[str, Any], attrs_json: dict[str, Any]
) -> ArrayV2Metadata | GroupMetadata:
    """
    Take a dict and convert it into the correct metadata type.
    """
    match zarr_json:
        case {"shape": _}:
            return ArrayV2Metadata.from_dict(zarr_json | {"attributes": attrs_json})
        case _:
            return GroupMetadata.from_dict(zarr_json | {"attributes": attrs_json})


def _build_node_v3(
    metadata: ArrayV3Metadata | GroupMetadata, store_path: StorePath
) -> AsyncArray[ArrayV3Metadata] | AsyncGroup:
    """
    Take a metadata object and return a node (AsyncArray or AsyncGroup).
    """
    match metadata:
        case ArrayV3Metadata():
            return AsyncArray(metadata, store_path=store_path)
        case GroupMetadata():
            return AsyncGroup(metadata, store_path=store_path)
        case _:
            raise ValueError(f"Unexpected metadata type: {type(metadata)}")


def _build_node_v2(
    metadata: ArrayV2Metadata | GroupMetadata, store_path: StorePath
) -> AsyncArray[ArrayV2Metadata] | AsyncGroup:
    """
    Take a metadata object and return a node (AsyncArray or AsyncGroup).
    """

    match metadata:
        case ArrayV2Metadata():
            return AsyncArray(metadata, store_path=store_path)
        case GroupMetadata():
            return AsyncGroup(metadata, store_path=store_path)
        case _:
            raise ValueError(f"Unexpected metadata type: {type(metadata)}")
