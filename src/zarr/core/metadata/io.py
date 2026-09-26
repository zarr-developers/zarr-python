from __future__ import annotations

import asyncio
from dataclasses import replace
from enum import Enum
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from zarr.abc.store import set_or_delete
from zarr.core._json import buffer_to_json_object, json_equal
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.buffer.cpu import buffer_prototype as cpu_buffer_prototype
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata.upgrades import mark_upgraded, upgrade_array_document
from zarr.errors import ArrayNotFoundError, ContainsArrayError
from zarr.storage._common import StorePath, ensure_no_existing_node

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.group import GroupMetadata
    from zarr.core.metadata import ArrayMetadata


class _Absent(Enum):
    ABSENT = "absent"


ABSENT: Final = _Absent.ABSENT
"""Where a `DocumentChange` has no value: the document, member or element is not there."""


class DocumentChange(NamedTuple):
    """A JSON value that differs between a node's stored metadata documents and the
    documents its metadata would store."""

    path: tuple[str | int, ...]
    """Where the value is: the document's key (e.g. `.zarray`), then the object members
    and array indices within it."""
    stored: JSON | _Absent
    new: JSON | _Absent


def diff_documents(
    stored: Mapping[str, JSON], new: Mapping[str, JSON]
) -> tuple[DocumentChange, ...]:
    """What differs between a node's stored metadata documents and the documents its
    metadata would store, each keyed by its store key (a document missing from `stored`
    is not stored). Empty if they are identical.

    Objects and arrays are compared member by member, so a change is the smallest value
    that differs; any other two values are identical only if their JSON encodings are, so
    `true` differs from `1` and `1.0` from `1`.
    """
    return tuple(_diff((), stored, new))


def _diff(
    path: tuple[str | int, ...], stored: JSON | _Absent, new: JSON | _Absent
) -> Iterator[DocumentChange]:
    match stored, new:
        case dict(), dict():
            for key in dict.fromkeys([*stored, *new]):
                yield from _diff((*path, key), stored.get(key, ABSENT), new.get(key, ABSENT))
        case list(), list():
            for index, pair in enumerate(zip_longest(stored, new, fillvalue=ABSENT)):
                yield from _diff((*path, index), *pair)
        case _ if stored is ABSENT or new is ABSENT or not json_equal(stored, new):
            yield DocumentChange(path, stored, new)


def encode_documents(
    store_path: StorePath, metadata: ArrayMetadata | GroupMetadata
) -> dict[str, Buffer]:
    """The metadata documents `metadata` stores under `store_path`, by key (see
    `to_buffer_dict`).

    An operation that deletes or writes store content encodes its documents first, so
    metadata that cannot be stored fails with the store untouched; the error then names
    the node at `store_path`, as the warnings about stored documents do.
    """
    from zarr.core.group import GroupMetadata

    try:
        return metadata.to_buffer_dict(default_buffer_prototype())
    except ValueError as e:
        node = "Group" if isinstance(metadata, GroupMetadata) else "Array"
        e.add_note(f"{node} {str(store_path)!r}: nothing was stored.")
        raise


async def store_documents(store_path: StorePath, documents: Mapping[str, Buffer]) -> None:
    """Store metadata documents encoded by `encode_documents` under `store_path`."""
    await asyncio.gather(
        *(set_or_delete(store_path / key, value) for key, value in documents.items())
    )


async def read_documents(store_path: StorePath, keys: Iterable[str]) -> dict[str, Buffer]:
    """The documents stored under `store_path` at `keys`, read concurrently, by key; a
    key with no document is left out."""
    keys = tuple(keys)
    buffers = await asyncio.gather(
        *((store_path / key).get(prototype=cpu_buffer_prototype) for key in keys)
    )
    return {key: buf for key, buf in zip(keys, buffers, strict=True) if buf is not None}


async def upsert_metadata(
    store_path: StorePath, metadata: ArrayMetadata | GroupMetadata, stored: Mapping[str, Buffer]
) -> tuple[DocumentChange, ...]:
    """Store the documents of `metadata` under `store_path` that differ from `stored`,
    the documents `read_documents` read there, and return how they differed (see
    `diff_documents`): empty if nothing was stored.

    The documents are encoded before any is stored, so metadata that cannot be stored
    fails with the store untouched.
    """
    documents = encode_documents(store_path, metadata)
    changes = diff_documents(
        {key: buffer_to_json_object(buf) for key, buf in stored.items() if key in documents},
        {key: buffer_to_json_object(buf) for key, buf in documents.items()},
    )
    changed = {change.path[0] for change in changes}
    await store_documents(store_path, {k: v for k, v in documents.items() if k in changed})
    return changes


ARRAY_DOCUMENTS: Final[Mapping[ZarrFormat, tuple[str, ...]]] = {
    2: (ZARRAY_JSON, ZATTRS_JSON),
    3: (ZARR_JSON,),
}
"""The store keys of the metadata documents of an array of each Zarr format."""


def parse_stored_array(
    documents: Mapping[str, Buffer], zarr_format: ZarrFormat, path: str | None = None
) -> ArrayMetadata:
    """The metadata of an array from its documents (by store key, see `ARRAY_DOCUMENTS`),
    read with the upgrades but without their warnings (whoever asks has warned, or reads
    metadata built in code), and marked (see `mark_upgraded`) if they had to be
    upgraded. Raises `ArrayNotFoundError` if there is no array document among them.

    Only operations that store metadata read documents this way, so documents read as a
    rectilinear chunk grid require the rectilinear chunks flag, as storing it does; the
    error names the array at `path`."""
    from zarr.core.array import (
        _array_metadata_dict_v2,
        _array_metadata_dict_v3,
        parse_array_metadata,
    )

    if zarr_format == 2 and ZARRAY_JSON in documents:
        stored = _array_metadata_dict_v2(documents[ZARRAY_JSON], documents.get(ZATTRS_JSON))
    elif zarr_format == 3 and ZARR_JSON in documents:
        stored = _array_metadata_dict_v3(documents[ZARR_JSON])
    else:
        raise ArrayNotFoundError(f"No Zarr format {zarr_format} array metadata document.")
    upgraded, readings = upgrade_array_document(stored, zarr_format)
    return mark_upgraded(parse_array_metadata(dict(upgraded), path), [None] * len(readings), None)


class MemberUpgrade(NamedTuple):
    """The upgrade of a consolidated member's own stored documents, to store along with
    the group (see `_refresh_consolidated`)."""

    store_path: StorePath
    metadata: ArrayMetadata
    """The member's metadata, read from `stored` (see `parse_stored_array`)."""
    stored: dict[str, Buffer]
    """The member's documents as `read_documents` read them."""


async def _refresh_array(
    store_path: StorePath, member: ArrayMetadata
) -> tuple[ArrayMetadata, list[MemberUpgrade]]:
    """The metadata of the array document stored at `store_path` as it now is, for
    `member`, a member of consolidated metadata that was read from a document that had
    to be upgraded, and the upgrade of that document if it still needs one; `member`
    itself if no array document that can be read is stored there (it was deleted, or
    replaced by a group)."""
    documents = await read_documents(store_path, ARRAY_DOCUMENTS[member.zarr_format])
    try:
        current = parse_stored_array(documents, member.zarr_format, str(store_path))
    except (KeyError, TypeError, ValueError):
        return member, []
    if current._stored_document_upgraded:
        return current, [MemberUpgrade(store_path, current, documents)]
    return current, []


async def _refresh_consolidated(
    store_path: StorePath, metadata: GroupMetadata
) -> tuple[GroupMetadata, list[MemberUpgrade]]:
    """`metadata` with each member of its consolidated metadata that was read from a
    document that had to be upgraded replaced by the metadata of the member's own
    document as it now is (see `_refresh_array`), read concurrently: the member document
    may have changed since, so the consolidated copy is never stored as if it were
    valid. Also returns the members whose own documents still need their upgrade
    stored. Reads the store; writes nothing."""
    from zarr.core.group import GroupMetadata

    consolidated = metadata.consolidated_metadata
    if consolidated is None:
        return metadata, []
    stale = {
        name: member
        for name, member in consolidated.metadata.items()
        if isinstance(member, GroupMetadata) or member._stored_document_upgraded
    }
    refreshed = await asyncio.gather(
        *(
            _refresh_consolidated(store_path / name, member)
            if isinstance(member, GroupMetadata)
            else _refresh_array(store_path / name, member)
            for name, member in stale.items()
        )
    )
    upgrades = [upgrade for _, member_upgrades in refreshed for upgrade in member_upgrades]
    if all(new is old for (new, _), old in zip(refreshed, stale.values(), strict=True)):
        return metadata, upgrades
    members = {
        **consolidated.metadata,
        **{name: new for name, (new, _) in zip(stale, refreshed, strict=True)},
    }
    refreshed_metadata = replace(
        metadata, consolidated_metadata=replace(consolidated, metadata=members)
    )
    return refreshed_metadata, upgrades


class EncodedNode[M: (ArrayMetadata, GroupMetadata)](NamedTuple):
    """What storing a node's metadata writes, encoded before anything is written (see
    `encode_node`)."""

    metadata: M
    """The metadata stored: for a group, with its consolidated members refreshed (see
    `_refresh_consolidated`), for the group to adopt."""
    documents: dict[str, Buffer]
    """The node's own documents, by key (see `encode_documents`)."""
    members: list[MemberUpgrade]
    """Members of a group's consolidated metadata whose own stored documents are
    upgraded along with it."""


async def encode_node[M: (ArrayMetadata, GroupMetadata)](
    store_path: StorePath, metadata: M
) -> EncodedNode[M]:
    """Encode what storing `metadata` under `store_path` writes, so metadata that cannot
    be stored fails with the store untouched. Group metadata is stored with its
    consolidated metadata, whose upgraded members are first refreshed from their own
    stored documents (see `_refresh_consolidated`); encoding the group then encodes
    them too."""
    from zarr.core.group import GroupMetadata

    members: list[MemberUpgrade] = []
    if isinstance(metadata, GroupMetadata):
        metadata, members = await _refresh_consolidated(store_path, metadata)
    return EncodedNode(metadata, encode_documents(store_path, metadata), members)


async def store_node(store_path: StorePath, encoded: EncodedNode[Any]) -> None:
    """Store what `encode_node` encoded under `store_path`. The members whose upgrades
    it stores no longer need them."""
    await asyncio.gather(
        store_documents(store_path, encoded.documents),
        *(upsert_metadata(*member) for member in encoded.members),
    )
    for member in encoded.members:
        object.__setattr__(member.metadata, "_stored_document_upgraded", False)


def _build_parents(store_path: StorePath, zarr_format: ZarrFormat) -> dict[str, GroupMetadata]:
    from zarr.core.group import GroupMetadata

    path = store_path.path
    if not path:
        return {}

    required_parts = path.split("/")[:-1]

    # the root group
    parents = {"": GroupMetadata(zarr_format=zarr_format)}

    for i, part in enumerate(required_parts):
        parent_path = "/".join(required_parts[:i] + [part])
        parents[parent_path] = GroupMetadata(zarr_format=zarr_format)

    return parents


async def save_metadata[M: (ArrayMetadata, GroupMetadata)](
    store_path: StorePath, metadata: M, ensure_parents: bool = False
) -> M:
    """Asynchronously save the array or group metadata, and return the metadata saved
    (see `EncodedNode.metadata`).

    Parameters
    ----------
    store_path : StorePath
        Location to save metadata.
    metadata : ArrayMetadata | GroupMetadata
        Metadata to save.
    ensure_parents : bool, optional
        Create any missing parent groups, and check no existing parents are arrays.

    Raises
    ------
    ValueError
    """
    encoded = await encode_node(store_path, metadata)
    set_awaitables = [store_node(store_path, encoded)]

    if ensure_parents:
        # To enable zarr.create(store, path="a/b/c"), we need to create all the intermediate groups.
        parents = _build_parents(store_path, metadata.zarr_format)
        ensure_array_awaitables = []

        for parent_path, parent_metadata in parents.items():
            parent_store_path = StorePath(store_path.store, parent_path)

            # Error if an array already exists at any parent location. Only groups can have child nodes.
            ensure_array_awaitables.append(
                ensure_no_existing_node(
                    parent_store_path, parent_metadata.zarr_format, node_type="array"
                )
            )
            set_awaitables.extend(
                [
                    (parent_store_path / key).set_if_not_exists(value)
                    for key, value in parent_metadata.to_buffer_dict(
                        default_buffer_prototype()
                    ).items()
                ]
            )

        # Checks for parent arrays must happen first, before any metadata is modified
        try:
            await asyncio.gather(*ensure_array_awaitables)
        except ContainsArrayError as e:
            # clear awaitables to avoid RuntimeWarning: coroutine was never awaited
            for awaitable in set_awaitables:
                awaitable.close()

            raise ValueError(
                f"A parent of {store_path} is an array - only groups may have child nodes."
            ) from e

    await asyncio.gather(*set_awaitables)
    return encoded.metadata
