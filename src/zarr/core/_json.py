"""Helpers for moving JSON documents in and out of zarr stores.

These are free functions, deliberately not methods on the ``Store`` ABC:
reading and writing JSON is a composition of the store's ``get``/``set``
primitives with a buffer/JSON conversion, not part of the store contract.
Keeping them as functions means stores cannot (and need not) override them,
and the ``Store`` definition stays free of any dependency on the buffer
prototype / global config.

Two layers:

- ``buffer_to_json`` / ``json_to_buffer`` convert between a ``Buffer`` and a
  parsed JSON value. The buffer prototype lives here, at buffer construction,
  where it is meaningful.
- ``get_json`` / ``set_json`` compose those with ``Store.get`` / ``Store.set``.
  ``get_json`` returns ``None`` for a missing key (the contract most callers
  want); callers that require presence check for ``None`` themselves.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import config

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import JSON


def buffer_to_json(buffer: Buffer) -> JSON:
    """Parse the contents of a `Buffer` as a JSON value."""
    # json.loads is typed as returning Any; the result is by definition JSON.
    return cast("JSON", json.loads(buffer.to_bytes()))


def json_to_buffer(obj: JSON, *, prototype: BufferPrototype | None = None) -> Buffer:
    """Serialize a JSON value into a `Buffer`.

    The serialization policy (`indent` from `config["json_indent"]` and
    `allow_nan=True`) is defined here, once, for every JSON document zarr
    writes.

    Parameters
    ----------
    obj : JSON
        The JSON-serializable value to encode.
    prototype : BufferPrototype, optional
        The buffer prototype to construct the result with. Defaults to
        `default_buffer_prototype()`.
    """
    if prototype is None:
        prototype = default_buffer_prototype()
    json_indent = config.get("json_indent")
    return prototype.buffer.from_bytes(json.dumps(obj, indent=json_indent, allow_nan=True).encode())


async def get_json(store: Store, key: str, *, byte_range: ByteRequest | None = None) -> JSON | None:
    """Read and parse the JSON document at `key`, or `None` if it is absent.

    Parameters
    ----------
    store : Store
        The store to read from.
    key : str
        The key identifying the JSON document.
    byte_range : ByteRequest, optional
        If given, read only this portion of the value. Note that a partial
        read of a JSON document may not be valid JSON.

    Returns
    -------
    JSON or None
        The parsed JSON value, or `None` if `key` does not exist.
    """
    buffer = await store.get(key, default_buffer_prototype(), byte_range)
    if buffer is None:
        return None
    return buffer_to_json(buffer)


async def set_json(
    store: Store, key: str, obj: JSON, *, prototype: BufferPrototype | None = None
) -> None:
    """Serialize `obj` as JSON and write it to `key` in `store`."""
    await store.set(key, json_to_buffer(obj, prototype=prototype))
