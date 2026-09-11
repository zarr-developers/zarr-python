"""Helpers for moving JSON documents in and out of zarr stores.

These are free functions, deliberately not methods on the ``Store`` ABC:
reading and writing JSON is a composition of the store's ``get``/``set``
primitives with a buffer/JSON conversion, not part of the store contract.
Keeping them as functions means stores cannot (and need not) override them,
and the ``Store`` definition stays free of any dependency on the buffer
prototype.

These functions are pure: the JSON encoding parameters (``indent``,
``allow_nan``) are explicit arguments rather than read from the global config.
Callers that want zarr's configured indentation pass
``indent=config.get("json_indent")``.

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

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import JSON


def buffer_to_json(buffer: Buffer) -> JSON:
    """Parse the contents of a `Buffer` as a JSON value."""
    # json.loads is typed as returning Any; the result is by definition JSON.
    return cast("JSON", json.loads(buffer.to_bytes()))


def buffer_to_json_object(buffer: Buffer) -> dict[str, JSON]:
    """Parse the contents of a `Buffer` as a JSON object (a `dict`).

    Every metadata document zarr reads is a JSON object, so this narrows the
    `JSON` union to `dict[str, JSON]` once, here, instead of at each call site.

    Parameters
    ----------
    buffer
        The buffer whose contents are parsed as a JSON object.

    Raises
    ------
    TypeError
        If the parsed value is not a JSON object.
    """
    obj = buffer_to_json(buffer)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object, got {type(obj).__name__}.")
    return obj


def json_to_buffer(
    obj: JSON,
    *,
    prototype: BufferPrototype | None = None,
    indent: int | None = None,
    allow_nan: bool = True,
) -> Buffer:
    """Serialize a JSON value into a `Buffer`.

    Parameters
    ----------
    obj
        The JSON-serializable value to encode.
    prototype
        The buffer prototype to construct the result with. Defaults to
        `default_buffer_prototype()`.
    indent
        Indentation passed to `json.dumps`. `None` (the default) writes
        without newline indentation, using json's default separators.
        Callers that want zarr's configured indentation pass
        `indent=config.get("json_indent")`.
    allow_nan
        Whether to permit `NaN`/`Infinity` in the output, passed to
        `json.dumps`.
    """
    if prototype is None:
        prototype = default_buffer_prototype()
    return prototype.buffer.from_bytes(json.dumps(obj, indent=indent, allow_nan=allow_nan).encode())


async def get_json(store: Store, key: str, *, byte_range: ByteRequest | None = None) -> JSON | None:
    """Read and parse the JSON document at `key`, or `None` if it is absent.

    Parameters
    ----------
    store
        The store to read from.
    key
        The key identifying the JSON document.
    byte_range
        If given, read only this portion of the value. Note that a partial
        read of a JSON document may not be valid JSON.

    Returns
    -------
    JSON or None
        The parsed JSON value, or `None` if `key` does not exist.
    """
    buffer = await store.get(key, default_buffer_prototype(), byte_range)
    return None if buffer is None else buffer_to_json(buffer)


async def set_json(
    store: Store,
    key: str,
    obj: JSON,
    *,
    prototype: BufferPrototype | None = None,
    indent: int | None = None,
    allow_nan: bool = True,
) -> None:
    """Serialize `obj` as JSON and write it to `key` in `store`.

    `indent` and `allow_nan` are forwarded to `json_to_buffer`.
    """
    await store.set(
        key, json_to_buffer(obj, prototype=prototype, indent=indent, allow_nan=allow_nan)
    )
