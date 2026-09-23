"""
The extensions that Zarr metadata is read with, and the state of a read in progress.

A `Context` is the set of extensions a read may resolve, by kind. It is a value: the same for a
whole read, and passed in by the caller as `context=`. A `Reading` is where a read has got to: the
context, the Zarr format of the document, and the location in the document. It changes at every
nested step, and is threaded through the read by zarr.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeJSON
    from zarr.core.dtype.registry import DataTypeRegistry
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


@dataclass(frozen=True, kw_only=True)
class Context:
    """
    The extensions that Zarr metadata is read with, by kind.

    Attributes
    ----------
    data_types : DataTypeRegistry
        The data types a read may resolve.
    """

    data_types: DataTypeRegistry

    @classmethod
    def default(cls) -> Context:
        """
        The context of the default registries, such as `zarr.dtype.data_type_registry`.

        It refers to the registries themselves, so it includes extensions registered after it is
        created.
        """
        # avoid circular import
        from zarr.core.dtype import data_type_registry

        return cls(data_types=data_type_registry)


type JSONLocation = tuple[str | int, ...]


def json_pointer(loc: JSONLocation) -> str:
    """
    The JSON Pointer (RFC 6901) of a location in a JSON document, e.g. `/codecs/0/configuration`.
    """
    return "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in loc)


@dataclass(frozen=True, kw_only=True)
class Reading:
    """
    The state of a read of Zarr metadata in progress.

    An extension that contains other extensions, such as a structured data type, reads each of them
    with the reading returned by `at`, so that they come from the same context, use the same Zarr
    format, and report errors with their location.

    Attributes
    ----------
    context : Context
        The extensions the read may resolve.
    zarr_format : ZarrFormat
        The Zarr format of the document.
    loc : tuple[str | int, ...]
        The location in the document of the JSON being read, as keys and indices. A Zarr V2 data
        type is located by its name as written in the document (the value of `"dtype"`), not by
        the `{"name": ..., "object_codec_id": ...}` form zarr reads it in.
    """

    context: Context
    zarr_format: ZarrFormat
    loc: JSONLocation = ()

    def at(self, *keys: str | int) -> Reading:
        """
        The reading of the JSON at `keys` within the JSON of this reading.
        """
        return replace(self, loc=(*self.loc, *keys))

    def resolve_data_type(self, data: DTypeJSON) -> ZDType[TBaseDType, TBaseScalar]:
        """
        Resolve the JSON representation of a data type at the location of this reading.

        Raises
        ------
        ValueError
            If no data type in the context matches `data`.
        """
        return self.context.data_types._match_json(data, reading=self)
