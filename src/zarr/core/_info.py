import dataclasses
import textwrap
from typing import Any, Literal

import numcodecs.abc
import numpy as np

from zarr.abc.codec import Codec
from zarr.core.metadata.v3 import DataType


@dataclasses.dataclass(kw_only=True)
class GroupInfo:
    """
    Visual summary for a Group.

    Note that this method and its properties is not part of
    Zarr's public API.
    """

    _name: str
    _type: Literal["Group"] = "Group"
    _zarr_format: Literal[2, 3]
    _read_only: bool
    _store_type: str
    _count_members: int | None = None
    _count_arrays: int | None = None
    _count_groups: int | None = None

    def __repr__(self) -> str:
        template = textwrap.dedent("""\
        Name        : {_name}
        Type        : {_type}
        Zarr format : {_zarr_format}
        Read-only   : {_read_only}
        Store type  : {_store_type}""")

        if self._count_members is not None:
            template += "\nNo. members : {_count_members}"
        if self._count_arrays is not None:
            template += "\nNo. arrays  : {_count_arrays}"
        if self._count_groups is not None:
            template += "\nNo. groups  : {_count_groups}"
        return template.format(**dataclasses.asdict(self))


def human_readable_size(size: int) -> str:
    if size < 2**10:
        return f"{size}"
    elif size < 2**20:
        return f"{size / float(2**10):.1f}K"
    elif size < 2**30:
        return f"{size / float(2**20):.1f}M"
    elif size < 2**40:
        return f"{size / float(2**30):.1f}G"
    elif size < 2**50:
        return f"{size / float(2**40):.1f}T"
    else:
        return f"{size / float(2**50):.1f}P"


def byte_info(size: int) -> str:
    if size < 2**10:
        return str(size)
    else:
        return f"{size} ({human_readable_size(size)})"


@dataclasses.dataclass(kw_only=True)
class ArrayInfo:
    """
    Visual summary for an Array.

    Note that this method and its properties is not part of
    Zarr's public API.
    """

    _type: Literal["Array"] = "Array"
    _zarr_format: Literal[2, 3]
    _data_type: np.dtype[Any] | DataType
    _shape: tuple[int, ...]
    _chunk_shape: tuple[int, ...] | None = None
    _order: Literal["C", "F"]
    _read_only: bool
    _store_type: str
    _compressor: numcodecs.abc.Codec | None = None
    _filters: tuple[numcodecs.abc.Codec, ...] | None = None
    _codecs: list[Codec] | None = None
    _count_bytes: int | None = None
    _count_bytes_stored: int | None = None
    _count_chunks_initialized: int | None = None

    def __repr__(self) -> str:
        template = textwrap.dedent("""\
        Type               : {_type}
        Zarr format        : {_zarr_format}
        Data type          : {_data_type}
        Shape              : {_shape}
        Chunk shape        : {_chunk_shape}
        Order              : {_order}
        Read-only          : {_read_only}
        Store type         : {_store_type}""")

        kwargs = dataclasses.asdict(self)
        if self._chunk_shape is None:
            # for non-regular chunk grids
            kwargs["chunk_shape"] = "<variable>"
        if self._compressor is not None:
            template += "\nCompressor         : {_compressor}"

        if self._filters is not None:
            template += "\nFilters            : {_filters}"

        if self._codecs is not None:
            template += "\nCodecs             : {_codecs}"

        if self._count_bytes is not None:
            template += "\nNo. bytes          : {_count_bytes}"
            kwargs["_count_bytes"] = byte_info(self._count_bytes)

        if self._count_bytes_stored is not None:
            template += "\nNo. bytes stored   : {_count_bytes_stored}"
            kwargs["_count_stored"] = byte_info(self._count_bytes_stored)

        if (
            self._count_bytes is not None
            and self._count_bytes_stored is not None
            and self._count_bytes_stored > 0
        ):
            template += "\nStorage ratio      : {_storage_ratio}"
            kwargs["_storage_ratio"] = f"{self._count_bytes / self._count_bytes_stored:.1f}"

        if self._count_chunks_initialized is not None:
            template += "\nChunks Initialized : {_count_chunks_initialized}"
        return template.format(**kwargs)
