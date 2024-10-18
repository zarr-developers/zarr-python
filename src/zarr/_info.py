import dataclasses
import textwrap
from typing import Literal

# Group
# Name        : /
# Type        : zarr.hierarchy.Group
# Read-only   : False
# Store type  : zarr.storage.MemoryStore
# No. members : 0
# No. arrays  : 0
# No. groups  : 0


@dataclasses.dataclass(kw_only=True)
class GroupInfo:
    name: str
    type: Literal["Group"] = "Group"
    zarr_format: Literal[2, 3]
    read_only: bool
    store_type: str
    count_members: int | None = None
    count_arrays: int | None = None
    count_groups: int | None = None

    def __repr__(self) -> str:
        template = textwrap.dedent("""\
        Name        : {name}
        Type        : {type}
        Zarr format : {zarr_format}
        Read-only   : {read_only}
        Store type  : {store_type}""")

        if self.count_members is not None:
            template += "\nNo. members : {count_members}"
        if self.count_arrays is not None:
            template += "\nNo. arrays  : {count_arrays}"
        if self.count_groups is not None:
            template += "\nNo. groups  : {count_groups}"
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
        return size
    else:
        return f"{size} ({human_readable_size(size)})"


@dataclasses.dataclass(kw_only=True)
class ArrayInfo:
    type: Literal["Array"] = "Array"
    zarr_format: Literal[2, 3]
    data_type: str
    shape: tuple[int,]
    chunk_shape: tuple[int,]
    order: Literal["C", "F"]
    read_only: bool
    store_type: str
    compressor: str | None = None
    filters: list[str] | None = None
    codecs: list[str] | None = None
    count_bytes: int | None = None
    count_bytes_stored: int | None = None
    count_chunks_initialized: int | None = None

    def __repr__(self) -> str:
        template = textwrap.dedent("""\
        Type               : {type}
        Zarr format        : {zarr_format}
        Data type          : {data_type}
        Shape              : {shape}
        Chunk shape        : {chunk_shape}
        Order              : {order}
        Read-only          : {read_only}
        Store type         : {store_type}""")

        kwargs = dataclasses.asdict(self)
        if self.compressor is not None:
            template += "\nCompressor         : {compressor}"

        if self.filters is not None:
            template += "\nFilters            : {filters}"

        if self.codecs is not None:
            template += "\nCodecs             : {codecs}"

        if self.count_bytes is not None:
            template += "\nNo. bytes          : {count_bytes}"
            kwargs["count_bytes"] = byte_info(self.count_bytes)

        if self.count_bytes_stored is not None:
            template += "\nNo. bytes stored   : {count_bytes_stored}"
            kwargs["count_stored"] = byte_info(self.count_bytes_stored)

        if (
            self.count_bytes is not None
            and self.count_bytes_stored is not None
            and self.count_bytes_stored > 0
        ):
            template += "\nStorage ratio      : {storage_ratio}"
            kwargs["storage_ratio"] = f"{self.count_bytes / self.count_bytes_stored:.1f}"

        if self.count_chunks_initialized is not None:
            template += "\nChunks Initialized : {count_chunks_initialized}"
        return template.format(**kwargs)
