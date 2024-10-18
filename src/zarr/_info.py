import dataclasses
import textwrap
from typing import Literal

import zarr.abc.store

# Group
# Name        : /
# Type        : zarr.hierarchy.Group
# Read-only   : False
# Store type  : zarr.storage.MemoryStore
# No. members : 0
# No. arrays  : 0
# No. groups  : 0


# In [19]: z.info
# Out[19]:
# Type               : zarr.core.Array
# Data type          : int32
# Shape              : (1000000,)
# Chunk shape        : (100000,)
# Order              : C
# Read-only          : False
# Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
# Store type         : zarr.storage.KVStore
# No. bytes          : 4000000 (3.8M)
# No. bytes stored   : 320
# Storage ratio      : 12500.0
# Chunks initialized : 0/10


@dataclasses.dataclass(kw_only=True)
class GroupInfo:
    name: str
    type: Literal["Group"] = "Group"
    read_only: bool
    store_type: str
    count_members: int | None = None
    count_arrays: int | None = None
    count_groups: int | None = None

    def __repr__(self) -> str:
        template = textwrap.dedent("""\
        Name        : {name}
        Type        : {type}
        Read-only   : {read_only}
        Store type  : {store_type}""")

        if self.count_members is not None:
            template += ("\nNo. members : {count_members}")
        if self.count_arrays is not None:
            template += ("\nNo. arrays  : {count_arrays}")
        if self.count_groups is not None:
            template += ("\nNo. groups  : {count_groups}")
        return template.format(
            **dataclasses.asdict(self)
        )

    # def _repr_html_(self): ...
