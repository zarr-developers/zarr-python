from typing import Dict, List, Tuple, Union

BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]
Attributes = Dict[
    str, Union[Dict[str, "Attributes"], List["Attributes"], str, int, float, bool, None]
]
