from pathlib import Path

import numpy as np
import pytest
from numcodecs import GZip

from zarr.core.group import GroupMetadata, create_hierarchy
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.storage import LocalStore


@pytest.fixture
def hierarchy_model(request: pytest.FixtureRequest) -> dict[str, ArrayV2Metadata | GroupMetadata]:
    dtype = np.uint8()
    return {
        "": GroupMetadata(attributes={"foo": "bar"}, zarr_format=2),
        "/array": ArrayV2Metadata(
            shape=(10, 10),
            dtype=dtype,
            chunks=(10, 10),
            compressor=GZip(),
            fill_value=1,
            order="C",
            filters=[GZip()],
        ),
    }


async def test_copy(
    tmp_path: Path, hierarchy_model: dict[str, ArrayV2Metadata | GroupMetadata]
) -> None:
    # create the hierarchy
    store = LocalStore(tmp_path)
    [x async for x in create_hierarchy(store=store, nodes=hierarchy_model)]
    breakpoint()
