from __future__ import annotations

from typing import Any, Literal

from donfig import Config

config = Config(
    "zarr",
    defaults=[
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "codec_pipeline": {"batch_size": 1},
        }
    ],
)


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return data
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
