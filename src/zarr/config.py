from __future__ import annotations

from typing import Any, Literal, cast

from donfig import Config

config = Config(
    "zarr",
    defaults=[
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "codec_pipeline": {"batch_size": 1},
            "json_indent": 2,
        }
    ],
)


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast(Literal["C", "F"], data)
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
