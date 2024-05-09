from __future__ import annotations

from asyncio import AbstractEventLoop
from dataclasses import dataclass
from typing import Any, Literal, Optional

from donfig import Config

config = Config(
    "zarr",
    defaults=[{"async": {"concurrency": None, "timeout": None}}],
)


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return data
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)


def parse_asyncio_loop(data: Any) -> AbstractEventLoop | None:
    if data is None or isinstance(data, AbstractEventLoop):
        return data
    raise TypeError(f"Expected AbstractEventLoop or None, got {type(data)}")


@dataclass(frozen=True)
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    asyncio_loop: Optional[AbstractEventLoop] = None

    def __init__(
        self,
        order: Literal["C", "F"] = "C",
        asyncio_loop: Optional[AbstractEventLoop] = None,
    ):
        order_parsed = parse_indexing_order(order)
        asyncio_loop_parsed = parse_asyncio_loop(asyncio_loop)

        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "asyncio_loop_parsed", asyncio_loop_parsed)
