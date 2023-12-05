from __future__ import annotations

from asyncio import AbstractEventLoop
from typing import Literal, Optional
from attr import frozen


@frozen
class SyncConfiguration:
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


@frozen
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    # TODO: remove these in favor of the SyncConfiguration object
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None
