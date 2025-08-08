from __future__ import annotations

import asyncio

from zarr.storage._wrapper import WrapperStore
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype


class LatencyStore(WrapperStore[Store]):
    """
    A wrapper class that takes any store class in its constructor and
    adds latency to the `set` and `get` methods. This can be used for
    performance testing.

    Particularly useful for testing downstream applications which will 
    interact with a high-latency zarr store implementation, 
    such as one which read from or writes to remote object storage. 
    For example, by using this class to wrap a ``MemoryStore`` instance, 
    you can (crudely) simulate the latency of reading and writing from S3 
    without having to actually use the network, or a mock like MinIO.

    Parameters
    ----------
    store : Store
        Store to wrap
    get_latency : float
        Amount of latency to add to each get call, in seconds. Default is 0.
    set_latency : float
        Amount of latency to add to each set call, in seconds. Default is 0.
    """

    get_latency: float
    set_latency: float

    def __init__(self, cls: Store, *, get_latency: float = 0, set_latency: float = 0) -> None:
        self.get_latency = float(get_latency)
        self.set_latency = float(set_latency)
        self._store = cls

    async def set(self, key: str, value: Buffer) -> None:
        """
        Add latency to the ``set`` method.

        Calls ``asyncio.sleep(self.set_latency)`` before invoking the wrapped ``set`` method.

        Parameters
        ----------
        key : str
            The key to set
        value : Buffer
            The value to set

        Returns
        -------
        None
        """
        await asyncio.sleep(self.set_latency)
        await self._store.set(key, value)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """
        Add latency to the ``get`` method.

        Calls ``asyncio.sleep(self.get_latency)`` before invoking the wrapped ``get`` method.

        Parameters
        ----------
        key : str
            The key to get
        prototype : BufferPrototype
            The BufferPrototype to use.
        byte_range : ByteRequest, optional
            An optional byte range.

        Returns
        -------
        buffer : Buffer or None
        """
        await asyncio.sleep(self.get_latency)
        return await self._store.get(key, prototype=prototype, byte_range=byte_range)
