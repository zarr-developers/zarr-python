from __future__ import annotations

import asyncio

from zarr.core.sync import Runner, SyncRunner


async def _coro() -> int:
    await asyncio.sleep(0)
    return 42


def test_sync_runner_runs_coroutine() -> None:
    runner = SyncRunner()
    assert runner.run(_coro()) == 42


def test_sync_runner_is_runner() -> None:
    assert isinstance(SyncRunner(), Runner)
