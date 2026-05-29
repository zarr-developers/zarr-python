from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True)
class Layout:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
