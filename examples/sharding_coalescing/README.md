# Sharded Read Coalescing

This example demonstrates byte-range coalescing for partial reads of sharded
arrays, a performance optimization added in Zarr-Python 3.3.0 and enabled by
default.

A shard is one stored object containing many inner chunks, each occupying its own
byte range. Reading N inner chunks could mean N separate byte-range requests.
Because byte ranges are intervals, nearby ranges can be merged: `[a, b)` and
`[b, c)` together cover `[a, c)`, so a single request can serve both. Merging
trades reading some bytes you did not ask for against issuing fewer requests --
worthwhile whenever a request is expensive, as with object storage.

## What it shows

- Reading scattered inner chunks from one shard with coalescing **off** issues
  one store request per inner chunk; with the **default** settings the same read
  collapses to a single request.
- The resulting wall-clock difference against a store with simulated latency.
- That coalescing changes only *how* data is fetched, never *what* is returned --
  the script asserts both configurations produce identical arrays.
- A case where coalescing changes nothing: a contiguous selection already has
  adjacent byte ranges, so it merges under any setting.

## Running

```bash
uv run sharding_coalescing.py
```

## How the comparison is set up

Two details make the effect observable, and both are worth understanding if you
adapt this script:

- **The selection must have gaps.** A contiguous read produces adjacent byte
  ranges that merge regardless of configuration. The strided selections skip
  inner chunks, creating the gaps that the `sharding_coalesce_max_gap_bytes`
  budget decides whether to bridge.
- **Latency must be charged per merged fetch.** The example defines a small
  `WrapperStore` subclass that sleeps in `get`. It deliberately does *not* keep
  `WrapperStore.get_ranges`, which forwards straight to the wrapped store and
  would bypass the latency entirely; inheriting the `Store` ABC's `get_ranges`
  instead runs the coalescer over its own `get`, so each merged fetch pays once.

  `zarr.testing.store` ships a ready-made `LatencyStore`, but importing it pulls
  in `pytest`. Defining the wrapper inline keeps the example runnable with only
  `zarr` and `numpy` installed.

## Configuration

Two settings control the behavior, both settable globally via `zarr.config` or
per array via `config=` on `zarr.create_array` / `Array.with_config`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `sharding_coalesce_max_gap_bytes` | 1 MiB | Merge two ranges only if the gap between them is no larger than this |
| `sharding_coalesce_max_bytes` | 16 MiB | Never let a merged read exceed this size |

Setting the gap to `0` merges only exactly-adjacent ranges, which approximates
the pre-3.3.0 behavior; that is how the example emulates the old path. Raising
the gap reads more unwanted bytes in exchange for fewer round trips -- the right
value depends on how expensive a request is against how fast your link is.
