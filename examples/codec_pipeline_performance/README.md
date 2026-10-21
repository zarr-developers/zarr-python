# Codec Pipeline Performance

This example compares the default `BatchedCodecPipeline` against the opt-in
`FusedCodecPipeline` on a sharded array, across two stores (memory and local)
and two codec regimes (uncompressed and gzip), at one worker and at `cpu_count`.

A *codec pipeline* turns chunks of array data into stored bytes and back, running
the configured codecs and performing the storage IO. The default
`BatchedCodecPipeline` schedules both asynchronously -- roughly one coroutine per
chunk operation. That model pays off for high-latency stores, where there is
useful work to do while waiting on IO. For low-latency stores (in-process memory,
the local filesystem) the IO completes too quickly for the overlap to be worth
its cost, and the async scheduling becomes pure overhead.

`FusedCodecPipeline` runs codec compute and synchronous IO synchronously,
removing that overhead. It is
[experimental](https://zarr.readthedocs.io/en/stable/user-guide/experimental/)
and opt-in; the default pipeline is unchanged.

## What it shows

- How to select a pipeline with `zarr.config.set`, and why the array must be
  *created* inside the config block: the pipeline class is resolved at array
  construction time and then travels with the array.
- That the benefit depends strongly on layout and on whether compression is in
  play. Some configurations are slower under the fused pipeline -- the script
  reports speedups below 1.00x rather than hiding them.
- That `codec_pipeline.max_workers` is read only by `FusedCodecPipeline`; the
  default pipeline ignores it entirely.

## Running

```bash
uv run codec_pipeline_performance.py
```

The script has no arguments and writes only to an in-memory store.

## Interpreting the output

The numbers are specific to your CPU, your Python build, and the workload chosen
here. They are a measurement of your machine, not a published benchmark -- treat
a single run as indicative and re-measure against your own data and store before
switching pipelines in production.

Two effects are worth watching for:

- **The two codec regimes tell opposite stories about `max_workers`.**
  Uncompressed IO is dominated by per-chunk *scheduling*, so `Fused (1 worker)`
  is already fastest and a thread pool only adds overhead. gzip is genuinely
  CPU-bound: a single worker compresses chunk after chunk sequentially and can
  be *slower than the default*, while a thread pool spreads that compression
  over cores and reclaims the win. That flip is why the fused pipeline is
  threaded by default, and why pinning `max_workers=1` is worth it for
  memory-backed uncompressed data.
- **Chunk size decides whether threading can help at all.** The 64×64 inner
  chunks here are small enough that per-chunk scheduling dominates uncompressed
  IO, yet large enough that per-chunk gzip is real work to parallelize. Much
  coarser chunks leave the pool with too few items to spread.
