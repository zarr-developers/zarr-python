# Chunk planning benchmarks

Run from the repository root, with `zarr-indexing` installed in the selected
Hatch environment:

```sh
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/chunk_planning.py --repeats 9
```

`chunk_planning.py` measures construction, projection traversal, local-column
access, and chunk-coordinate enumeration. See the script for workloads and output fields.

Use the same interpreter, dependencies, inputs, and script revision when
comparing checkouts. Put the intended checkout's package source on `PYTHONPATH`;
installed package metadata is also required for version lookup. Record both git
revisions and the interpreter/dependency versions alongside saved output.

Compare complete operations, not only constructor times: plans can defer work
until iteration. Measure allocation separately from elapsed time. Distinguish
streaming consumption from retaining all projections or coordinates, and report
whether input arrays and transform construction are included. Repeated local
column access measures cache reuse, which trades allocation against retained
memory. Bounded coordinate batches avoid constructing the full coordinate array.

The current script reports `peak_mib` as the incremental peak tracked by
`tracemalloc` during a separate invocation after the timed calls. It is not
process RSS or total retained memory. Inputs and most transforms are constructed
before measurement. The case walks construct plans and consume projections
without retaining them; `all_coordinates` materializes the complete coordinate
array. There is no bounded-batch coordinate workload in this script.

`local_rows` reuses one prepared table: its first timed invocation populates the
local-coordinate cache, while later timed invocations and the allocation probe
reuse that cache. Its median mixes a cold first call with warm calls, and its
reported peak excludes the already-retained cache. A separate fresh-table
measurement would be needed to quantify cold cache construction.

These scripts measure planning rather than codec or storage throughput. Repeat
measurements with alternating operation order before interpreting small timing
differences. Preserve raw benchmark output as an experiment artifact rather than
accumulating successive result tables in this README.
