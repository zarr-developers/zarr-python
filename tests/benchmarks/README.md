# Zarr Benchmarks

This directory contains performance benchmarks for zarr-python.

## Running Benchmarks

```bash
# Run all benchmarks
hatch run test.py3.12-2.2-minimal:pytest tests/benchmarks --benchmark-enable

# Run specific test
hatch run test.py3.12-2.2-minimal:pytest tests/benchmarks/test_e2e.py::test_write_array_large_chunks --benchmark-enable

# Compare results between branches (see root compare_benchmarks.py)
pytest tests/benchmarks --benchmark-only --benchmark-json=results.json
```

## Test Structure

### test_e2e.py

End-to-end read/write performance tests.

#### Standard Tests
- **test_write_array**: Small chunks (1024 elements), various shard configurations
- **test_read_array**: Small chunks (1024 elements), various shard configurations

Configuration:
- Array: 1 MB (1,048,576 elements)
- Chunks: 1,024 elements (4 KB)
- Shards: None, 1,024, or 65,536 elements
- Data types: uint8
- Compression: None, gzip
- Stores: memory, local

**Use case**: Typical small-array workloads, general performance monitoring

#### Large-Chunk Tests
- **test_write_array_large_chunks**: Many chunks per shard (1000 chunks)
- **test_read_array_large_chunks**: Many chunks per shard (1000 chunks)

Configuration:
- Arrays:
  - 1D: 100K elements (391 KB), 1000 chunks of 100 elements each
  - 2D: 10M elements (38 MB), 1000 chunks of 10K elements each
- Chunks: Large (100-10K elements)
- Shards: Very large (100K-10M elements, containing 1000 chunks)
- Data types: float32
- Compression: None
- Stores: memory only (for speed)

**Use case**: Tests the np.concatenate optimization in sharding codec. Mirrors the workload from `sharding_benchmark.py`:
```bash
uv run --with-editable . sharding_benchmark.py \
  --num-iterations 2 \
  --shape 1000 100000 \
  --chunks 1 100000 \
  --shard-chunks 100 1000000
```

**Why this matters**: The sharding codec concatenates all chunks when writing a shard. With 1000 chunks per shard, the old implementation (multiple `np.concatenate` calls) was O(N²), while the new implementation (single concatenate) is O(N). This benchmark captures that difference.

## Understanding the Results

### Small-Chunk Tests
Expected improvements from optimization:
- **4-7% faster** for write operations
- Read operations mostly neutral

### Large-Chunk Tests
Expected improvements from optimization:
- **10-50% faster** for write operations with many chunks per shard
- Demonstrates the value of the np.concatenate optimization

### What to Look For
- **Write performance**: Should improve consistently across all configurations
- **Read performance**: Should remain stable or improve slightly
- **Scaling**: Large-chunk tests should show bigger improvements than small-chunk tests

## Benchmark History

- **PR #3562**: Added initial benchmark infrastructure
- **nbren12/main**: Optimized shard writing by replacing multiple `np.concatenate` calls with single call
  - Small chunks: 4-7% improvement
  - Large chunks: ~50% overhead reduction (13.7x → 7.0x slower than unsharded)

## Adding New Benchmarks

When adding new benchmarks:
1. Use `@pytest.mark.parametrize` to test multiple configurations
2. Document what workload the benchmark represents
3. Use descriptive function and parameter names
4. Add comments explaining why the configuration matters
5. Update this README
