import asyncio
import zarr
from zarr.storage import MemoryStore, LocalStore
import tempfile
import shutil

async def test_list_prefix_consistency():
    """Test that MemoryStore and LocalStore have consistent list_prefix behavior"""
    
    print("\n=== Testing list_prefix consistency ===\n")
    
    # Test case 1: Basic directory matching
    print("Test 1: Basic directory matching")
    print("  Setup: keys = ['0/a', '0/b', '0_c/d', '1/e']")
    
    # MemoryStore
    mem_store = MemoryStore()
    mem_store._store_dict = {
        '0/a': zarr.core.buffer.cpu.Buffer.from_bytes(b'a'),
        '0/b': zarr.core.buffer.cpu.Buffer.from_bytes(b'b'),
        '0_c/d': zarr.core.buffer.cpu.Buffer.from_bytes(b'd'),
        '1/e': zarr.core.buffer.cpu.Buffer.from_bytes(b'e'),
    }
    
    mem_prefix_0 = sorted([k async for k in mem_store.list_prefix('0')])
    mem_prefix_0_slash = sorted([k async for k in mem_store.list_prefix('0/')])
    mem_prefix_0_c = sorted([k async for k in mem_store.list_prefix('0_c')])
    mem_prefix_1 = sorted([k async for k in mem_store.list_prefix('1')])
    
    print(f"  MemoryStore.list_prefix('0'): {mem_prefix_0}")
    print(f"  MemoryStore.list_prefix('0/'): {mem_prefix_0_slash}")
    print(f"  MemoryStore.list_prefix('0_c'): {mem_prefix_0_c}")
    print(f"  MemoryStore.list_prefix('1'): {mem_prefix_1}")
    
    # LocalStore
    tmpdir = tempfile.mkdtemp()
    local_store = LocalStore(tmpdir)
    
    # Create files
    for key, val in mem_store._store_dict.items():
        path = local_store.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(val.to_bytes())
    
    local_prefix_0 = sorted([k async for k in local_store.list_prefix('0')])
    local_prefix_0_slash = sorted([k async for k in local_store.list_prefix('0/')])
    local_prefix_0_c = sorted([k async for k in local_store.list_prefix('0_c')])
    local_prefix_1 = sorted([k async for k in local_store.list_prefix('1')])
    
    print(f"  LocalStore.list_prefix('0'): {local_prefix_0}")
    print(f"  LocalStore.list_prefix('0/'): {local_prefix_0_slash}")
    print(f"  LocalStore.list_prefix('0_c'): {local_prefix_0_c}")
    print(f"  LocalStore.list_prefix('1'): {local_prefix_1}")
    
    # Assertions
    assert mem_prefix_0 == local_prefix_0, f"Mismatch for prefix '0': {mem_prefix_0} != {local_prefix_0}"
    assert mem_prefix_0_slash == local_prefix_0_slash, f"Mismatch for prefix '0/': {mem_prefix_0_slash} != {local_prefix_0_slash}"
    assert mem_prefix_0_c == local_prefix_0_c, f"Mismatch for prefix '0_c': {mem_prefix_0_c} != {local_prefix_0_c}"
    assert mem_prefix_1 == local_prefix_1, f"Mismatch for prefix '1': {mem_prefix_1} != {local_prefix_1}"
    
    print("  ✓ All assertions passed!\n")
    
    # Test case 2: Original issue reproduction
    print("Test 2: Original issue reproduction")
    print("  Setup: zarr group with '0' group and '0_c' array")
    
    mem_store2 = MemoryStore()
    root_mem = zarr.open_group(mem_store2, mode="w")
    root_mem.create_group("0")
    root_mem.create_array("0_c", shape=(1,), dtype="i4")
    
    mem_result = sorted([k async for k in mem_store2.list_prefix("0")])
    print(f"  MemoryStore.list_prefix('0'): {mem_result}")
    
    tmpdir2 = tempfile.mkdtemp()
    local_store2 = LocalStore(tmpdir2)
    root_local = zarr.open_group(local_store2, mode="w")
    root_local.create_group("0")
    root_local.create_array("0_c", shape=(1,), dtype="i4")
    
    local_result = sorted([k async for k in local_store2.list_prefix("0")])
    print(f"  LocalStore.list_prefix('0'): {local_result}")
    
    assert mem_result == local_result, f"Results don't match: {mem_result} != {local_result}"
    assert mem_result == ['0/zarr.json'], f"Expected ['0/zarr.json'], got {mem_result}"
    print("  ✓ Original issue is fixed!\n")
    
    # Cleanup
    shutil.rmtree(tmpdir)
    shutil.rmtree(tmpdir2)
    
    print("=== All tests passed! ===\n")

# Run the test
asyncio.run(test_list_prefix_consistency())
