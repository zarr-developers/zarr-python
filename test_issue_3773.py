import asyncio
import zarr
from zarr.storage import MemoryStore, LocalStore
import tempfile
import shutil

async def test_issue_3773_list_prefix_divergence():
    """
    Test for issue #3773: Divergent behavior of memory store and localstore list_prefix
    
    When calling list_prefix with a prefix like "0", it should only match keys
    that are under the "0/" directory, not keys that happen to start with "0"
    like "0_c/...".
    
    Example:
    - Keys: "0/zarr.json", "0_c/zarr.json"
    - list_prefix("0") should return: ["0/zarr.json"]
    - Should NOT return: ["0_c/zarr.json"]
    """
    print("\n=== Issue #3773: list_prefix divergence ===\n")
    
    # Test with actual zarr groups
    print("Test: Creating zarr group with '0' group and '0_c' array")
    
    # MemoryStore test
    print("\n  MemoryStore:")
    mem_store = MemoryStore()
    root_mem = zarr.open_group(mem_store, mode="w")
    root_mem.create_group("0")
    root_mem.create_array("0_c", shape=(1,), dtype="i4")
    
    mem_result = sorted([k async for k in mem_store.list_prefix("0")])
    print(f"    list_prefix('0'): {mem_result}")
    
    # LocalStore test
    print("  LocalStore:")
    tmpdir = tempfile.mkdtemp()
    try:
        local_store = LocalStore(tmpdir)
        root_local = zarr.open_group(local_store, mode="w")
        root_local.create_group("0")
        root_local.create_array("0_c", shape=(1,), dtype="i4")
        
        local_result = sorted([k async for k in local_store.list_prefix("0")])
        print(f"    list_prefix('0'): {local_result}")
    finally:
        shutil.rmtree(tmpdir)
    
    # Verify consistency
    print("\n  Verification:")
    print(f"    Results match: {mem_result == local_result}")
    print(f"    Expected behavior (LocalStore): {mem_result == ['0/zarr.json']}")
    
    assert mem_result == local_result, (
        f"MemoryStore and LocalStore results don't match!\n"
        f"  MemoryStore: {mem_result}\n"
        f"  LocalStore: {local_result}"
    )
    
    assert mem_result == ['0/zarr.json'], (
        f"Expected ['0/zarr.json'], got {mem_result}.\n"
        f"The prefix '0' should only match '0/' directory, not '0_c/'."
    )
    
    print("\n✓ Issue #3773 is fixed!\n")

asyncio.run(test_issue_3773_list_prefix_divergence())
