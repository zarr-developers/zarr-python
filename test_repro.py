import asyncio
import zarr

async def list_prefix(store, prefix):
    return sorted([k async for k in store.list_prefix(prefix)])

async def run():
    print("\n=== Testing with actual data ===")
    
    # Create memory store
    print("\nMemoryStore:")
    store_mem = zarr.storage.MemoryStore()
    root_mem = zarr.open_group(store_mem, mode="w")
    root_mem.create_group("0")
    root_mem.create_array("0_c", shape=(1,), dtype="i4")
    result = await list_prefix(store_mem, "0")
    print(f"  list_prefix('0'): {result}")
    
    # Also print all keys
    all_keys = sorted([k async for k in store_mem.list()])
    print(f"  all keys: {all_keys}")
    
    # Create local store
    print("\nLocalStore:")
    import shutil
    import tempfile
    tmpdir = tempfile.mkdtemp()
    store_local = zarr.storage.LocalStore(root=tmpdir)
    root_local = zarr.open_group(store_local, mode="w")
    root_local.create_group("0")
    root_local.create_array("0_c", shape=(1,), dtype="i4")
    result = await list_prefix(store_local, "0")
    print(f"  list_prefix('0'): {result}")
    
    # Also print all keys
    all_keys = sorted([k async for k in store_local.list()])
    print(f"  all keys: {all_keys}")
    
    # Cleanup
    shutil.rmtree(tmpdir)
    
    print("\n=== The Issue ===")
    print("When calling list_prefix('0'):")
    print("- MemoryStore returns both '0/zarr.json' and '0_c/zarr.json'")
    print("- LocalStore returns only '0/zarr.json'")
    print("\nThe divergence is in how they interpret the prefix:")
    print("- MemoryStore: Uses string.startswith() which matches '0_c' because it starts with '0'")
    print("- LocalStore: Treats '0' as a directory path, so it doesn't match '0_c'")

asyncio.run(run())
