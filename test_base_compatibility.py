import asyncio
from zarr.storage import MemoryStore
from zarr.core.buffer import cpu

async def test_base_store_list_prefix_compatibility():
    """
    Test compatibility with the base StoreTests.test_list_prefix test.
    The base test creates these keys:
    - "zarr.json"
    - "a/zarr.json"
    - "a/b/zarr.json"
    - "a/b/c/zarr.json"
    
    And expects that list_prefix returns keys that startswith the prefix.
    """
    store = MemoryStore()
    buffer_cls = cpu.Buffer
    
    prefixes = ("", "a/", "a/b/", "a/b/c/")
    data = buffer_cls.from_bytes(b"")
    fname = "zarr.json"
    store_dict = {p + fname: data for p in prefixes}
    
    await store._set_many(store_dict.items())
    
    print("Keys in store:", list(store_dict.keys()))
    print()
    
    for prefix in prefixes:
        observed = sorted([k async for k in store.list_prefix(prefix)])
        
        # What the base test expects (simple startswith)
        expected_base = []
        for key in store_dict:
            if key.startswith(prefix):
                expected_base.append(key)
        expected_base = sorted(expected_base)
        
        print(f"prefix='{prefix}':")
        print(f"  observed: {observed}")
        print(f"  expected (base test): {expected_base}")
        
        # With the directory-aware fix, prefixes without "/" get "/" added
        # So "a" becomes "a/" and should match "a/zarr.json" and "a/b/zarr.json"
        if prefix == "":
            # Empty prefix should return all keys
            expected_new = sorted(store_dict.keys())
        else:
            # Non-empty prefix gets "/" appended
            normalized_prefix = prefix if prefix.endswith("/") else prefix + "/"
            expected_new = [k for k in sorted(store_dict.keys()) if k.startswith(normalized_prefix)]
        
        print(f"  expected (dir-aware): {expected_new}")
        
        # Check if observed matches expected
        if observed == expected_base:
            print(f"  ✓ Matches base test")
        elif observed == expected_new:
            print(f"  ✓ Matches dir-aware behavior")
        else:
            print(f"  ✗ MISMATCH!")
        print()

asyncio.run(test_base_store_list_prefix_compatibility())
