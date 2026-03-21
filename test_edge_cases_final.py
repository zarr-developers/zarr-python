import asyncio
from zarr.storage import MemoryStore
from zarr.core.buffer import cpu

async def test_edge_cases():
    """Test edge cases for the list_prefix fix"""
    print("\n=== Testing edge cases ===\n")
    
    store = MemoryStore()
    buffer_cls = cpu.Buffer
    
    # Test case 1: Empty prefix
    print("Test 1: Empty prefix")
    store._store_dict = {
        'a': buffer_cls.from_bytes(b''),
        'b': buffer_cls.from_bytes(b''),
        'ab': buffer_cls.from_bytes(b''),
    }
    result = sorted([k async for k in store.list_prefix("")])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix(''): {result}")
    assert result == ['a', 'ab', 'b'], f"Empty prefix should return all keys, got {result}"
    print("  ✓ Empty prefix works correctly\n")
    
    # Test case 2: Prefix with trailing slash
    print("Test 2: Prefix with trailing slash")
    store._store_dict = {
        'dir/a': buffer_cls.from_bytes(b''),
        'dir/b': buffer_cls.from_bytes(b''),
        'dir_other/c': buffer_cls.from_bytes(b''),
    }
    result_with_slash = sorted([k async for k in store.list_prefix("dir/")])
    result_without_slash = sorted([k async for k in store.list_prefix("dir")])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix('dir/'): {result_with_slash}")
    print(f"  list_prefix('dir'): {result_without_slash}")
    assert result_with_slash == result_without_slash, "Trailing slash should not affect results"
    assert result_with_slash == ['dir/a', 'dir/b'], f"Expected ['dir/a', 'dir/b'], got {result_with_slash}"
    print("  ✓ Prefix with/without trailing slash works correctly\n")
    
    # Test case 3: Deeply nested paths
    print("Test 3: Deeply nested paths")
    store._store_dict = {
        'a/b/c/d': buffer_cls.from_bytes(b''),
        'a/b/c/e': buffer_cls.from_bytes(b''),
        'a/b/x/f': buffer_cls.from_bytes(b''),
    }
    result_a = sorted([k async for k in store.list_prefix("a")])
    result_ab = sorted([k async for k in store.list_prefix("a/b")])
    result_abc = sorted([k async for k in store.list_prefix("a/b/c")])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix('a'): {result_a}")
    print(f"  list_prefix('a/b'): {result_ab}")
    print(f"  list_prefix('a/b/c'): {result_abc}")
    assert result_a == ['a/b/c/d', 'a/b/c/e', 'a/b/x/f']
    assert result_ab == ['a/b/c/d', 'a/b/c/e', 'a/b/x/f']
    assert result_abc == ['a/b/c/d', 'a/b/c/e']
    print("  ✓ Deeply nested paths work correctly\n")
    
    # Test case 4: Issue #3773 - prefix matching should be directory-aware
    print("Test 4: Issue #3773 - Directory-aware prefix matching")
    store._store_dict = {
        '0/a': buffer_cls.from_bytes(b''),
        '0/b': buffer_cls.from_bytes(b''),
        '0_c/d': buffer_cls.from_bytes(b''),
        '1/e': buffer_cls.from_bytes(b''),
    }
    result_0 = sorted([k async for k in store.list_prefix('0')])
    result_0_c = sorted([k async for k in store.list_prefix('0_c')])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix('0'): {result_0}")
    print(f"  list_prefix('0_c'): {result_0_c}")
    assert result_0 == ['0/a', '0/b'], f"'0' should NOT match '0_c/', got {result_0}"
    assert result_0_c == ['0_c/d'], f"'0_c' should match '0_c/', got {result_0_c}"
    print("  ✓ Issue #3773 is fixed - directory-aware matching works\n")
    
    print("=== All edge cases passed! ===\n")

asyncio.run(test_edge_cases())
