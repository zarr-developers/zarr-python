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
        'a/b': buffer_cls.from_bytes(b''),
    }
    result_a = sorted([k async for k in store.list_prefix("a")])
    result_ab = sorted([k async for k in store.list_prefix("a/b")])
    result_abc = sorted([k async for k in store.list_prefix("a/b/c")])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix('a'): {result_a}")
    print(f"  list_prefix('a/b'): {result_ab}")
    print(f"  list_prefix('a/b/c'): {result_abc}")
    assert result_a == ['a/b', 'a/b/c/d', 'a/b/c/e', 'a/b/x/f']
    assert result_ab == ['a/b', 'a/b/c/d', 'a/b/c/e', 'a/b/x/f']
    assert result_abc == ['a/b/c/d', 'a/b/c/e']
    print("  ✓ Deeply nested paths work correctly\n")
    
    # Test case 4: Special characters in prefix
    print("Test 4: Special characters in prefix")
    store._store_dict = {
        'arr[0]/data': buffer_cls.from_bytes(b''),
        'arr/data': buffer_cls.from_bytes(b''),
    }
    result = sorted([k async for k in store.list_prefix("arr[")])
    print(f"  Keys: {sorted(store._store_dict.keys())}")
    print(f"  list_prefix('arr['): {result}")
    assert result == ['arr[0]/data'], f"Expected ['arr[0]/data'], got {result}"
    print("  ✓ Special characters in prefix work correctly\n")
    
    print("=== All edge cases passed! ===\n")

asyncio.run(test_edge_cases())
