import numpy as np

from zarr.core.dtype import UInt32

# This should work but fails with generic dtype
arr = np.array([1, 2, 3], dtype=np.uint32)
print(f"Array dtype: {arr.dtype}")
print(f"Array dtype type: {type(arr.dtype)}")

# Try to validate it
try:
    result = UInt32.from_native_dtype(arr.dtype)
    print("✅ SUCCESS: dtype validation works")
except Exception as e:
    print(f"❌ FAILED: {e}")
