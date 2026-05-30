# zarr-interfaces

Interface definitions (ABCs and protocols) for zarr codecs and data types.

This package provides the abstract base classes and protocols that external
codec and data type implementations should subclass or implement. It has
minimal dependencies (only numpy) and does not depend on zarr-python itself.

## Usage

```python
from zarr_interfaces.codec.v1 import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr_interfaces.data_type.v1 import ZDType
```

Interfaces are versioned under a `v1` namespace to support future evolution
without breaking existing implementations.
