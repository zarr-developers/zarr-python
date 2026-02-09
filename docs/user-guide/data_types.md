# Array data types

## Zarr's Data Type Model

Zarr is designed for interoperability with NumPy, so if you are familiar with NumPy or any other
N-dimensional array library, Zarr's model for array data types should seem familiar. However, Zarr
data types have some unique features that are described in this document.

Zarr arrays operate under an essential design constraint: unlike NumPy arrays, Zarr arrays
are designed to be stored and accessed by other Zarr implementations. This means that, among other things,
Zarr data types must be serializable to metadata documents in accordance with the Zarr specifications,
which adds some unique aspects to the Zarr data type model.

The following sections explain Zarr's data type model in greater detail and demonstrate the
Zarr Python APIs for working with Zarr data types.

### Array Data Types

Every Zarr array has a data type, which defines the meaning of the array's elements. An array's data
type is encoded in the JSON metadata for the array. This means that the data type of an array must be
JSON-serializable.

In Zarr V2, the data type of an array is stored in the `dtype` field in array metadata.
Zarr V3 changed the name of this field to `data_type` and also defined new rules for the values
that can be assigned to the `data_type` field.

For example, in Zarr V2, the boolean array data type was represented in array metadata as the
string `"|b1"`. In Zarr V3, the same type is represented as the string `"bool"`.

### Scalars

Zarr also specifies how array elements, i.e., scalars, are encoded in array metadata. This is necessary
because Zarr uses a field in array metadata to define a default value for chunks that are not stored.
This field, called `fill_value` in both Zarr V2 and Zarr V3 metadata documents, contains a
JSON value that can be decoded to a scalar value compatible with the array's data type.

For the boolean data type, the scalar encoding is simple—booleans are natively supported by
JSON, so Zarr saves booleans as JSON booleans. Other scalars, like floats or raw bytes, have
more elaborate encoding schemes, and in some cases, this scheme depends on the Zarr format version.

## Data Types in Zarr Version 2

Version 2 of the Zarr format defined its data types relative to
[NumPy's data types](https://numpy.org/doc/2.1/reference/arrays.dtypes.html#data-type-objects-dtype),
and added a few non-NumPy data types as well. With one exception ([structured data types](#structured-data-type)), the Zarr
V2 JSON identifier for a data type is just the NumPy `str` attribute of that data type:

```python exec="true" session="data_types" source="above" result="ansi"
import zarr
import numpy as np
import json

store = {}
np_dtype = np.dtype('int64')
print(np_dtype.str)
```

```python exec="true" session="data_types" source="above" result="ansi"
z = zarr.create_array(store=store, shape=(1,), dtype=np_dtype, zarr_format=2)
dtype_meta = json.loads(store['.zarray'].to_bytes())["dtype"]
print(dtype_meta)
```

!!! note

    The `<` character in the data type metadata encodes the
    [endianness](https://numpy.org/doc/2.2/reference/generated/numpy.dtype.byteorder.html),
    or "byte order," of the data type. As per the NumPy model,
    in Zarr version 2 each data type has an endianness where applicable.
    However, Zarr version 3 data types do not store endianness information.

There are two special cases to consider: ["structured" data types](#structured-data-type), and
["object"](#object-data-type) data types.

### Structured Data Type

NumPy allows the construction of a so-called "structured" data types comprised of ordered collections
of named fields, where each field is itself a distinct NumPy data type. See the NumPy documentation
[here](https://numpy.org/doc/stable/user/basics.rec.html).

Crucially, NumPy does not use a special data type for structured data types—instead, NumPy
implements structured data types as an optional feature of the so-called "Void" data type, which models
arbitrary fixed-size byte strings. The `str` attribute of a regular NumPy void
data type is the same as the `str` of a NumPy structured data type. This means that the `str`
attribute does not convey information about the fields contained in a structured data type.
For these reasons, Zarr V2 uses a special data type encoding for structured data types.
They are stored in JSON as lists of pairs, where the first element is a string, and the second
element is a Zarr V2 data type specification. This representation supports recursion.

For example:

```python exec="true" session="data_types" source="above" result="ansi"
store = {}
np_dtype = np.dtype([('field_a', '>i2'), ('field_b', [('subfield_c', '>f4'), ('subfield_d', 'i2')])])
print(np_dtype.str)
```

```python exec="true" session="data_types" source="above" result="ansi"
z = zarr.create_array(store=store, shape=(1,), dtype=np_dtype, zarr_format=2)
dtype_meta = json.loads(store['.zarray'].to_bytes())["dtype"]
print(dtype_meta)
```

### Object Data Type

The NumPy "object" type is essentially an array of references to arbitrary Python objects.
It can model arrays of variable-length UTF-8 strings, arrays of variable-length byte strings, or
even arrays of variable-length arrays, each with a distinct data type. This makes the "object" data
type expressive, but also complicated to store.

Zarr Python cannot persistently store references to arbitrary Python objects. But if each of those Python
objects has a consistent type, then we can use a special encoding procedure to store the array. This
is how Zarr Python stores variable-length UTF-8 strings, or variable-length byte strings.

Although these are separate data types in this library, they are both "object" arrays in NumPy, which means
they have the *same* Zarr V2 string representation: `"|O"`.

So for Zarr V2 we have to disambiguate different "object" data type arrays on the basis of their
encoding procedure, i.e., the codecs declared in the `filters` and `compressor` attributes of array
metadata.

If an array with data type "object" used the `"vlen-utf8"` codec, then it was interpreted as an
array of variable-length strings. If an array with data type "object" used the `"vlen-bytes"`
codec, then it was interpreted as an array of variable-length byte strings.

This all means that the `dtype` field alone does not fully specify a data type in Zarr V2.
The name of the object codec used, if one was used, is also required.
Although this fact can be ignored for many simple numeric data types, any comprehensive approach to
Zarr V2 data types must either reject the "object" data types or include the "object codec"
identifier in the JSON form of the basic data type model.

## Data Types in Zarr Version 3

The NumPy-based Zarr V2 data type representation was effective for simple data types but struggled
with more complex data types, like "object" and "structured" data types. To address these limitations,
Zarr V3 introduced several key changes to how data types are represented:

- Instead of copying NumPy character codecs, Zarr V3 defines an identifier for each data type.
  The basic data types are identified by strings like `"int8"`, `"int16"`, etc., and data types
  that require a configuration can be identified by a JSON object.

  For example, this JSON object declares a datetime data type:

  ```json
  {
    "name": "numpy.datetime64",
    "configuration": {
      "unit": "s",
      "scale_factor": 10
    }
  }
  ```

- Zarr V3 data types do not have endianness. This is a departure from Zarr V2, where multi-byte
  data types are defined with endianness information. Instead, Zarr V3 requires that the endianness
  of encoded array chunks is specified in the `codecs` attribute of array metadata. The Zarr
  V3 specification leaves the in-memory endianness of decoded array chunks as an implementation detail.

For more about data types in Zarr V3, see the
[V3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html).

## Data Types in Zarr Python

The two Zarr formats that Zarr Python supports specify data types in different ways: data types in
Zarr version 2 are encoded as NumPy-compatible strings (or lists, in the case of structured data
types), while data types in Zarr V3 are encoded as either strings or JSON objects. Zarr V3 data
types do not have any associated endianness information, unlike Zarr V2 data types.

Zarr Python needs to support both Zarr V2 and V3, which means we need to abstract over these differences.
We do this with an abstract Zarr data type class: [ZDType][zarr.dtype.ZDType]
which provides Zarr V2 and Zarr V3 compatibility routines for "native" data types.

In this context, a "native" data type is a Python class, typically defined in another library, that
models an array's data type. For example, [`numpy.dtypes.UInt8DType`][] is a native data type defined in NumPy.
Zarr Python wraps the NumPy `uint8` with a [ZDType][zarr.dtype.ZDType] instance called
[UInt8][zarr.dtype.UInt8].

As of this writing, the only native data types Zarr Python supports are NumPy data types. We could
avoid the "native data type" jargon and just say "NumPy data type," but we do not want to rule out the
possibility of using non-NumPy array backends in the future.

Each data type supported by Zarr Python is modeled by a [ZDType][zarr.dtype.ZDType] subclass, which provides an
API for the following operations:

- Encoding and decoding a native data type
- Encoding and decoding a data type to and from Zarr V2 and Zarr V3 array metadata
- Encoding and decoding a scalar value to and from Zarr V2 and Zarr V3 array metadata
- Casting a Python object to a scalar value consistent with the data type

### List of data types

The following section lists the data types built in to Zarr Python. With a few exceptions, Zarr
Python supports nearly all of the data types in NumPy. If you need a data type that is not listed
here, it's possible to create it yourself: see [Adding New Data Types](#adding-new-data-types).

#### Boolean
- [Boolean][zarr.dtype.Bool]

#### Integral
- [Signed 8-bit integer][zarr.dtype.Int8]
- [Signed 16-bit integer][zarr.dtype.Int16]
- [Signed 32-bit integer][zarr.dtype.Int32]
- [Signed 64-bit integer][zarr.dtype.Int64]
- [Unsigned 8-bit integer][zarr.dtype.UInt8]
- [Unsigned 16-bit integer][zarr.dtype.UInt16]
- [Unsigned 32-bit integer][zarr.dtype.UInt32]
- [Unsigned 64-bit integer][zarr.dtype.UInt64]

#### Floating-point
- [16-bit floating-point][zarr.dtype.Float16]
- [32-bit floating-point][zarr.dtype.Float32]
- [64-bit floating-point][zarr.dtype.Float64]
- [64-bit complex floating-point][zarr.dtype.Complex64]
- [128-bit complex floating-point][zarr.dtype.Complex128]

#### String
- [Fixed-length UTF-32 string][zarr.dtype.FixedLengthUTF32]
- [Variable-length UTF-8 string][zarr.dtype.VariableLengthUTF8]

#### Bytes
- [Fixed-length null-terminated bytes][zarr.dtype.NullTerminatedBytes]
- [Fixed-length raw bytes][zarr.dtype.RawBytes]
- [Variable-length bytes][zarr.dtype.VariableLengthBytes]

#### Temporal
- [DateTime64][zarr.dtype.DateTime64]
- [TimeDelta64][zarr.dtype.TimeDelta64]

#### Struct-like
- [Structured][zarr.dtype.Structured]

### Example Usage

This section will demonstrates the basic usage of Zarr data types.

Create a `ZDType` from a native data type:

```python exec="true" session="data_types" source="above"
from zarr.core.dtype import Int8
import numpy as np
int8 = Int8.from_native_dtype(np.dtype('int8'))
```

Convert back to a native data type:

```python exec="true" session="data_types" source="above"
native_dtype = int8.to_native_dtype()
assert native_dtype == np.dtype('int8')
```

Get the default scalar value for the data type:

```python exec="true" session="data_types" source="above"
default_value = int8.default_scalar()
assert default_value == np.int8(0)
```

Serialize to JSON for Zarr V2:

```python exec="true" session="data_types" source="above" result="ansi"
json_v2 = int8.to_json(zarr_format=2)
print(json_v2)
{'name': '|i1', 'object_codec_id': None}
```

!!! note

    The representation returned by `to_json(zarr_format=2)` is more abstract than the literal contents
    of Zarr V2 array metadata, because the JSON representation used by the `ZDType` classes must be
    distinct across different data types. As noted [earlier](#object-data-type), Zarr V2 identifies
    multiple distinct data types with the "object" data type identifier `"|O"`. Extra information
    is needed to disambiguate these data types from one another. That's the reason for the
    `object_codec_id` field you see here.

And for V3:

```python exec="true" session="data_types" source="above" result="ansi"
json_v3 = int8.to_json(zarr_format=3)
print(json_v3)
```

Serialize a scalar value to JSON:

```python exec="true" session="data_types" source="above" result="ansi"
json_value = int8.to_json_scalar(42, zarr_format=3)
print(json_value)
```

Deserialize a scalar value from JSON:

```python exec="true" session="data_types" source="above"
scalar_value = int8.from_json_scalar(42, zarr_format=3)
assert scalar_value == np.int8(42)
```

### Adding New Data Types

Each Zarr data type is a separate Python class that inherits from
[ZDType][zarr.dtype.ZDType]. You can define a custom data type by
writing your own subclass of [ZDType][zarr.dtype.ZDType] and adding
your data type to the data type registry. To see an executable demonstration
of this process, see the [`custom_dtype` example](../user-guide/examples/custom_dtype.md).

### Data Type Resolution

Although Zarr Python uses a different data type model from NumPy, you can still define a Zarr array
with a NumPy data type object:

```python exec="true" session="data_types" source="above" result="ansi"
from zarr import create_array
import numpy as np
a = create_array({}, shape=(10,), dtype=np.dtype('int'))
print(a)
```

Or a string representation of a NumPy data type:

```python exec="true" session="data_types" source="above" result="ansi"
a = create_array({}, shape=(10,), dtype='<i8')
print(a)
```

The `Array` object presents itself like a NumPy array, including exposing a NumPy
data type as its `dtype` attribute:

```python exec="true" session="data_types" source="above" result="ansi"
print(type(a.dtype))
```

But if we inspect the metadata for the array, we can see the Zarr data type object:

```python
type(a.metadata.data_type)
<class 'zarr.core.dtype.npy.int.Int64'>
```

This example illustrates a general problem Zarr Python has to solve: how can we allow users to
specify a data type as a string or a NumPy `dtype` object, and produce the right Zarr data type
from that input? We call this process "data type resolution." Zarr Python also performs data type
resolution when reading stored arrays, although in this case the input is a JSON value instead
of a NumPy data type.

For simple data types like `int`, the solution could be extremely simple: just
maintain a lookup table that maps a NumPy data type to the Zarr data type equivalent. But not all
data types are so simple. Consider this case:

```python exec="true" session="data_types" source="above"
from zarr import create_array
import warnings
import numpy as np
warnings.simplefilter("ignore", category=FutureWarning)
a = create_array({}, shape=(10,), dtype=[('a', 'f8'), ('b', 'i8')])
print(a.dtype) # this is the NumPy data type
```

```python exec="true" session="data_types" source="above"
print(a.metadata.data_type) # this is the Zarr data type
```

In this example, we created a
[NumPy structured data type](https://numpy.org/doc/stable/user/basics.rec.html#structured-datatypes).
This data type is a container that can hold any NumPy data type, which makes it recursive. It is
not possible to make a lookup table that relates all NumPy structured data types to their Zarr
equivalents, as there is a nearly unbounded number of different structured data types. So instead of
a static lookup table, Zarr Python relies on a dynamic approach to data type resolution.

Zarr Python defines a collection of Zarr data types. This collection, called a "data type registry,"
is essentially a dictionary where the keys are strings (a canonical name for each data type), and the
values are the data type classes themselves. Dynamic data type resolution entails iterating over
these data type classes, invoking that class' [from_native_dtype][zarr.dtype.ZDType.from_native_dtype]
method, and returning a concrete data type instance if and only if exactly one of those constructor
invocations is successful.

In plain language, we take some user input, like a NumPy data type, offer it to all the
known data type classes, and return an instance of the one data type class that can accept that user input.

We want to avoid a situation where the same native data type matches multiple Zarr data types; that is,
a NumPy data type should *uniquely* specify a single Zarr data type. But data type resolution is
dynamic, so it's not possible to statically guarantee this uniqueness constraint. Therefore, we
attempt data type resolution against *every* data type class, and if, for some reason, a native data
type matches multiple Zarr data types, we treat this as an error and raise an exception.

If you have a NumPy data type and you want to get the corresponding `ZDType` instance, you can use
the `parse_dtype` function, which will use the dynamic resolution described above. `parse_dtype`
handles a range of input types:

- NumPy data types:

  ```python exec="true" session="data_types" source="above" result="ansi"
  import numpy as np
  from zarr.dtype import parse_dtype
  my_dtype = np.dtype('>M8[10s]')
  print(parse_dtype(my_dtype, zarr_format=2))
  ```

- NumPy data type-compatible strings:

  ```python exec="true" session="data_types" source="above" result="ansi"
  dtype_str = '>M8[10s]'
  print(parse_dtype(dtype_str, zarr_format=2))
  ```

- `ZDType` instances:

  ```python exec="true" session="data_types" source="above" result="ansi"
  from zarr.dtype import DateTime64
  zdt = DateTime64(endianness='big', scale_factor=10, unit='s')
  print(parse_dtype(zdt, zarr_format=2)) # Use a ZDType (this is a no-op)
  ```

- Python dictionaries (requires `zarr_format=3`). These dictionaries must be consistent with the
  `JSON` form of the data type:

  ```python exec="true" session="data_types" source="above" result="ansi"
  dt_dict = {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 10}}
  print(parse_dtype(dt_dict, zarr_format=3))
  ```

  ```python exec="true" session="data_types" source="above" result="ansi"
  print(parse_dtype(dt_dict, zarr_format=3).to_json(zarr_format=3))
  ```
