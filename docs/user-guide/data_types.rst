Data types
==========

Zarr's data type model
----------------------

Every Zarr array has a "data type", which defines the meaning and physical layout of the
array's elements. Zarr is heavily influenced by `NumPy <https://numpy.org/doc/stable/>`_, and
Zarr-Python supports creating arrays with Numpy data types::

  >>> import zarr
  >>> import numpy as np
  >>> z = zarr.create_array(store={}, shape=(10,), dtype=np.dtype('uint8'))
  >>> z
  <Array memory:... shape=(10,) dtype=uint8>

Unlike Numpy arrays, Zarr arrays are designed to be persisted to storage and read by Zarr implementations in different programming languages.
This means Zarr data types must be interpreted correctly when clients read an array. So each Zarr data type defines a procedure for
encoding/decoding that data type to/from Zarr array metadata, and also encoding/decoding **instances** of that data type to/from
array metadata. These serialization procedures depend on the Zarr format.

Data types in Zarr version 2
-----------------------------

Version 2 of the Zarr format defined its data types relative to `Numpy's data types <https://numpy.org/doc/2.1/reference/arrays.dtypes.html#data-type-objects-dtype>`_, and added a few non-Numpy data types as well.
Thus the JSON identifier for a Numpy-compatible data type is just the Numpy ``str`` attribute of that dtype::

    >>> import zarr
    >>> import numpy as np
    >>> import json
    >>> store = {}
    >>> np_dtype = np.dtype('int64')
    >>> z = zarr.create_array(store=store, shape=(1,), dtype=np_dtype, zarr_format=2)
    >>> dtype_meta = json.loads(store['.zarray'].to_bytes())["dtype"]
    >>> assert dtype_meta == np_dtype.str  # True
    >>> dtype_meta
    '<i8'

.. note::
   The ``<`` character in the data type metadata encodes the `endianness <https://numpy.org/doc/2.2/reference/generated/numpy.dtype.byteorder.html>`_, or "byte order", of the data type. Following Numpy's example,
   in Zarr version 2 each data type has an endianness where applicable. However, Zarr version 3 data types do not store endianness information.

In addition to defining a representation of the data type itself (which in the example above was just a simple string ``"<i8"``), Zarr also
defines a metadata representation of scalars associated with that data type. Integers are stored as ``JSON`` numbers,
as are floats, with the caveat that `NaN`, positive infinity, and negative infinity are stored as special strings.

Data types in Zarr version 3
-----------------------------

Zarr V3 brings several key changes to how data types are represented:

- Zarr V3 identifies the basic data types as strings like ``int8``, ``int16``, etc. In Zarr V2 ``int8`` would represented as ``|i1``,  ``int16`` would be ``>i2`` **or** ``<i2``, depending on the endianness.
- A Zarr V3 data type does not have endianness. This is a departure from Zarr V2, where multi-byte data types would be stored in ``JSON`` with an encoding that included endianness. Instead,
  Zarr V3 requires that endianness, where applicable, is specified in the ``codecs`` attribute of array metadata.
- Zarr V3 data types can also take the form of a ``JSON`` object like
  ``{"name": "foo", "configuration": {"parameter": "value"}}``. This structure facilitates specifying data types that take parameters.


Data types in Zarr-Python
-------------------------

The two Zarr formats that Zarr-Python supports specify data types in two different ways:
data types in Zarr version 2 are encoded as Numpy-compatible strings, while data types in Zarr version
3 are encoded as either strings or ``JSON`` objects,
and the Zarr V3 data types don't have any associated endianness information, unlike Zarr V2 data types.

To abstract over these syntactical and semantic differences, Zarr-Python uses a class called `ZDType <../api/zarr/dtype/index.html#zarr.dtype.ZDType>`_ to wrap native data types (e.g., Numpy data types) and provide Zarr V2 and Zarr V3 compatibility routines.
Each data type supported by Zarr-Python is modeled by a subclass of ``ZDType``, which provides an API for the following operations:

- Wrapping / unwrapping a native data type
- Encoding / decoding a data type to / from Zarr V2 and Zarr V3 array metadata.
- Encoding / decoding a scalar value to / from Zarr V2 and Zarr V3 array metadata.


Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from zarr.core.dtype.wrapper import Int8

    # Create a ZDType instance from a native dtype
    int8 = Int8.from_dtype(np.dtype('int8'))

    # Convert back to native dtype
    native_dtype = int8.to_dtype()
    assert native_dtype == np.dtype('int8')

    # Get the default value
    default_value = int8.default_value()
    assert default_value == np.int8(0)

    # Serialize to JSON
    json_representation = int8.to_json(zarr_format=3)

    # Serialize a scalar value
    json_value = int8.to_json_value(42, zarr_format=3)
    assert json_value == 42

    # Deserialize a scalar value
    scalar_value = int8.from_json_value(42, zarr_format=3)
    assert scalar_value == np.int8(42)

Custom Data Types
~~~~~~~~~~~~~~~~~

Users can define custom data types by subclassing `ZDType` and implementing the required methods.
Once defined, the custom data type can be registered with Zarr-Python to enable seamless integration with the library.

<TODO: example of defining a custom data type>