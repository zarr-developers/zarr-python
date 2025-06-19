Data types
==========

Zarr's data type model
----------------------

Every Zarr array has a "data type", which defines the meaning and physical layout of the
array's elements. As Zarr Python is tightly integrated with `NumPy <https://numpy.org/doc/stable/>`_,
it's easy to create arrays with NumPy data types:

.. code-block:: python

  >>> import zarr
  >>> import numpy as np
  >>> z = zarr.create_array(store={}, shape=(10,), dtype=np.dtype('uint8'))
  >>> z
  <Array memory:... shape=(10,) dtype=uint8>

Unlike NumPy arrays, Zarr arrays are designed to accessed by Zarr
implementations in different programming languages. This means Zarr data types must be interpreted
correctly when clients read an array. Each Zarr data type defines procedures for
encoding and decoding both the data type itself, and scalars from that data type to and from Zarr array metadata. And these serialization procedures
depend on the Zarr format.

Data types in Zarr version 2
-----------------------------

Version 2 of the Zarr format defined its data types relative to
`NumPy's data types <https://numpy.org/doc/2.1/reference/arrays.dtypes.html#data-type-objects-dtype>`_,
and added a few non-NumPy data types as well. Thus the JSON identifier for a NumPy-compatible data
type is just the NumPy ``str`` attribute of that data type:

.. code-block:: python

  >>> import zarr
  >>> import numpy as np
  >>> import json
  >>>
  >>> store = {}
  >>> np_dtype = np.dtype('int64')
  >>> z = zarr.create_array(store=store, shape=(1,), dtype=np_dtype, zarr_format=2)
  >>> dtype_meta = json.loads(store['.zarray'].to_bytes())["dtype"]
  >>> dtype_meta
  '<i8'
  >>> assert dtype_meta == np_dtype.str

.. note::
   The ``<`` character in the data type metadata encodes the
   `endianness <https://numpy.org/doc/2.2/reference/generated/numpy.dtype.byteorder.html>`_,
   or "byte order", of the data type. Following NumPy's example,
   in Zarr version 2 each data type has an endianness where applicable.
   However, Zarr version 3 data types do not store endianness information.

In addition to defining a representation of the data type itself (which in the example above was
just a simple string ``"<i8"``), Zarr also
defines a metadata representation for scalars associated with each data type. This is necessary
because Zarr arrays have a ``JSON``-serializable ``fill_value`` attribute that defines a scalar value to use when reading
uninitialized chunks of a Zarr array.
Integer and float scalars are stored as ``JSON`` numbers, except for special floats like ``NaN``,
positive infinity, and negative infinity, which are stored as strings.

More broadly, each Zarr data type defines its own rules for how scalars of that type are stored in
``JSON``.


Data types in Zarr version 3
-----------------------------

Zarr V3 brings several key changes to how data types are represented:

- Zarr V3 identifies the basic data types as strings like ``"int8"``, ``"int16"``, etc.

  By contrast, Zarr V2 uses the NumPy character code representation for data types:
  In Zarr V2, ``int8`` is represented as ``"|i1"``.
- A Zarr V3 data type does not have endianness. This is a departure from Zarr V2, where multi-byte
  data types are defined with endianness information. Instead, Zarr V3 requires that endianness,
  where applicable, is specified in the ``codecs`` attribute of array metadata.
- While some Zarr V3 data types are identified by strings, others can be identified by a ``JSON``
  object. For example, consider this specification of a ``datetime`` data type:

  .. code-block:: json

    {
      "name": "numpy.datetime64",
      "configuration": {
        "unit": "s",
        "scale_factor": 10
      }
    }


  Zarr V2 generally uses structured string representations to convey the same information. The
  data type given in the previous example would be represented as the string ``">M[10s]"`` in
  Zarr V2. This is more compact, but can be harder to parse.

For more about data types in Zarr V3, see the
`V3 specification <https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html>`_.

Data types in Zarr Python
-------------------------

The two Zarr formats that Zarr Python supports specify data types in two different ways:
data types in Zarr version 2 are encoded as NumPy-compatible strings, while data types in Zarr version
3 are encoded as either strings or ``JSON`` objects,
and the Zarr V3 data types don't have any associated endianness information, unlike Zarr V2 data types.

To abstract over these syntactical and semantic differences, Zarr Python uses a class called
`ZDType <../api/zarr/dtype/index.html#zarr.dtype.ZDType>`_ provide Zarr V2 and Zarr V3 compatibility
routines for ""native" data types. In this context, a "native" data type is a Python class,
typically defined in another library, that models an array's data type. For example, ``np.uint8`` is a native
data type defined in NumPy, which Zarr Python wraps with a ``ZDType`` instance called
`UInt8 <../api/zarr/dtype/index.html#zarr.dtype.ZDType>`_.

Each data type supported by Zarr Python is modeled by ``ZDType`` subclass, which provides an
API for the following operations:

- Wrapping / unwrapping a native data type
- Encoding / decoding a data type to / from Zarr V2 and Zarr V3 array metadata.
- Encoding / decoding a scalar value to / from Zarr V2 and Zarr V3 array metadata.


Example Usage
~~~~~~~~~~~~~

Create a ``ZDType`` from a native data type:

.. code-block:: python

  >>> from zarr.core.dtype import Int8
  >>> import numpy as np
  >>> int8 = Int8.from_native_dtype(np.dtype('int8'))

Convert back to native data type:

.. code-block:: python

  >>> native_dtype = int8.to_native_dtype()
  >>> assert native_dtype == np.dtype('int8')

Get the default scalar value for the data type:

.. code-block:: python

  >>> default_value = int8.default_scalar()
  >>> assert default_value == np.int8(0)


Serialize to JSON for Zarr V2 and V3

.. code-block:: python

  >>> json_v2 = int8.to_json(zarr_format=2)
  >>> json_v2
  {'name': '|i1', 'object_codec_id': None}
  >>> json_v3 = int8.to_json(zarr_format=3)
  >>> json_v3
  'int8'

Serialize a scalar value to JSON:

.. code-block:: python

  >>> json_value = int8.to_json_scalar(42, zarr_format=3)
  >>> json_value
  42

Deserialize a scalar value from JSON:

.. code-block:: python

  >>> scalar_value = int8.from_json_scalar(42, zarr_format=3)
  >>> assert scalar_value == np.int8(42)

Adding new data types
~~~~~~~~~~~~~~~~~~~~~

Each Zarr data type is a separate Python class that inherits from
`ZDType <../api/zarr/dtype/index.html#zarr.dtype.ZDType>`_. You can define a custom data type by
writing your own subclass of `ZDType <../api/zarr/dtype/index.html#zarr.dtype.ZDType>`_ and adding
your data type to the data type registry. A complete example of this process is included below.

The source code for this example can be found in the ``examples/custom_dtype.py`` file in the Zarr
Python project directory.

.. literalinclude:: ../../examples/custom_dtype.py
  :language: python


Data type resolution
~~~~~~~~~~~~~~~~~~~~

Although Zarr Python uses a different data type model from NumPy, you can still define a Zarr array
with a NumPy data type object:

.. code-block:: python

  >>> from zarr import create_array
  >>> import numpy as np
  >>> a = create_array({}, shape=(10,), dtype=np.dtype('int'))
  >>> a
  <Array memory:... shape=(10,) dtype=int64>

Or a string representation of a NumPy data type:

.. code-block:: python

  >>> a = create_array({}, shape=(10,), dtype='<i8')
  >>> a
  <Array memory:... shape=(10,) dtype=int64>

The ``Array`` object presents itself like a NumPy array, including exposing a NumPy
data type as its ``dtype`` attribute:

.. code-block:: python

  >>> type(a.dtype)
  <class 'numpy.dtypes.Int64DType'>

But if we inspect the metadata for the array, we can see the Zarr data type object:

.. code-block:: python

  >>> type(a.metadata.data_type)
  <class 'zarr.core.dtype.npy.int.Int64'>

This example illustrates a general problem Zarr Python has to solve -- how can we allow users to
specify a data type as a string, or a NumPy ``dtype`` object, and produce the right Zarr data type
from that input? We call this process "data type resolution". Zarr Python also performs data type
resolution when reading stored arrays, although in this case the input is a ``JSON`` value instead
of a NumPy data type.

For simple data types like ``int`` the solution could be extremely simple: just
maintain a lookup table that relates a NumPy data type to the Zarr data type equivalent. But not all
data types are so simple. Consider this case:

.. code-block:: python

  >>> from zarr import create_array
  >>> import numpy as np
  >>> a = create_array({}, shape=(10,), dtype=[('a', np.dtype('float')), ('b', 'i8')])
  >>> a.dtype # this is the NumPy data type
  dtype([('a', '<f8'), ('b', '<i8')])
  >>> a.metadata.data_type # this is the Zarr data type
  Structured(fields=(('a', Float64(endianness='little')), ('b', Int64(endianness='little'))))

In this example, we created a
`NumPy structured data type <https://numpy.org/doc/stable/user/basics.rec.html#structured-datatypes>`_.
This data type is a container that can contain any NumPy data type, which makes it recursive. It is
not possible to make a lookup table that relates all NumPy structured data types to their Zarr
equivalents, as there is a nearly unbounded number of different structured data types. So instead of
a static lookup table, Zarr Python relies on a dynamic approach to data type resolution.

Zarr Python defines a collection of Zarr data types. This collection, called a "data type registry",
is essentially a dict where the keys are strings (a canonical name for each data type), and the values are
the data type classes themselves. Dynamic data type resolution entails iterating over these data
type classes, invoking a special class constructor defined on each one, and returning a concrete
data type instance if and only if exactly 1 of those constructor invocations was successful.

In plain language, we take some user input (a NumPy array), offer it to all the known data type
classes, and return an instance of the one data type class that could accept that user input.

We want to avoid a situation where the same NumPy data type matches multiple Zarr data types. I.e.,
a NumPy data type should uniquely specify a single Zarr data type. But data type resolution is
dynamic, so it's not possible to guarantee this uniqueness constraint. So we attempt data type
resolution against every data type class, and if for some reason a NumPy data type matches multiple
Zarr data types, we treat this as an error and raise an exception.

