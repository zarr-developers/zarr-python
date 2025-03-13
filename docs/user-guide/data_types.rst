Data types
==========

Zarr's data type model
----------------------

Every Zarr array has a "data type", which defines the meaning and physical layout of the
array's elements. Zarr is heavily influenced by `NumPy <https://numpy.org/doc/stable/>`_, and
Zarr arrays can use many of the same data types as numpy arrays::
    >>> import zarr
    >>> import numpy as np
    >>> zarr.create_array(store={}, shape=(10,), dtype=np.dtype('uint8'))
    >>> z
    <Array memory://126225407345920 shape=(10,) dtype=uint8>

But Zarr data types and Numpy data types are also very different in one key respect:
Zarr arrays are designed to be persisted to storage and later read, possibly by Zarr implementations in different programming languages.
So in addition to defining a memory layout for array elements, each Zarr data type defines a procedure for
reading and writing that data type to Zarr array metadata, and also reading and writing **instances** of that data type to
array metadata.

Data types in Zarr version 2
-----------------------------

Version 2 of the Zarr format defined its data types relative to `Numpy's data types <https://numpy.org/doc/2.1/reference/arrays.dtypes.html#data-type-objects-dtype>`_, and added a few non-Numpy data types as well.
Thus the JSON identifer for a Numpy-compatible data type is just the Numpy ``str`` attribute of that dtype:

    >>> import zarr
    >>> import numpy as np
    >>> import json
    >>> np_dtype = np.dtype('int64')
    >>> z = zarr.create_array(shape=(1,), dtype=np_dtype, zarr_format=2)
    >>> dtype_meta = json.loads(store['.zarray'].to_bytes())["dtype"]
    >>> assert dtype_meta == np_dtype.str # True
    >>> dtype_meta
    <i8

.. note::
    The ``<`` character in the data type metadata encodes the `endianness https://numpy.org/doc/2.2/reference/generated/numpy.dtype.byteorder.html`_, or "byte order", of the data type. Following Numpy's example,
Zarr version 2 data types associate each data type with an endianness where applicable. Zarr version 3 data types do not store endianness information.

In addition to defining a representation of the data type itself (which in the example above was just a simple string ``"<i8"``, Zarr also
defines a metadata representation of scalars associated with that data type. Integers are stored as ``JSON`` numbers,
as are floats, with the caveat that `NaN`, positive infinity, and negative infinity are stored as special strings.

Data types in Zarr version 3
----------------------------

* No endianness
* Data type can be encoded as a string or a ``JSON`` object with the structure ``{"name": <string identifier>, "configuration": {...}}``

Data types in Zarr-Python
-------------------------

Zarr-Python supports two different Zarr formats, and those two formats specify data types in rather different ways:
data types in Zarr version 2 are encoded as Numpy-compatible strings, while data types in Zarr version 3 are encoded as either strings or ``JSON`` objects,
and the Zarr V3 data types don't have any associated endianness information, unlike Zarr V2 data types.

If that wasn't enough, we want Zarr-Python to support data types beyond what's available in Numpy. So it's crucial that we have a
model of array data types that can adapt to the differences between Zarr V2 and V3 and doesn't over-fit to Numpy.

Here are the operations we need to perform on data types in Zarr-Python:

* Round-trip native data types to fields in array metadata documents.
    For example, the Numpy data type ``np.dtype('>i2')`` should be saved as ``{..., "dtype" : ">i2"}`` in Zarr V2 metadata.

    In Zarr V3 metadata, the same Numpy data type would be saved as  ``{..., "data_type": "int16", "codecs": [..., {"name": "bytes", "configuration": {"endian": "big"}, ...]}``

* Define a default fill value. This is not mandated by the Zarr specifications, but it's convenient for users
  to have a useful default. For numeric types like integers and floats the default can be statically set to 0, but for
  parametric data types like fixed-length strings the default can only be generated after the data type has been parametrized at runtime.

* Round-trip scalars to the ``fill_value`` field in Zarr V2 and V3 array metadata documents. The Zarr V2 and V3 specifications
  define how scalars of each data type should be stored as JSON in array metadata documents, and in principle each data type
  can define this encoding separately.

* Do all of the above for *user-defined data types*. Zarr-Python should support data types added as extensions,so we cannot
  hard-code the list of data types. We need to ensure that users can easily (or easily enough) define a python object
  that models their custom data type and register this object with Zarr-Python, so that the above operations all succeed for their
  custom data type.

To achieve these goals, Zarr Python uses a class called :class:`zarr.core.dtype.DTypeWrapper` to wrap native data types. Each data type
supported by Zarr Python is modeled by a subclass of `DTypeWrapper`, which has the following structure:

(attribute) ``dtype_cls``
^^^^^^^^^^^^^
The ``dtype_cls`` attribute is a **class variable** that is bound to a class that can produce
an instance of a native data type. For example, on the ``DTypeWrapper`` used to model the boolean
data type, the ``dtype_cls`` attribute is bound to the numpy bool data type class: ``np.dtypes.BoolDType``.
This attribute is used when we need to create an instance of the native data type, for example when
defining a Numpy array that will contain Zarr data.

It might seem odd that ``DTypeWrapper.dtype_cls`` binds to a *class* that produces a native data type instead of an instance of that native data type --
why not have a ``DTypeWrapper.dtype`` attribute that binds to ``np.dtypes.BoolDType()``? The reason why ``DTypeWrapper``
doesn't wrap a concrete data type instance is because data type instances may have endianness information, but Zarr V3
data types do not. To model Zarr V3 data types, we need endianness to be an **instance variable** which is
defined when creating an instance of the ```DTypeWrapper``. Subclasses of ``DTypeWrapper`` that model data types with
byte order semantics thus have ``endianness`` as an instance variable, and this value can be set when creating an instance of the wrapper.


(attribute) ``_zarr_v3_name``
^^^^^^^^^^^^^
The ``_zarr_v3_name`` attribute encodes the canonical name for a data type for Zarr V3. For many data types these names
are defined in the `Zarr V3 specification https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#data-types`_ For nearly all of the
data types defined in Zarr V3, this name can be used to uniquely specify a data type. The one exception is the ``r*`` data type,
which is parametrized by a number of bits, and so may take the form ``r8``, ``r16``, ... etc.

(class method) ``from_dtype(cls, dtype) -> Self``
^^^^^^^^^
This method defines a procedure for safely converting a native dtype instance into an instance of ``DTypeWrapper``. It should perform
validation of its input to ensure that the native dtype is an instance of the ``dtype_cls`` class attribute, for example. For some
data types, additional checks are needed -- in Numpy "structured" data types and "void" data types use the same class, with different properties.
A ``DTypeWrapper`` that wraps Numpy structured data types must do additional checks to ensure that the input ``dtype`` is actually a structured data type.
If input validation succeeds, this method will call ``_from_dtype_unsafe``.

(class method) ``_from_dtype_unsafe(cls, dtype) -> Self``
^^^^^^^^^^
This method defines the procedure for converting a native data type instance, like ``np.dtype('uint8')``,
into a wrapper class instance. The ``unsafe`` prefix on the method name denotes that this method should not
perform any input validation. Input validation should be done by the routine that calls this method.

For many data types, creating the wrapper class takes no arguments and so this method can just return ``cls()``.
But for data types with runtime attributes like endianness or length (for fixed-size strings), this ``_from_dtype_unsafe``
ensures that those attributes of ``dtype`` are mapped on to the correct parameters in the ``DTypeWrapper`` class constructor.

(method) ``to_dtype(self) -> dtype``
^^^^^^^
This method produces a native data type consistent with the properties of the ``DTypeWrapper``. Together
with ``from_dtype``, this method allows round-trip conversion of a native data type in to a wrapper class and then out again.

That is, for some ``DTypeWrapper`` class ``FooWrapper`` that wraps a native data type called ``foo``, ``FooWrapper.from_dtype(instance_of_foo).to_dtype() == instance_of_foo`` should be true.

(method) ``to_dict(self) -> dict``
^^^^^
This method generates a JSON-serialiazable representation of the wrapped data type which can be stored in
Zarr metadata.

(method) ``cast_value(self, value: object) -> scalar``
^^^^^
Cast a python object to an instance of the wrapped data type. This is used for generating the default
value associated with this data type.


(method) ``default_value(self) -> scalar``
^^^^
Return the default value for the wrapped data type. Zarr-Python uses this method to generate a default fill value
for an array when a user has not requested one.

Why is this a method and not a static attribute? Although some data types
can have a static default value, parametrized data types like fixed-length strings or structured data types cannot. For these data types,
a default value must be calculated based on the attributes of the wrapped data type.

(method) ``check_dtype(cls, dtype)``



