**Breaking:** `cast_value` and `scale_offset` refuse the data types their
specs do not list. Each spec names the integer and floating-point types as
"the following data types defined in this repository" that it supports, and
that repository also defines the numpy time types and the complex types. So
both codecs now refuse `numpy.datetime64` and `numpy.timedelta64`, and
`cast_value` refuses them as the data type it casts to, and `scale_offset`
now refuses `complex64` and `complex128`. All of these were accepted. The time
types' own spec says they are "compatible with any codec that supports arrays
of signed 64-bit integers", which the two codecs' specs do not bear out.
`zarr_metadata.testing.strategies` no longer draws these combinations.
