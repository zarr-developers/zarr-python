Compressors and filters (``zarr.codecs``)
=========================================
.. module:: zarr.codecs

This module contains compressor and filter classes for use with Zarr.

Other codecs can be registered dynamically with Zarr. All that is required
is to implement a class that provides the same interface as the classes listed
below, and then to add the class to the ``codec_registry``. See the source
code of this module for details.

.. autoclass:: BloscCompressor
.. autoclass:: ZlibCompressor
.. autoclass:: BZ2Compressor
.. autoclass:: LZMACompressor
.. autoclass:: DeltaFilter
.. autoclass:: FixedScaleOffsetFilter
.. autoclass:: QuantizeFilter
.. autoclass:: PackBitsFilter
.. autoclass:: CategorizeFilter
