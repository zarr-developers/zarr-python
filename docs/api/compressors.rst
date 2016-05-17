Compressors (``zarr.compressors``)
==================================
.. module:: zarr.compressors

This module contains compressor classes for use with Zarr. Note that normally
there is no need to use these classes directly, they are used under the hood
by Zarr when looking up an implementation of a particular compression.

Other compressors can be registered dynamically with Zarr. All that is required
is to implement a class that provides the same interface as the classes listed
below, and then to add the class to the compressor registry. See the source
code of this module for details.

.. autoclass:: BloscCompressor

.. autoclass:: ZlibCompressor

.. autoclass:: BZ2Compressor

.. autoclass:: LZMACompressor

.. autoclass:: NoCompressor

