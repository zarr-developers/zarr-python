Compressors and filters (``zarr.codecs``)
=========================================
.. module:: zarr.codecs

This module contains compressor and filter classes for use with Zarr.

Other codecs can be registered dynamically with Zarr. All that is required
is to implement a class that provides the same interface as the classes listed
below, and then to add the class to the ``codec_registry``. See the source
code of this module for details.

.. autoclass:: Codec

    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config

.. autoclass:: Blosc
.. autoclass:: Zlib
.. autoclass:: BZ2
.. autoclass:: LZMA
.. autoclass:: Delta
.. autoclass:: FixedScaleOffset
.. autoclass:: Quantize
.. autoclass:: PackBits
.. autoclass:: Categorize
