Compressors and filters (``zarr.codecs``)
=========================================
.. module:: zarr.codecs

This module contains compressor and filter classes for use with Zarr. Please note that this module
is provided for backwards compatibility with previous versions of Zarr. From Zarr version 2.2
onwards, all codec classes have been moved to a separate package called Numcodecs_. The two
packages (Zarr and Numcodecs_) are designed to be used together. For example, a Numcodecs_ codec
class can be used as a compressor for a Zarr array::

    >>> import zarr
    >>> from numcodecs import Blosc
    >>> z = zarr.zeros(1000000, compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

Codec classes can also be used as filters. See the tutorial section on :ref:`tutorial_filters`
for more information.

Please note that it is also relatively straightforward to define and register custom codec
classes. See the Numcodecs `codec API <http://numcodecs.readthedocs.io/en/latest/abc.html>`_ and
`codec registry <http://numcodecs.readthedocs.io/en/latest/registry.html>`_ documentation for more
information.

.. _Numcodecs: http://numcodecs.readthedocs.io/
