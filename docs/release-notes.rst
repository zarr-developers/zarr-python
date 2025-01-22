Release notes
=============

.. towncrier release notes start


3.0.1 (Jan. 17, 2025)
---------------------

Bug fixes
~~~~~~~~~
* Fixes ``order`` argument for Zarr format 2 arrays (:issue:`2679`).

* Fixes a bug that prevented reading Zarr format 2 data with consolidated
  metadata written using ``zarr-python`` version 2 (:issue:`2694`).

* Ensure that compressor=None results in no compression when writing Zarr
  format 2 data (:issue:`2708`).

* Fix for empty consolidated metadata dataset: backwards compatibility with
  Zarr-Python 2 (:issue:`2695`).

Documentation
~~~~~~~~~~~~~
* Add v3.0.0 release announcement banner (:issue:`2677`).

* Quickstart guide alignment with V3 API (:issue:`2697`).

* Fix doctest failures related to numcodecs 0.15 (:issue:`2727`).

Other
~~~~~
* Removed some unnecessary files from the source distribution
  to reduce its size. (:issue:`2686`).

* Enable codecov in GitHub actions (:issue:`2682`).

* Speed up hypothesis tests (:issue:`2650`).

* Remove multiple imports for an import name (:issue:`2723`).


.. _release_3.0.0:

3.0.0 (Jan. 9, 2025)
--------------------

3.0.0 is a new major release of Zarr-Python, with many breaking changes.
See the :ref:`v3 migration guide` for a listing of what's changed.

Normal release note service will resume with further releases in the 3.0.0
series.

Release notes for the zarr-python 2.x and 1.x releases can be found here:
https://zarr.readthedocs.io/en/support-v2/release.html
