Release notes
=============

Unreleased
----------

New features
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
* Fixes ``order`` argument for Zarr format 2 arrays (:issue:`2679`).
* Backwards compatibility for Zarr format 2 structured arrays (:issue:`2134`)

* Fixes a bug that prevented reading Zarr format 2 data with consolidated metadata written using ``zarr-python`` version 2 (:issue:`2694`).

Behaviour changes
~~~~~~~~~~~~~~~~~

Other
~~~~~
* Removed some unnecessary files from the source distribution
  to reduce its size. (:issue:`2686`)


.. _release_3.0.0:

3.0.0
-----

3.0.0 is a new major release of Zarr-Python, with many breaking changes.
See the :ref:`v3 migration guide` for a listing of what's changed.

Normal release note service will resume with further releases in the 3.0.0
series.

Release notes for the zarr-python 2.x and 1.x releases can be found here:
https://zarr.readthedocs.io/en/support-v2/release.html
