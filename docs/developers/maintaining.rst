Maintainer's guide
------------------

The following information is mainly for core developers, but may also be of interest to
contributors.

Merging pull requests
~~~~~~~~~~~~~~~~~~~~~

Pull requests submitted by an external contributor should be reviewed and approved by at least
one core developers before being merged. Ideally, pull requests submitted by a core developer
should be reviewed and approved by at least one other core developers before being merged.

Pull requests should not be merged until all CI checks have passed (GitHub Actions
Codecov) against code that has had the latest main merged in.

Compatibility and versioning policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because Zarr is a data storage library, there are two types of compatibility to
consider: API compatibility and data format compatibility.

API compatibility
"""""""""""""""""

All functions, classes and methods that are included in the API
documentation (files under ``docs/api/*.rst``) are considered as part of the Zarr **public API**,
except if they have been documented as an experimental feature, in which case they are part of
the **experimental API**.

Any change to the public API that does **not** break existing third party
code importing Zarr, or cause third party code to behave in a different way, is a
**backwards-compatible API change**. For example, adding a new function, class or method is usually
a backwards-compatible change. However, removing a function, class or method; removing an argument
to a function or method; adding a required argument to a function or method; or changing the
behaviour of a function or method, are examples of **backwards-incompatible API changes**.

If a release contains no changes to the public API (e.g., contains only bug fixes or
other maintenance work), then the micro version number should be incremented (e.g.,
2.2.0 -> 2.2.1). If a release contains public API changes, but all changes are
backwards-compatible, then the minor version number should be incremented
(e.g., 2.2.1 -> 2.3.0). If a release contains any backwards-incompatible public API changes,
the major version number should be incremented (e.g., 2.3.0 -> 3.0.0).

Backwards-incompatible changes to the experimental API can be included in a minor release,
although this should be minimised if possible. I.e., it would be preferable to save up
backwards-incompatible changes to the experimental API to be included in a major release, and to
stabilise those features at the same time (i.e., move from experimental to public API), rather than
frequently tinkering with the experimental API in minor releases.

Data format compatibility
"""""""""""""""""""""""""

The data format used by Zarr is defined by a specification document, which should be
platform-independent and contain sufficient detail to construct an interoperable
software library to read and/or write Zarr data using any programming language. The
latest version of the specification document is available on the
`Zarr specifications website <https://zarr-specs.readthedocs.io>`_.

Here, **data format compatibility** means that all software libraries that implement a
particular version of the Zarr storage specification are interoperable, in the sense
that data written by any one library can be read by all others. It is obviously
desirable to maintain data format compatibility wherever possible. However, if a change
is needed to the storage specification, and that change would break data format
compatibility in any way, then the storage specification version number should be
incremented (e.g., 2 -> 3).

The versioning of the Zarr software library is related to the versioning of the storage
specification as follows. A particular version of the Zarr library will
implement a particular version of the storage specification. For example, Zarr version
2.2.0 implements the Zarr storage specification version 2. If a release of the Zarr
library implements a different version of the storage specification, then the major
version number of the Zarr library should be incremented. E.g., if Zarr version 2.2.0
implements the storage spec version 2, and the next release of the Zarr library
implements storage spec version 3, then the next library release should have version
number 3.0.0. Note however that the major version number of the Zarr library may not
always correspond to the spec version number. For example, Zarr versions 2.x, 3.x, and
4.x might all implement the same version of the storage spec and thus maintain data
format compatibility, although they will not maintain API compatibility.

When to make a release
~~~~~~~~~~~~~~~~~~~~~~

Ideally, any bug fixes that don't change the public API should be released as soon as
possible. It is fine for a micro release to contain only a single bug fix.

When to make a minor release is at the discretion of the core developers. There are no
hard-and-fast rules, e.g., it is fine to make a minor release to make a single new
feature available; equally, it is fine to make a minor release that includes a number of
changes.

Major releases obviously need to be given careful consideration, and should be done as
infrequently as possible, as they will break existing code and/or affect data
compatibility in some way.

Release procedure
~~~~~~~~~~~~~~~~~

.. note::

   Most of the release process is now handled by GitHub workflow which should
   automatically push a release to PyPI if a tag is pushed.

Before releasing, make sure that all pull requests which will be
included in the release have been properly documented in
`docs/release.rst`.

To make a new release, go to
https://github.com/zarr-developers/zarr-python/releases and
click "Draft a new release". Choose a version number prefixed
with a `v` (e.g. `v0.0.0`). For pre-releases, include the
appropriate suffix (e.g. `v0.0.0a1` or `v0.0.0rc2`).


Set the description of the release to::

    See release notes https://zarr.readthedocs.io/en/stable/release.html#release-0-0-0

replacing the correct version numbers. For pre-release versions,
the URL should omit the pre-release suffix, e.g. "a1" or "rc1".

Click on "Generate release notes" to auto-file the description.

After creating the release, the documentation will be built on
https://readthedocs.io. Full releases will be available under
`/stable <https://zarr.readthedocs.io/en/stable>`_ while
pre-releases will be available under
`/latest <https://zarr.readthedocs.io/en/latest>`_.

Also review and merge the https://github.com/conda-forge/zarr-feedstock
pull request that will be automatically generated.
