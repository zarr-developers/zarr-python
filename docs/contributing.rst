Contributing to Zarr
====================

Zarr is a community maintained project. We welcome contributions in the form of bug
reports, bug fixes, documentation, enhancement proposals and more. This page provides
information on how best to contribute.

Asking for help
---------------

If you have a question about how to use Zarr, please post your question on
StackOverflow using the `"zarr" tag <https://stackoverflow.com/questions/tagged/zarr>`_.
If you don't get a response within a day or two, feel free to raise a `GitHub issue
<https://github.com/zarr-developers/zarr-python/issues/new>`_ including a link to your StackOverflow
question. We will try to respond to questions as quickly as possible, but please bear
in mind that there may be periods where we have limited time to answer questions
due to other commitments.

Bug reports
-----------

If you find a bug, please raise a `GitHub issue
<https://github.com/zarr-developers/zarr-python/issues/new>`_. Please include the following items in
a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can
   format the code nicely using markdown, e.g.::


    ```python
    import zarr
    g = zarr.group()
    # etc.
    ```

2. An explanation of why the current behaviour is wrong/not desired, and what you
   expect instead.

3. Information about the version of Zarr, along with versions of dependencies and the
   Python interpreter, and installation information. The version of Zarr can be obtained
   from the ``zarr.__version__`` property. Please also state how Zarr was installed,
   e.g., "installed via pip into a virtual environment", or "installed using conda".
   Information about other packages installed can be obtained by executing ``pip freeze``
   (if using pip to install packages) or ``conda env export`` (if using conda to install
   packages) from the operating system command prompt. The version of the Python
   interpreter can be obtained by running a Python interactive session, e.g.::

    $ python
    Python 3.6.1 (default, Mar 22 2017, 06:17:05)
    [GCC 6.3.0 20170321] on linux

Enhancement proposals
---------------------

If you have an idea about a new feature or some other improvement to Zarr, please raise a
`GitHub issue <https://github.com/zarr-developers/zarr-python/issues/new>`_ first to discuss.

We very much welcome ideas and suggestions for how to improve Zarr, but please bear in
mind that we are likely to be conservative in accepting proposals for new features. The
reasons for this are that we would like to keep the Zarr code base lean and focused on
a core set of functionalities, and available time for development, review and maintenance
of new features is limited. But if you have a great idea, please don't let that stop
you from posting it on GitHub, just please don't be offended if we respond cautiously.

Contributing code and/or documentation
--------------------------------------

Forking the repository
~~~~~~~~~~~~~~~~~~~~~~

The Zarr source code is hosted on GitHub at the following location:

* `https://github.com/zarr-developers/zarr-python <https://github.com/zarr-developers/zarr-python>`_

You will need your own fork to work on the code. Go to the link above and hit
the "Fork" button. Then clone your fork to your local machine::

    $ git clone git@github.com:your-user-name/zarr.git
    $ cd zarr
    $ git remote add upstream git@github.com:zarr-developers/zarr-python.git

Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work with the Zarr source code, it is recommended to set up a Python virtual
environment and install all Zarr dependencies using the same versions as are used by
the core developers and continuous integration services. Assuming you have a Python
3 interpreter already installed, and have also installed the virtualenv package, and
you have cloned the Zarr source code and your current working directory is the root of
the repository, you can do something like the following::

    $ mkdir -p ~/pyenv/zarr-dev
    $ virtualenv --no-site-packages --python=/usr/bin/python3.8 ~/pyenv/zarr-dev
    $ source ~/pyenv/zarr-dev/bin/activate
    $ pip install -r requirements_dev_minimal.txt -r requirements_dev_numpy.txt
    $ pip install -e .

To verify that your development environment is working, you can run the unit tests::

    $ pytest -v zarr

Creating a branch
~~~~~~~~~~~~~~~~~

Before you do any new work or submit a pull request, please open an issue on GitHub to
report the bug or propose the feature you'd like to add.

It's best to synchronize your fork with the upstream repository, then create a
new, separate branch for each piece of work you want to do. E.g.::

    git checkout master
    git fetch upstream
    git rebase upstream/master
    git push
    git checkout -b shiny-new-feature
    git push -u origin shiny-new-feature

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in
this branch specific to one bug or feature so it is clear what the branch brings to
Zarr.

To update this branch with latest code from Zarr, you can retrieve the changes from
the master branch and perform a rebase::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest Zarr git master. If this leads to
merge conflicts, these need to be resolved before submitting a pull request.
Alternatively, you can merge the changes in from upstream/master instead of rebasing,
which can be simpler::

    git fetch upstream
    git merge upstream/master

Again, any conflicts need to be resolved before submitting a pull request.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

Zarr includes a suite of unit tests, as well as doctests included in
function and class docstrings and in the tutorial and storage
spec. The simplest way to run the unit tests is to activate your
development environment (see `creating a development environment`_ above)
and invoke::

    $ pytest -v zarr

Some tests require optional dependencies to be installed, otherwise
the tests will be skipped. To install all optional dependencies, run::

    $ pip install -r requirements_dev_optional.txt

To also run the doctests within docstrings (requires optional
depencies to be installed), run::

    $ pytest -v --doctest-plus zarr

To run the doctests within the tutorial and storage spec (requires
optional dependencies to be installed), run::

    $ python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS docs/tutorial.rst docs/spec/v2.rst

Note that some tests also require storage services to be running
locally. To run the Azure Blob Service storage tests, run an Azure
storage emulator (e.g., azurite) and set the environment variable
``ZARR_TEST_ABS=1``. If you're using Docker to run azurite, start the service with::

    docker run --rm -p 10000:10000 mcr.microsoft.com/azure-storage/azurite azurite-blob --loose --blobHost 0.0.0.0 

To run the Mongo DB storage tests, run a Mongo
server locally and set the environment variable ``ZARR_TEST_MONGO=1``.
To run the Redis storage tests, run a Redis server locally on port
6379 and set the environment variable ``ZARR_TEST_REDIS=1``.

All tests are automatically run via GitHub Actions for every pull
request and must pass before code can be accepted. Test coverage is
also collected automatically via the Codecov service, and total
coverage over all builds must be 100% (although individual builds
may be lower due to Python 2/3 or other differences).

Code standards
~~~~~~~~~~~~~~

All code must conform to the PEP8 standard. Regarding line length, lines up to 100
characters are allowed, although please try to keep under 90 wherever possible.
Conformance can be checked by running::

    $ flake8 --max-line-length=100 zarr

Test coverage
~~~~~~~~~~~~~

Zarr maintains 100% test coverage under the latest Python stable release (currently
Python 3.8). Both unit tests and docstring doctests are included when computing
coverage. Running ``tox -e py38`` will automatically run the test suite with coverage
and produce a coverage report. This should be 100% before code can be accepted into the
main code base.

When submitting a pull request, coverage will also be collected across all supported
Python versions via the Codecov service, and will be reported back within the pull
request. Codecov coverage must also be 100% before code can be accepted.

Documentation
~~~~~~~~~~~~~

Docstrings for user-facing classes and functions should follow the
`numpydoc
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
standard, including sections for Parameters and Examples. All examples
should run and pass as doctests under Python 3.8. To run doctests,
activate your development environment, install optional requirements,
and run::

    $ pytest -v --doctest-plus zarr

Zarr uses Sphinx for documentation, hosted on readthedocs.org. Documentation is
written in the RestructuredText markup language (.rst files) in the ``docs`` folder.
The documentation consists both of prose and API documentation. All user-facing classes
and functions should be included in the API documentation, under the ``docs/api``
folder. Any new features or important usage information should be included in the
tutorial (``docs/tutorial.rst``). Any changes should also be included in the release
notes (``docs/release.rst``).

The documentation can be built locally by running::

    $ tox -e docs

The resulting built documentation will be available in the ``.tox/docs/tmp/html`` folder.

Development best practices, policies and procedures
---------------------------------------------------

The following information is mainly for core developers, but may also be of interest to
contributors.

Merging pull requests
~~~~~~~~~~~~~~~~~~~~~

Pull requests submitted by an external contributor should be reviewed and approved by at least
one core developers before being merged. Ideally, pull requests submitted by a core developer
should be reviewed and approved by at least one other core developers before being merged.

Pull requests should not be merged until all CI checks have passed (GitHub Actions
Codecov) against code that has had the latest master merged in.

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
latest version of the specification document is available from the :ref:`spec` page.

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
format compatibility, although they will not maintain API compatibility. The version number
of the storage specification that is currently implemented is stored under the
``zarr.meta.ZARR_FORMAT`` variable.

Note that the Zarr test suite includes a data fixture and tests to try and ensure that
data format compatibility is not accidentally broken. See the
:func:`test_format_compatibility` function in the :mod:`zarr.tests.test_storage` module
for details.

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

   Most of the release process is now handled by github workflow which should
   automatically push a release to PyPI if a tag is pushed. 

Checkout and update the master branch::

    $ git checkout master
    $ git pull

Verify all tests pass on all supported Python versions, and docs build::

    $ tox

Tag the version (where "X.X.X" stands for the version number, e.g., "2.2.0")::

    $ version=X.X.X
    $ git tag -a v$version -m v$version
    $ git push --tags

Release source code to PyPI::

    $ twine upload dist/zarr-${version}.tar.gz

Obtain checksum for release to conda-forge::

    $ openssl sha256 dist/zarr-${version}.tar.gz

Release to conda-forge by making a pull request against the zarr-feedstock conda-forge
repository, incrementing the version number.
