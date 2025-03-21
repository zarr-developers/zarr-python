.. _dev-guide-contributing:

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
      Python 3.12.7 | packaged by conda-forge | (main, Oct  4 2024, 15:57:01) [Clang 17.0.6 ] on darwin

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
the `"Fork" <https://github.com/zarr-developers/zarr-python/fork>`_ button.
Then clone your fork to your local machine::

    $ git clone git@github.com:your-user-name/zarr-python.git
    $ cd zarr-python
    $ git remote add upstream git@github.com:zarr-developers/zarr-python.git

Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work with the Zarr source code, it is recommended to use
`hatch <https://hatch.pypa.io/latest/index.html>`_ to create and manage development
environments. Hatch will automatically install all Zarr dependencies using the same
versions as are used by the core developers and continuous integration services.
Assuming you have a Python 3 interpreter already installed, and you have cloned the
Zarr source code and your current working directory is the root of the repository,
you can do something like the following::

    $ pip install hatch
    $ hatch env show  # list all available environments

To verify that your development environment is working, you can run the unit tests
for one of the test environments, e.g.::

    $ hatch env run --env test.py3.12-2.1-optional run-pytest

Creating a branch
~~~~~~~~~~~~~~~~~

Before you do any new work or submit a pull request, please open an issue on GitHub to
report the bug or propose the feature you'd like to add.

It's best to synchronize your fork with the upstream repository, then create a
new, separate branch for each piece of work you want to do. E.g.::

    git checkout main
    git fetch upstream
    git checkout -b shiny-new-feature upstream/main
    git push -u origin shiny-new-feature

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in
this branch specific to one bug or feature so it is clear what the branch brings to
Zarr.

To update this branch with latest code from Zarr, you can retrieve the changes from
the main branch and perform a rebase::

    git fetch upstream
    git rebase upstream/main

This will replay your commits on top of the latest Zarr git main. If this leads to
merge conflicts, these need to be resolved before submitting a pull request.
Alternatively, you can merge the changes in from upstream/main instead of rebasing,
which can be simpler::

    git pull upstream main

Again, any conflicts need to be resolved before submitting a pull request.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

Zarr includes a suite of unit tests. The simplest way to run the unit tests
is to activate your development environment
(see `creating a development environment`_ above) and invoke::

    $ hatch env run --env test.py3.12-2.1-optional run-pytest

All tests are automatically run via GitHub Actions for every pull
request and must pass before code can be accepted. Test coverage is
also collected automatically via the Codecov service.

.. note::
    Previous versions of Zarr-Python made extensive use of doctests. These tests were
    not maintained during the 3.0 refactor but may be brought back in the future.
    See :issue:`2614` for more details.

Code standards - using pre-commit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All code must conform to the PEP8 standard. Regarding line length, lines up to 100
characters are allowed, although please try to keep under 90 wherever possible.

``Zarr`` uses a set of ``pre-commit`` hooks and the ``pre-commit`` bot to format,
type-check, and prettify the codebase. ``pre-commit`` can be installed locally by
running::

    $ python -m pip install pre-commit

The hooks can be installed locally by running::

    $ pre-commit install

This would run the checks every time a commit is created locally. These checks will also run
on every commit pushed to an open PR, resulting in some automatic styling fixes by the
``pre-commit`` bot. The checks will by default only run on the files modified by a commit,
but the checks can be triggered for all the files by running::

    $ pre-commit run --all-files

If you would like to skip the failing checks and push the code for further discussion, use
the ``--no-verify`` option with ``git commit``.


Test coverage
~~~~~~~~~~~~~

.. note::
    Test coverage for Zarr-Python 3 is currently not at 100%. This is a known issue and help
    is welcome to bring test coverage back to 100%. See :issue:`2613` for more details.

Zarr strives to maintain 100% test coverage under the latest Python stable release
Both unit tests and docstring doctests are included when computing coverage. Running::

    $ hatch env run --env test.py3.12-2.1-optional run-coverage

will automatically run the test suite with coverage and produce a XML coverage report.
This should be 100% before code can be accepted into the main code base.

You can also generate an HTML coverage report by running::

     $ hatch env run --env test.py3.12-2.1-optional run-coverage-html

When submitting a pull request, coverage will also be collected across all supported
Python versions via the Codecov service, and will be reported back within the pull
request. Codecov coverage must also be 100% before code can be accepted.

Documentation
~~~~~~~~~~~~~

Docstrings for user-facing classes and functions should follow the
`numpydoc
<https://numpydoc.readthedocs.io/en/stable/format.html#docstring-standard>`_
standard, including sections for Parameters and Examples. All examples
should run and pass as doctests under Python 3.11.

Zarr uses Sphinx for documentation, hosted on readthedocs.org. Documentation is
written in the RestructuredText markup language (.rst files) in the ``docs`` folder.
The documentation consists both of prose and API documentation. All user-facing classes
and functions are included in the API documentation, under the ``docs/api`` folder
using the `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
extension to sphinx. Any new features or important usage information should be included in the
user-guide (``docs/user-guide``). Any changes should also be included as a new file in the
:file:`changes` directory.

The documentation can be built locally by running::

    $ hatch --env docs run build

The resulting built documentation will be available in the ``docs/_build/html`` folder.

Hatch can also be used to serve continuously updating version of the documentation
during development at `http://0.0.0.0:8000/ <http://0.0.0.0:8000/>`_. This can be done by running::

    $ hatch --env docs run serve

.. _changelog:

Changelog
~~~~~~~~~

zarr-python uses `towncrier`_ to manage release notes. Most pull requests should
include at least one news fragment describing the changes. To add a release
note, you'll need the GitHub issue or pull request number and the type of your
change (``feature``, ``bugfix``, ``doc``, ``removal``, ``misc``). With that, run
```towncrier create``` with your development environment, which will prompt you
for the issue number, change type, and the news text::

   towncrier create

Alternatively, you can manually create the files in the ``changes`` directory
using the naming convention ``{issue-number}.{change-type}.rst``.

See the `towncrier`_ docs for more.

.. _towncrier: https://towncrier.readthedocs.io/en/stable/tutorial.html

Development best practices, policies and procedures
---------------------------------------------------

The following information is mainly for core developers, but may also be of interest to
contributors.

Merging pull requests
~~~~~~~~~~~~~~~~~~~~~

Pull requests submitted by an external contributor should be reviewed and approved by at least
one core developer before being merged. Ideally, pull requests submitted by a core developer
should be reviewed and approved by at least one other core developer before being merged.

Pull requests should not be merged until all CI checks have passed (GitHub Actions
Codecov) against code that has had the latest main merged in.

Compatibility and versioning policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Versioning
""""""""""
Versions of this library are identified by a triplet of integers with the form
``<major>.<minor>.<patch>``, for example ``3.0.4``. A release of ``zarr-python`` is associated with a new
version identifier. That new identifier is generated by incrementing exactly one of the components of
the previous version identifier by 1. When incrementing the ``major`` component of the version identifier,
the ``minor`` and ``patch`` components is reset to 0. When incrementing the minor component,
the patch component is reset to 0.

Releases are classified by the library changes contained in that release. This classification
determines which component of the version identifier is incremented on release.

* ``major`` releases (for example, ``2.18.0`` -> ``3.0.0``) are for changes that will
  require extensive adaptation efforts from many users and downstream projects.
  For example, breaking changes to widely-used user-facing APIs should only be applied in a major release.


  Users and downstream projects should carefully consider the impact of a major release before
  adopting it.
  In advance of a major release, developers should communicate the scope of the upcoming changes,
  and help users prepare for them.

* ``minor`` releases (or example, ``3.0.0`` -> ``3.1.0``) are for changes that do not require
  significant effort from most users or downstream downstream projects to respond to. API changes
  are possible in minor releases if the burden on users imposed by those changes is sufficiently small.

  For example, a recently released API may need fixes or refinements that are breaking, but low impact
  due to the recency of the feature. Such API changes are permitted in a minor release.


  Minor releases are safe for most users and downstream projects to adopt.


* ``patch`` releases (for example, ``3.1.0`` -> ``3.1.1``) are for changes that contain no breaking
  or behaviour changes for downstream projects or users. Examples of changes suitable for a patch release are
  bugfixes and documentation improvements.


  Users should always feel safe upgrading to a the latest patch release.

Note that this versioning scheme is not consistent with `Semantic Versioning <https://semver.org/>`_.
Contrary to SemVer, the Zarr library may release breaking changes in ``minor`` releases, or even
``patch`` releases under exceptional circumstances. But we should strive to avoid doing so.

A better model for our versioning scheme is `Intended Effort Versioning <https://jacobtomlinson.dev/effver/>`_,
or "EffVer". The guiding principle off EffVer is to categorize releases based on the *expected effort
required to upgrade to that release*.

Zarr developers should make changes as smooth as possible for users. This means making
backwards-compatible changes wherever possible. When a backwards-incompatible change is necessary,
users should be notified well in advance, e.g. via informative deprecation warnings.

Data format compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^

The Zarr library is an implementation of a file format standard defined externally --
see the `Zarr specifications website <https://zarr-specs.readthedocs.io>`_ for the list of
Zarr file format specifications.


If an existing Zarr format version changes, or a new version of the Zarr format is released, then
the Zarr library will generally require changes. It is very likely that a new Zarr format will
require extensive breaking changes to the Zarr library, and so support for a new Zarr format in the
Zarr library will almost certainly come in new ``major`` release.
When the Zarr library adds support for a new Zarr format, there may be a period of accelerated
changes as developers refine newly added APIs and deprecate old APIs. In such a transitional phase
breaking changes may be more frequent than usual.


Release procedure
~~~~~~~~~~~~~~~~~

.. note::

   Most of the release process is now handled by GitHub workflow which should
   automatically push a release to PyPI if a tag is pushed.

Pre-release
"""""""""""
1. Make sure that all pull requests which will be included in the release
   have been properly documented as changelog files in :file:`changes`.
2. Run ``towncrier build --version x.y.z`` to create the changelog.

Releasing
"""""""""
To make a new release, go to
https://github.com/zarr-developers/zarr-python/releases and
click "Draft a new release". Choose a version number prefixed
with a `v` (e.g. `v0.0.0`). For pre-releases, include the
appropriate suffix (e.g. `v0.0.0a1` or `v0.0.0rc2`).


Set the description of the release to::

    See release notes https://zarr.readthedocs.io/en/stable/release-notes.html#release-0-0-0

replacing the correct version numbers. For pre-release versions,
the URL should omit the pre-release suffix, e.g. "a1" or "rc1".

Click on "Generate release notes" to auto-file the description.

After creating the release, the documentation will be built on
https://readthedocs.io. Full releases will be available under
`/stable <https://zarr.readthedocs.io/en/stable>`_ while
pre-releases will be available under
`/latest <https://zarr.readthedocs.io/en/latest>`_.

Post-release
""""""""""""

- Review and merge the pull request on the
  `conda-forge feedstock <https://github.com/conda-forge/zarr-feedstock>`_ that will be
  automatically generated.
- Create a new "Unreleased" section in the release notes
