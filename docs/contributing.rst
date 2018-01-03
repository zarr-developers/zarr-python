Contributing to Zarr
====================

Zarr is a community maintained project. We welcome contributions in the form of bug
reports, bug fixes, documentation, enhancement proposals and more. This page provides
information on how best to contribute.

Asking for help
---------------

If you have a question about how to use Zarr, please post your question on
StackOverflow using the `"zarr" tag <https://stackoverflow.com/questions/tagged/zarr>`_
. If you don't get a response within a day or two, feel free to raise a `GitHub issue
<https://github.com/alimanfoo/zarr/issues/new>`_ including a link to your StackOverflow
question. We will try to respond to questions as quickly as possible, but please bear
in mind that there may be periods where we have limited time to answer questions
due to other commitments.

Bug reports
-----------

If you find a bug, please raise a `GitHub issue
<https://github.com/alimanfoo/zarr/issues/new>`_. Please include the following items in
a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can
   format the code nicely using markdown, e.g.::


    ```python
    >>> import zarr
    >>> g = zarr.group()
    ...
    ```

2. Information about the version of Zarr, along with versions of dependencies and the
   Python interpreter, and installation information. The version of Zarr can be obtained
   from the ``zarr.__version__`` property. Please also state how Zarr was installed,
   e.g., "installed via pip into a virtual environment", or "installed using conda".
   Information about other packages installed can be obtained by executing ``pip list``
   (if using pip to install packages) or ``conda list`` (if using conda to install
   packages) from the operating system command prompt. The version of the Python
   interpreter can be obtained by running a Python interactive session, e.g.::

    $ python
    Python 3.6.1 (default, Mar 22 2017, 06:17:05)
    [GCC 6.3.0 20170321] on linux

3. An explanation of why the current behaviour is wrong/not desired, and what you
   expect instead.

Enhancement proposals
---------------------

If you have an idea about a new feature or some other improvement to Zarr, please raise a
`GitHub issue <https://github.com/alimanfoo/zarr/issues/new>`_ first to discuss.

We very much welcome ideas and suggestions for how to improve Zarr, but please bear in
mind that we are likely to be conservative in accepting proposals for new features. The
reasons for this are that we would like to keep the Zarr code base lean and focused on
a core set of functionalities, and available time for development, review and maintenance
of new features is limited. But if you have a great idea, please don't let that stop
you posting it on GitHub, just please don't be offended if we respond cautiously.

Working with the code
---------------------

Forking
~~~~~~~

The Zarr source code is hosted on GitHub at the following location:

* `https://github.com/zarr-developers/zarr <https://github.com/zarr-developers/zarr>`_

You will need your own fork to work on the code. Go to the link above and hit
the "Fork" button. Then clone your fork to your local machine::

    $ git clone git@github.com:your-user-name/zarr.git
    $ cd zarr
    $ git remote add upstream git@github.com:zarr-developers/zarr.git

Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work with the Zarr source code, it is recommended to set up a Python virtual
environment and install all Zarr dependencies using the same versions as are used by
the core developers and continuous integration services. Assuming you have a Python
3 interpreter already installed, and have also installed the virtualenv package, and
you have cloned the Zarr source code and your current working directory is the root of
the repository, you can do something like the following::

    $ mkdir -p ~/pyenv/zarr-dev
    $ virtualenv --no-site-packages --python=/usr/bin/python3.6 ~/pyenv/zarr-dev
    $ source ~/pyenv/zarr-dev/bin/activate
    $ pip install -r requirements_dev.txt

To verify that your development environment is working, you can run the unit test suite::

    $ pytest -v zarr


