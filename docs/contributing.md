# Contributing

Zarr is a community maintained project. We welcome contributions in the form of bug reports, bug fixes, documentation, enhancement proposals and more. This page provides information on how best to contribute.

## Asking for help

If you have a question about how to use Zarr, please post your question on StackOverflow using the ["zarr" tag](https://stackoverflow.com/questions/tagged/zarr). If you don't get a response within a day or two, feel free to raise a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new) including a link to your StackOverflow question. We will try to respond to questions as quickly as possible, but please bear in mind that there may be periods where we have limited time to answer questions due to other commitments.

## Bug reports

If you find a bug, please raise a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new). Please include the following items in a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can format the code nicely using markdown, e.g.:

```python exec="false" reason="illustrative pseudocode with a '# etc.' placeholder, not runnable"
import zarr
g = zarr.group()
# etc.
```

2. An explanation of why the current behaviour is wrong/not desired, and what you expect instead.

3. Information about the version of Zarr, along with versions of dependencies and the Python interpreter, and installation information. The version of Zarr can be obtained from the `zarr.__version__` property. Please also state how Zarr was installed, e.g., "installed via pip into a virtual environment", or "installed using conda". Information about other packages installed can be obtained by executing `pip freeze` (if using pip to install packages) or `conda env export` (if using conda to install packages) from the operating system command prompt. The version of the Python interpreter can be obtained by running a Python interactive session, e.g.:

```console
python
```

```ansi
Python 3.12.7 | packaged by conda-forge | (main, Oct  4 2024, 15:57:01) [Clang 17.0.6 ] on darwin
```

## Enhancement proposals

If you have an idea about a new feature or some other improvement to Zarr, please raise a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new) first to discuss.

We very much welcome ideas and suggestions for how to improve Zarr, but please bear in mind that we are likely to be conservative in accepting proposals for new features. The reasons for this are that we would like to keep the Zarr code base lean and focused on a core set of functionalities, and available time for development, review and maintenance of new features is limited. But if you have a great idea, please don't let that stop you from posting it on GitHub, just please don't be offended if we respond cautiously.

## AI-assisted contributions

AI coding tools are increasingly common in open source development. These tools are welcome in Zarr-Python, but the same standards apply to all contributions regardless of how they were produced — whether written by hand, with AI assistance, or generated entirely by an AI tool.

### You are responsible for your changes

If you submit a pull request, you are responsible for understanding and having fully reviewed the changes. You must be able to explain why each change is correct and how it fits into the project.

### Communication must be your own

PR descriptions, issue comments, and review responses must be in your own words. The substance and reasoning must come from you. Using AI to polish grammar or phrasing is fine, but do not paste AI-generated text as comments or review responses.

### Review every line

You must have personally reviewed and understood all changes before submitting. If you used AI to generate code, you are expected to have read it critically and tested it. The PR description should explain the approach and reasoning — do not leave it to reviewers to figure out what the code does and why.

### Keep PRs reviewable

Generating code with AI is fast; reviewing it is not. A large diff shifts the burden from the contributor to the reviewer. PRs that cannot be reviewed in reasonable time with reasonable effort may be closed, regardless of their potential usefulness or correctness. Use AI tools not only to write code but to prepare better, more reviewable PRs — well-structured commits, clear descriptions, and minimal scope.

If you are planning a large AI-assisted contribution (e.g., a significant refactor or a new subsystem), **open an issue first** to discuss the scope and approach with maintainers. Maintainers may also request that large changes be broken into smaller, reviewable pieces.

### Documentation

The same principles apply to documentation. Zarr has domain-specific semantics (chunked storage, codec pipelines, Zarr v2/v3 format details) that AI tools frequently get wrong. Do not submit documentation that you haven't carefully read and verified.

## Contributing code and/or documentation

### Forking the repository

The Zarr source code is hosted on GitHub at the following location:

* [https://github.com/zarr-developers/zarr-python](https://github.com/zarr-developers/zarr-python)

You will need your own fork to work on the code. Go to the link above and hit the ["Fork"](https://github.com/zarr-developers/zarr-python/fork) button. Then clone your fork to your local machine:

```bash
git clone git@github.com:your-user-name/zarr-python.git
cd zarr-python
git remote add upstream git@github.com:zarr-developers/zarr-python.git
```

### Creating a development environment

To work with the Zarr source code, it is recommended to use [hatch](https://hatch.pypa.io/latest/index.html) to create and manage development environments. Hatch will automatically install all Zarr dependencies using the same versions as are used by the core developers and continuous integration services. Assuming you have a Python 3 interpreter already installed, and you have cloned the Zarr source code and your current working directory is the root of the repository, you can do something like the following:

```bash
pip install hatch
hatch env show  # list all available environments
```

To verify that your development environment is working, you can run the unit tests for one of the test environments, e.g.:

```bash
hatch env run --env test.py3.12-optional run
```

### Creating a branch

Before you do any new work or submit a pull request, please open an issue on GitHub to report the bug or propose the feature you'd like to add.

It's best to synchronize your fork with the upstream repository, then create a new, separate branch for each piece of work you want to do. E.g.:

```bash
git checkout main
git fetch upstream
git checkout -b shiny-new-feature upstream/main
git push -u origin shiny-new-feature
```

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to Zarr.

To update this branch with latest code from Zarr, you can retrieve the changes from the main branch and perform a rebase:

```bash
git fetch upstream
git rebase upstream/main
```

This will replay your commits on top of the latest Zarr git main. If this leads to merge conflicts, these need to be resolved before submitting a pull request. Alternatively, you can merge the changes in from upstream/main instead of rebasing, which can be simpler:

```bash
git pull upstream main
```

Again, any conflicts need to be resolved before submitting a pull request.

### Running the test suite

Zarr includes a suite of unit tests. The simplest way to run the unit tests is to activate your development environment (see [creating a development environment](#creating-a-development-environment) above) and invoke:

```bash
hatch env run --env test.py3.12-optional run
```

All tests are automatically run via GitHub Actions for every pull request and must pass before code can be accepted. Test coverage is also collected automatically via the Codecov service.

> **Note:** Previous versions of Zarr-Python made extensive use of doctests. These tests were not maintained during the 3.0 refactor but may be brought back in the future. See issue #2614 for more details.

### Code standards - using prek

All code must conform to the PEP8 standard. Regarding line length, lines up to 100 characters are allowed, although please try to keep under 90 wherever possible.

`Zarr` uses a set of git hooks managed by [`prek`](https://github.com/j178/prek), a fast, Rust-based pre-commit hook manager that is fully compatible with `.pre-commit-config.yaml` files. `prek` can be installed locally by running:

```bash
uv tool install prek
```

or:

```bash
pip install prek
```

The hooks can be installed locally by running:

```bash
prek install
```

This will run the checks every time a commit is created locally. The checks will by default only run on the files modified by a commit, but the checks can be triggered for all the files by running:

```bash
prek run --all-files
```

You can also run hooks only for files in a specific directory:

```bash
prek run --directory src/zarr
```

Or run hooks for files changed in the last commit:

```bash
prek run --last-commit
```

To list all available hooks:

```bash
prek list
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

### Test coverage

> **Note:** Test coverage for Zarr-Python 3 is currently not at 100%. This is a known issue and help is welcome to bring test coverage back to 100%. See issue #2613 for more details.

Zarr strives to maintain 100% test coverage under the latest Python stable release. Both unit tests and docstring doctests are included when computing coverage. Running:

```bash
hatch env run --env test.py3.12-optional run-coverage
```

will automatically run the test suite with coverage and produce an XML coverage report. This should be 100% before code can be accepted into the main code base.

You can also generate an HTML coverage report by running:

```bash
hatch env run --env test.py3.12-optional run-coverage-html
```

When submitting a pull request, coverage will also be collected across all supported Python versions via the Codecov service, and will be reported back within the pull request. Codecov coverage must also be 100% before code can be accepted.

### Documentation

Docstrings for user-facing classes and functions should follow the [numpydoc](https://numpydoc.readthedocs.io/en/stable/format.html#docstring-standard) standard, including sections for Parameters and Examples. All examples should run and pass as doctests under Python 3.12.

Zarr uses mkdocs for documentation, hosted on readthedocs.org. Documentation is written in the Markdown markup language (.md files) in the `docs` folder. The documentation consists both of prose and API documentation. All user-facing classes and functions are included in the API documentation, under the `docs/api` folder using the [mkdocstrings](https://mkdocstrings.github.io/) extension. Add any new public functions or classes to the relevant markdown file in `docs/api/*.md`. Any new features or important usage information should be included in the user-guide (`docs/user-guide`). Any changes should also be included as a new file in the `changes` directory.

The documentation can be built locally by running:

```bash
hatch --env docs run build
```

The resulting built documentation will be available in the `docs/_build/html` folder.

Hatch can also be used to serve continuously updating version of the documentation during development at [http://0.0.0.0:8000/](http://0.0.0.0:8000/). This can be done by running:

```bash
hatch --env docs run serve
```

#### Adding executable code blocks in the documentation

Zarr uses [Markdown Exec](https://pawamoy.github.io/markdown-exec/usage/) to execute code blocks in Markdown files. Add `exec="true"` to a code block header for it to be executed when the docs are built. For example:

````md
```python exec="true"
print("Hello world")
```
````

Below are other useful options that can be added to the code block. See [Markdown Exec's documentation](https://pawamoy.github.io/markdown-exec/usage/#options-summary) for a full list:

  - `source="above"` makes sure the code within the code block is also rendered in the documentation (rather than just the output).
  - `session="<name-of-docs-page>"` executes code blocks in a named session reusing previously defined variables.
  - `result="ansi"` or `result="html"` to render the output. If the code does not produce output, you should leave off the `result` option to prevent an empty cell from rendering in the docs.

For example:

````md
```python exec="true" session="contributing" source="above" result="ansi"
print("Hello world")
```
````

renders as:

```python exec="true" session="contributing" source="above" result="ansi"
print("Hello world")
```

#### Validating code blocks: `exec` vs `test`

Every Python code block in the documentation is checked by a test
(`tests/test_docs.py`) so that examples cannot quietly rot — the bug that motivated
this was an example calling `zarr.create_array(..., mode="w")`, an argument that does
not exist, which went unnoticed because nothing ran it. A block declares *how* it is
validated using one of two independent attributes:

  - **`exec="true"`** — Markdown Exec runs the block **at docs-build time to render its
    output** into the page. This is the attribute described above; it is also what the
    test suite executes. Use it for ordinary examples whose output should appear in the
    docs.
  - **`test="true"`** — the block is **run by the test suite only**, *not* at build time.
    Use this for an example that should be validated but cannot run in the docs-build
    environment — for example one that needs a GPU or a cloud backend. Markdown Exec
    leaves a `test="true"` block as a static, syntax-highlighted snippet (it never
    executes it), while the test suite still runs it (see the marker note below).

A block may carry both (`exec="true" test="true"`), though in practice `exec="true"`
already implies it is tested, so you rarely need `test="true"` alongside it.

The two attributes are kept separate on purpose: `exec=` controls *build-time rendering*
and `test=` controls *test-time validation*. Tagging a GPU/cloud example `exec="true"`
would make `mkdocs build` try to run it on a machine without that infrastructure and fail
the build; `test="true"` lets it be validated without being built.

##### Opting a block out of validation

A handful of blocks genuinely cannot run and are not executable Python — a REPL
transcript, a deliberately-incorrect "before" snippet, a `--8<--` file include. Mark
these explicitly by opening the fence with
`exec="false" reason="REPL output transcript, not executable source"` (supply a reason
that fits the block).

`exec="false"` with a non-empty `reason` is an explicit, greppable opt-out. A test
(`test_no_unvalidated_blocks`) requires **every** Python block to be either `exec="true"`,
`test="true"`, or `exec="false"` with a reason — so a block can never silently skip
validation. A bare ` ```python ` fence, or a typo like `exec="on"`, fails that test.

##### Marker-bound blocks (GPU, S3)

A `test="true"` block that needs special infrastructure declares a pytest marker with
`markers="..."`, which binds it to that infrastructure in the test suite:

  - `markers="gpu"` — run only under `pytest -m gpu` (the GPU CI environment); skipped
    elsewhere via `importorskip("cupy")`.
  - `markers="s3"` — run against a mock S3 (moto) backend supplied by a test fixture, so
    the example can use a bare `s3://…` URL with no test-only connection details on show.

##### Placement of `test="true"` blocks

Because Markdown Exec does not execute a `test="true"` (or `exec="false"`) block, placing
one *before* an `exec="true"` block on the same page can disrupt the build-time execution
of that later block. Put `test="true"` blocks **after** all `exec="true"` blocks on the
page (or on a page where they are the only Python block). The `test_test_only_blocks_come_last`
test enforces this, and the CI docs build runs with `--strict` so any such breakage fails
the build rather than passing as a warning.

#### Building documentation without executing code blocks

Sometimes, you may want the documentation to build quicker. You can disable code block execution by commenting out the [markdown-exec plugin](https://github.com/zarr-developers/zarr-python/blob/884a8c91afcc3efe28b3da952be3b85125c453cb/mkdocs.yml#L132) in the mkdocs configuration file. This will make code blocks and cross references render incorrectly (i.e., expect build warnings), but also reduces build time by ~3x. Be sure to undo the commenting out before opening your pull request.

### Changelog

zarr-python uses [towncrier](https://towncrier.readthedocs.io/en/stable/tutorial.html) to manage release notes. Most pull requests should include at least one news fragment describing the changes. To add a release note, you'll need the GitHub issue or pull request number and the type of your change (`feature`, `bugfix`, `doc`, `removal`, `misc`). With that, run `towncrier create` with your development environment, which will prompt you for the issue number, change type, and the news text:

```bash
towncrier create
```

Alternatively, you can manually create the files in the `changes` directory using the naming convention `{issue-number}.{change-type}.md`.

See the [towncrier](https://towncrier.readthedocs.io/en/stable/tutorial.html) docs for more.

## Project governance

This section documents the processes that core developers follow to maintain the project. The current core developers are listed in [`TEAM.md`](https://github.com/zarr-developers/zarr-python/blob/main/TEAM.md).

### Merging pull requests

Pull requests submitted by an external contributor should be reviewed and approved by at least one core developer before being merged. Ideally, pull requests submitted by a core developer should be reviewed and approved by at least one other core developer before being merged.

Pull requests should not be merged until all CI checks have passed (GitHub Actions, Codecov) against code that has had the latest main merged in.

Before merging, the milestone must be set to decide whether a PR will be in the next patch, minor, or major release. The next section explains which types of changes go in each release.

### Self-merging pull requests

The default is that a pull request opened by a core developer is reviewed and approved by at least one other core developer before it is merged. We trust core developers to use their judgment, though, and we would rather bias toward action than make routine changes wait on review they do not really need.

So a core developer may merge their own pull request whenever they judge the change to be low-risk, provided the standard merge requirements are met — CI is green against code that has had the latest `main` merged in, a changelog fragment has been added, and the milestone is set — and other core developers have had a fair chance to weigh in. As a rule of thumb, leave the pull request open for a few days before self-merging, unless it is genuinely trivial or time-sensitive. If you are confident a change is fine, merge it; if you have real doubts, ask for a review. It is generally advisable to ping another developer in the PR description for awareness about the direction, even if you choose not to request a formal review.

Some changes warrant more caution, and a second reviewer is usually worth seeking even when you could self-merge: changes to the public API, anything touching data-format or on-disk compatibility, and performance-sensitive code. These are the most expensive to get wrong and the hardest to reverse. Reverts, by contrast, are cheap — if a self-merged change turns out to be a mistake, reverting it is itself a low-risk change that any core developer can make, and the reworked version can go through normal review. When something recently merged is actively causing harm — a broken `main`, a release blocker, or data corruption — fix it fast and request review after the fact rather than waiting.

This policy exists to lower the cost of routine work and to help newer core developers grow comfortable merging changes. It is not a license to merge past an unresolved objection: if another core developer asks to review a change, give them that chance.

### Release procedure

Open an issue on GitHub announcing the release using the release checklist template:
[https://github.com/zarr-developers/zarr-python/issues/new?template=release-checklist.md](https://github.com/zarr-developers/zarr-python/issues/new?template=release-checklist.md). The release checklist includes all steps necessary for the release.

#### Preparing a release

Releases are prepared using the ["Prepare release notes"](https://github.com/zarr-developers/zarr-python/actions/workflows/prepare_release.yml) workflow. To run it:

1. Go to the [workflow page](https://github.com/zarr-developers/zarr-python/actions/workflows/prepare_release.yml) and click "Run workflow".
2. Enter the release version (e.g. `3.2.0`) and the target branch (defaults to `main`).
3. The workflow will run `towncrier build` to render the changelog, remove consumed fragments from `changes/`, and open a pull request on the `release/v<version>` branch.
4. The release PR is automatically labeled `run-downstream`, which triggers the [downstream test workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/downstream.yml) to run Xarray and numcodecs integration tests against the release branch.
5. Review the rendered changelog in `docs/release-notes.md` and verify downstream tests pass before merging.

## Compatibility and versioning policies

### Versioning

Versions of this library are identified by a triplet of integers with the form `<major>.<minor>.<patch>`, for example `3.0.4`. A release of `zarr-python` is associated with a new version identifier. That new identifier is generated by incrementing exactly one of the components of the previous version identifier by 1. When incrementing the `major` component of the version identifier, the `minor` and `patch` components are reset to 0. When incrementing the minor component, the patch component is reset to 0.

Releases are classified by the library changes contained in that release. This classification determines which component of the version identifier is incremented on release.

* **major** releases (for example, `2.18.0` -> `3.0.0`) are for changes that will require extensive adaptation efforts from many users and downstream projects. For example, breaking changes to widely-used user-facing APIs should only be applied in a major release.

  Users and downstream projects should carefully consider the impact of a major release before adopting it. In advance of a major release, developers should communicate the scope of the upcoming changes, and help users prepare for them.

* **minor** releases (for example, `3.0.0` -> `3.1.0`) are for changes that do not require significant effort from most users or downstream projects to respond to. API changes are possible in minor releases if the burden on users imposed by those changes is sufficiently small.

  For example, a recently released API may need fixes or refinements that are breaking, but low impact due to the recency of the feature. Such API changes are permitted in a minor release.

  Minor releases are safe for most users and downstream projects to adopt.

* **patch** releases (for example, `3.1.0` -> `3.1.1`) are for changes that contain no breaking or behaviour changes for downstream projects or users. Examples of changes suitable for a patch release are bugfixes and documentation improvements.

  Users should always feel safe upgrading to the latest patch release.

Note that this versioning scheme is not consistent with [Semantic Versioning](https://semver.org/). Contrary to SemVer, the Zarr library may release breaking changes in `minor` releases, or even `patch` releases under exceptional circumstances. But we should strive to avoid doing so.

A better model for our versioning scheme is [Intended Effort Versioning](https://jacobtomlinson.dev/effver/), or "EffVer". The guiding principle of EffVer is to categorize releases based on the *expected effort required to upgrade to that release*.

Zarr developers should make changes as smooth as possible for users. This means making backwards-compatible changes wherever possible. When a backwards-incompatible change is necessary, users should be notified well in advance, e.g. via informative deprecation warnings.

### Data format compatibility

The Zarr library is an implementation of a file format standard defined externally -- see the [Zarr specifications website](https://zarr-specs.readthedocs.io) for the list of Zarr file format specifications.

If an existing Zarr format version changes, or a new version of the Zarr format is released, then the Zarr library will generally require changes. It is very likely that a new Zarr format will require extensive breaking changes to the Zarr library, and so support for a new Zarr format in the Zarr library will almost certainly come in new `major` release. When the Zarr library adds support for a new Zarr format, there may be a period of accelerated changes as developers refine newly added APIs and deprecate old APIs. In such a transitional phase breaking changes may be more frequent than usual.

### Deprecation policy

Our versioning policy (above) commits us to minimizing the effort users spend upgrading. A deprecation cycle is the main tool we use to honor that commitment: rather than removing or changing public behavior abruptly, we first ship a release that keeps the old behavior working while warning that it is going away, and only remove it in a later release. This section defines when a deprecation cycle is required, how to decide whether a removal is worth it, and the concrete steps a deprecation must follow.

This policy governs the user-facing Python API of `zarr-python`: importable names, function and method signatures, and observable runtime behavior. It does not govern the Zarr data format, whose compatibility is described under [Data format compatibility](#data-format-compatibility) above, nor the `zarr.experimental` namespace, which is governed by the [Experimental API policy](#experimental-api-policy) below. Private API (names prefixed with `_`, and anything not documented as public) may change at any time without a deprecation cycle.

Any backwards-incompatible change to public API -- removing a name, changing a signature in a non-additive way, or changing observable behavior -- requires a deprecation cycle unless it qualifies for one of the [exceptions](#exceptions) listed below.

#### Deciding whether to deprecate

A deprecation cycle has a real cost: every user of the affected API must eventually migrate, and we carry the deprecated code and its warning until removal. Before starting one, weigh the costs and benefits and record your reasoning in the pull request, so that proposals are evaluated on a shared scale. NumPy's [backwards compatibility policy (NEP 23)](https://numpy.org/neps/nep-0023-backwards-compatibility.html#general-principles) is a good guide to the trade-offs involved; the factors below follow the same principles.

- **How many users are affected.** `zarr-python` is widely used, including by downstream libraries, so assume that any public API is used by someone unless there is concrete evidence otherwise. Most users do not follow our issue tracker and discover a removal only when their code breaks after upgrading, so absence of reported usage is not evidence of absence of usage. Base the estimate on usage data where possible -- code search across downstream projects, documented usage, or download statistics.
- **The cost of migration to users.** Is there a drop-in replacement? Can the migration be performed mechanically, or does it require users to redesign their code? A change with a clear, easy migration path is much cheaper to justify than one without.
- **The cost of keeping the API.** What do we pay by not removing it -- maintenance burden, bug surface, duplicated logic, or a worse design that we can't improve while the old API exists? A high carrying cost strengthens the case for removal.
- **Who benefits.** Benefits can include improved functionality, usability, and performance for users, as well as lower maintenance cost and better future extensibility for developers. A change that gives users something they want is a stronger case than one that benefits only maintainers, since users pay the migration cost either way.
- **Whether the goal can be met without removal.** Often a refactor, an alias, or an additive change achieves the same end without breaking anyone. Prefer those. Removing public API is never free; the deprecation cycle is the price, and it is mandatory.

If, after weighing these factors, a removal is not clearly worthwhile, prefer to keep the API. A deprecation warning is a commitment to remove: if you only want to steer users away from an API without removing it, say so in its documentation rather than emitting a warning. Conversely, do not leave a decision to remove an API indefinitely deferred: a vague intention to "eventually" break something is itself a form of technical debt. Either commit to the deprecation and begin the cycle, or decide to keep the API and design around it.

#### How to deprecate

Once a deprecation is decided, it must follow these steps so that users get a consistent, actionable signal:

1. **Emit a warning at runtime.** Raise `zarr.errors.ZarrDeprecationWarning` for an API that will be *removed*. Use `zarr.errors.ZarrFutureWarning` instead when behavior will *change* rather than disappear (for example, a default value that will change), since `FutureWarning` is shown to end users by default. The [`typing_extensions.deprecated`](https://typing-extensions.readthedocs.io/en/latest/#deprecated) decorator is the preferred way to deprecate whole functions, methods, and classes. When calling `warnings.warn` directly, set `stacklevel` so the warning points at the caller's code rather than zarr internals.
2. **Write an actionable message.** The warning message must state the release the deprecation first appeared in, the planned removal (a specific version if known, otherwise "a future release"), and the migration path -- what the user should do instead. A deprecation warning with no replacement guidance is incomplete.
3. **Document it.** Mark the deprecation in the docstring (via the `deprecated` decorator or a "Deprecated" admonition) so it appears in the rendered API documentation, and update any affected user-guide prose.
4. **Add a changelog entry.** Add a `removal` changelog fragment in `changes/` (for example, `changes/1234.removal.md`) so the deprecation is announced under "Deprecations and Removals" in the release notes. When the API is finally removed, add a second `removal` fragment for that release.

When an API is being replaced rather than simply dropped, deprecate the old name and introduce its replacement in a single step, rather than repeatedly adjusting the same signature across releases. Users should have to migrate only once.

#### How long a deprecation lasts

A deprecated API must remain available, emitting its warning, for **at least 6 months and at least one minor release** before it is removed. These are minimums, not targets: for widely-used API, prefer a longer cycle, and communicate the upcoming removal beyond the warning itself (for example, in release notes or a migration guide).

The removal itself is a backwards-incompatible change, and is classified under our [versioning policy](#versioning) by the upgrade effort it imposes -- a `major` release for the removal of widely-used API, a `minor` release when the impact is genuinely small. The deprecation period, not the release type, is the guarantee users rely on: a removal is only permitted once the minimum period has elapsed, regardless of which release it lands in.

#### Exceptions

A full deprecation cycle may be shortened or skipped in these cases. When it is, explain why in the pull request and the changelog entry:

- **Security fixes and data-corruption bugs**, where continuing the old behavior actively harms users.
- **Changes forced by the Zarr format specification**, which are governed by [Data format compatibility](#data-format-compatibility) and may move faster during a format transition.
- **Recently introduced API** that has not yet appeared in a stable release, which can still be changed freely, and **experimental API**, which is exempt and governed by the [Experimental API policy](#experimental-api-policy).


## Experimental API policy

The `zarr.experimental` namespace contains features that are under active development and may change without notice. When contributing to or depending on experimental features, please keep the following in mind:

### For contributors

When adding a new feature to `zarr.experimental`:

1. Place the feature under `src/zarr/experimental/` and export it from `src/zarr/experimental/__init__.py`.
2. Document the feature in `docs/user-guide/experimental.md` and note clearly that it is experimental.
3. Add a changelog entry categorized as `feature`.

We aim to either **promote** or **remove** experimental features within **6 months** of their addition. To promote a feature to stable:

1. Move it from `zarr.experimental` to the appropriate stable module.
2. Keep a deprecated re-export in `zarr.experimental` for one minor release.
3. Update the documentation to reflect the stable location.

### For users

Features in `zarr.experimental` carry no stability guarantees. They may be changed or removed in any release, including patch releases. If you depend on an experimental feature, pin your `zarr-python` version accordingly.

## Benchmarks

Zarr uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) for running
performance benchmarks as part of our test suite. The benchmarks are found in `tests/benchmarks`.
By default pytest is configured to run these benchmarks as plain tests (i.e., no benchmarking). To run
a benchmark with timing measurements, use the `--benchmark-enable` when invoking `pytest`.

The benchmarks are run as part of the continuous integration suite through [codspeed](https://codspeed.io/zarr-developers/zarr-python).
