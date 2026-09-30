# Deprecations

This page indexes every deprecated part of the `zarr-python` API, with the
migration path and the planned removal. The [deprecation
policy](contributing.md#deprecation-policy) in the contributing guide explains
when an API is deprecated, how long it stays available (at least 6 months and
at least one minor release), and what a deprecation warning must say.

Both tables are generated at build time from the source, so they always match
the warnings the current release emits.

## Deprecated functions, classes, and properties

Each of these carries a `@deprecated` decorator. Its message, shown here and in
the API reference, states what to use instead and when the API is planned to go
away.

<!-- deprecated-api-table -->

## Other deprecations

Deprecations that cannot be expressed with a decorator, such as a deprecated
import location, a deprecated way of passing a value, or a behavior that will
change, are declared in `zarr._deprecations` and warn from inside the code
that handles them.

<!-- declared-deprecations-table -->
