"""The problems every reader in this package reports, and the error that carries them.

A reader returns problems as values: `typed_json.check`, `definition.resolve`
and the model readers each return a tuple of `ValidationProblem`. Each
problem has a `loc` (where it is), a `kind` (one of the `ProblemKind`
values, for code that handles problems by kind), a `message`, and the
`input` and `ctx` its message is made from. A constructor that cannot
return problems, such as a model's `from_json`, raises
`MetadataValidationError` with every problem in `.problems`.

`shown` formats a value for a problem's message, as the package's own
messages do: as JSON (`null`, `[1, 2]`, `"C"`). An extension's rules use it
so their messages read like the package's.
"""

from zarr_metadata._json import MetadataValidationError, ProblemKind, ValidationProblem, shown
from zarr_metadata.definition._definition import Problems

__all__ = [
    "MetadataValidationError",
    "ProblemKind",
    "Problems",
    "ValidationProblem",
    "shown",
]
