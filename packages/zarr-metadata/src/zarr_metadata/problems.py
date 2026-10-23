"""The problems every reader in this package reports, and the error that carries them.

A reader returns problems as values: `typed_json.check`, `definition.resolve`
and the model readers each return a tuple of `ValidationProblem`. Each
problem has a `loc` (where it is), a `kind` (one of the `ProblemKind`
values, for code that handles problems by kind), a `message`, and the
`input` and `ctx` its message is made from. A constructor that cannot
return problems, such as a model's `from_json`, raises
`MetadataValidationError` with every problem in `.problems`.
"""

from zarr_metadata._json import MetadataValidationError, ProblemKind, ValidationProblem

__all__ = [
    "MetadataValidationError",
    "ProblemKind",
    "ValidationProblem",
]
