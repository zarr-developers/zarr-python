"""Optional pydantic (v2) integration: field types over the core models.

Importing this module requires pydantic; the core package deliberately does
not depend on it, so this module is never imported by `zarr_metadata` itself.

Each exported name is an `Annotated` field type over the corresponding core
model class — the instances ARE the core classes, so values interoperate
freely with non-pydantic code (equality, isinstance, nesting). Validation
delegates to the library: a raw document routes through `from_json` (the
single source of truth for validation and normalization, so pydantic's
field-level coercion can never bypass it). A field type reads the fields
of its document in the scope given by pydantic's validation context
(pydantic passes this context to every validator). The scope is chosen
as follows:

- If the context is a `Context`, every field type of that format reads
  in it. A field type of the other format raises `TypeError`.
- If the context is a mapping, the v3 field types read in its
  `"zarr_metadata_context"` item and the v2 field types read in its
  `"zarr_metadata_context_v2"` item. This lets a model with fields of
  both formats give a scope for each format.
- If no scope is given for a format, its field types read in the default
  scope for that format, `CORE_AND_EXTENSIONS` or `CORE_V2`.

For example:

    TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(document, context=SCOPE)
    ArrayManifest.model_validate(data, context={"zarr_metadata_context": SCOPE})
    Mixed.model_validate(data, context={"zarr_metadata_context": V3, "zarr_metadata_context_v2": V2})

An existing model instance passes through unchanged, as pydantic's does, and
serialization emits the canonical document via `to_json`. A failed parse
raises a pydantic `ValidationError` with one line error per problem, in
the same format pydantic uses for its own errors. Each line error has the
problem's `kind` as its `type`, the problem's `loc` under the field's
`loc`, the `input` found there, and the problem's `ctx`. If the message
contains a ctx placeholder, the message is passed in the ctx as
`message`; see `_line_error`. Annotate a field with these types, not the
core classes, because pydantic cannot build a schema for the core classes.

A node's attributes may hold `NaN`, `Infinity` or `-Infinity`, which the
models read and write as zarr-python does. Pydantic writes JSON by its own
rules, and by default writes such a number as `null`. To keep these
numbers, a `BaseModel` with one of these fields needs
`ser_json_inf_nan="constants"` in its `model_config`. A `TypeAdapter`
over a field type takes no `config`, so write its value with the model's
`to_key_value` instead.

Pydantic's own JSON code paths limit nesting to less than the 256 levels
the readers support: `validate_json` rejects a document nested about 200
levels deep, and `dump_json` rejects one nested about 255 levels deep,
each with its own error. For a document that deep, call
`validate_python` on the parsed JSON, and write it with the model's
`to_key_value`.

Usage:

    import zarr_metadata.pydantic as zmp

    class ArrayManifest(BaseModel):
        path: str
        metadata: zmp.ZarrV3ArrayMetadataField

Static type checkers see each field type as its core model class, so
`manifest.metadata` is a `zarr_metadata.model.ZarrV3ArrayMetadata`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Final, LiteralString, Protocol, TypeVar, cast

from pydantic import BeforeValidator, InstanceOf, PlainSerializer, ValidationInfo
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

from zarr_metadata import model as _model
from zarr_metadata._json import MetadataValidationError, ValidationProblem, is_object, value_at
from zarr_metadata._pydantic_schema import (
    ZarrV2ArrayMetadataJSON as _ZarrV2ArrayMetadataSchema,
)
from zarr_metadata._pydantic_schema import (
    ZarrV2ConsolidatedMetadataJSON as _ZarrV2ConsolidatedMetadataSchema,
)
from zarr_metadata._pydantic_schema import (
    ZarrV2GroupMetadataJSON as _ZarrV2GroupMetadataSchema,
)
from zarr_metadata._pydantic_schema import (
    ZarrV3ArrayMetadataJSON as _ZarrV3ArrayMetadataSchema,
)
from zarr_metadata._pydantic_schema import (
    ZarrV3ConsolidatedMetadataJSON as _ZarrV3ConsolidatedMetadataSchema,
)
from zarr_metadata._pydantic_schema import (
    ZarrV3GroupMetadataJSON as _ZarrV3GroupMetadataSchema,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._context import (
    Context,
    is_scope,
    scoped,
)
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

_M = TypeVar("_M")


def _as_pydantic_raises(cls: type[_M], value: object, read: Callable[[], _M]) -> _M:
    """Return the result of `read` for `value`, or convert the `MetadataValidationError` it raises into a `pydantic_core.ValidationError`: one line error per problem, with the problem's kind as its type, at the problem's loc, with its input and ctx.

    For a missing key, the line error reports the object that lacks the
    key, as pydantic's own `missing` error does. For a problem with no
    `input` (because the value found was not JSON a reader can walk), the
    line error reports the value at the problem's loc.
    """
    try:
        return read()
    except MetadataValidationError as error:
        line_errors = [_line_error(problem, value) for problem in error.problems]
        raise ValidationError.from_exception_data(cls.__name__, line_errors) from error


def _line_error(problem: ValidationProblem, value: object) -> InitErrorDetails:
    """Convert `problem` to a `ValidationError` line error: the problem's kind as its type, at its loc, with its input and ctx, and its message as `msg`.

    pydantic treats a message as a template over its ctx. It replaces each
    `{key}` that the ctx holds, one key at a time in ctx order, and has no
    way to escape a placeholder. A problem message can contain such a
    placeholder, for example when it shows a document value written as
    `{expected}`. In that case the whole message is put in the ctx as
    `message`, as the last item, so no later key is substituted into it,
    and the template becomes `{message}`. This way `msg` is always the
    problem's message, whatever it contains. An existing ctx item named
    `message` is replaced.
    """
    template, ctx = problem.message, dict(problem.ctx)
    if any(f"{{{key}}}" in template for key in ctx):
        ctx.pop("message", None)
        template, ctx = "{message}", {**ctx, "message": problem.message}
    return InitErrorDetails(
        type=PydanticCustomError(problem.kind, cast("LiteralString", template), ctx),
        loc=problem.loc,
        input=_input_of(problem, value),
    )


def _input_of(problem: ValidationProblem, value: object) -> object:
    """The input a line error reports. This is the problem's `input` if it has one. For a missing key, it is the value in `value` at the parent loc, as pydantic's own `missing` error reports the object. For a problem whose value was not JSON a reader can walk (so the problem could not hold it), it is the value at the problem's loc. If nothing is found there either, it is `value` itself."""
    if problem.input is not UNSET:
        return problem.input
    if problem.kind == "missing_key":
        above = value_at(value, problem.loc[:-1]) if len(problem.loc) != 0 else UNSET
        return value if above is UNSET else above
    there = value_at(value, problem.loc)
    return value if there is UNSET else there


CONTEXT_KEY: Final = "zarr_metadata_context"
"""The key in a mapping validation context that holds the scope for the v3 field types."""
CONTEXT_KEY_V2: Final = "zarr_metadata_context_v2"
"""The key in a mapping validation context that holds the scope for the v2 field types."""


_Read_co = TypeVar("_Read_co", covariant=True)


class _Reads(Protocol[_Read_co]):
    def __call__(self, data: object, /, *, context: Context) -> _Read_co: ...


def _read_in_scope(
    cls: type[_M], read: _Reads[_M], default: Context[Any], key: str
) -> Callable[[object, ValidationInfo], _M]:
    """A validator that returns instances of `cls` unchanged, and reads any other value in the scope stored in the validation context under `key`, or in `default` if there is none."""

    def coerce(value: object, info: ValidationInfo) -> _M:
        if isinstance(value, cls):
            return value
        return _as_pydantic_raises(
            cls, value, lambda: read(value, context=_scope(info.context, default, key))
        )

    return coerce


def _scope(context: object, default: Context[Any], key: str) -> Context[Any]:
    """The scope a validation context gives for one format. If the context is a `Context`, it is the scope for every field type of its format. If it is a mapping with a `key` item, that item is the scope for this format. Otherwise the scope is `default`, the format's core scope.

    A bare `Context` of the other format raises `TypeError`, as it does in
    every reader. A model with fields of both formats must give each
    format's scope under its own key.
    """
    if is_scope(context):
        return scoped(context, default)
    if not is_object(context) or key not in context:
        return default
    scope = context[key]
    if not is_scope(scope):
        msg = f"{key}: the scope to read in is a Context, got {scope!r}"
        raise TypeError(msg)
    return scope


ZarrV3ArrayMetadataField = Annotated[
    InstanceOf[_model.ZarrV3ArrayMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3ArrayMetadata,
            _model.ZarrV3ArrayMetadata.from_json,
            CORE_AND_EXTENSIONS,
            CONTEXT_KEY,
        ),
        json_schema_input_type=_ZarrV3ArrayMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV3ArrayMetadata.to_json, return_type=_ZarrV3ArrayMetadataSchema),
]
"""Field type for a v3 array metadata document (`zarr.json` content)."""

ZarrV2ArrayMetadataField = Annotated[
    InstanceOf[_model.ZarrV2ArrayMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV2ArrayMetadata,
            _model.ZarrV2ArrayMetadata.from_json,
            CORE_V2,
            CONTEXT_KEY_V2,
        ),
        json_schema_input_type=_ZarrV2ArrayMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV2ArrayMetadata.to_json, return_type=_ZarrV2ArrayMetadataSchema),
]
"""Field type for a v2 array metadata document (merged `.zarray` + `.zattrs` form)."""

ZarrV3GroupMetadataField = Annotated[
    InstanceOf[_model.ZarrV3GroupMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3GroupMetadata,
            _model.ZarrV3GroupMetadata.from_json,
            CORE_AND_EXTENSIONS,
            CONTEXT_KEY,
        ),
        json_schema_input_type=_ZarrV3GroupMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV3GroupMetadata.to_json, return_type=_ZarrV3GroupMetadataSchema),
]
"""Field type for a v3 group metadata document (`zarr.json` content)."""

ZarrV2GroupMetadataField = Annotated[
    InstanceOf[_model.ZarrV2GroupMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV2GroupMetadata,
            _model.ZarrV2GroupMetadata.from_json,
            CORE_V2,
            CONTEXT_KEY_V2,
        ),
        json_schema_input_type=_ZarrV2GroupMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV2GroupMetadata.to_json, return_type=_ZarrV2GroupMetadataSchema),
]
"""Field type for a v2 group metadata document (merged `.zgroup` + `.zattrs` form)."""

ZarrV3ConsolidatedMetadataField = Annotated[
    InstanceOf[_model.ZarrV3ConsolidatedMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3ConsolidatedMetadata,
            _model.ZarrV3ConsolidatedMetadata.from_json,
            CORE_AND_EXTENSIONS,
            CONTEXT_KEY,
        ),
        json_schema_input_type=_ZarrV3ConsolidatedMetadataSchema,
    ),
    PlainSerializer(
        _model.ZarrV3ConsolidatedMetadata.to_json,
        return_type=_ZarrV3ConsolidatedMetadataSchema,
    ),
]
"""Field type for v3 inline consolidated metadata."""

ZarrV2ConsolidatedMetadataField = Annotated[
    InstanceOf[_model.ZarrV2ConsolidatedMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV2ConsolidatedMetadata,
            _model.ZarrV2ConsolidatedMetadata.from_json,
            CORE_V2,
            CONTEXT_KEY_V2,
        ),
        json_schema_input_type=_ZarrV2ConsolidatedMetadataSchema,
    ),
    PlainSerializer(
        _model.ZarrV2ConsolidatedMetadata.to_json,
        return_type=_ZarrV2ConsolidatedMetadataSchema,
    ),
]
"""Field type for a v2 `.zmetadata` document."""

__all__ = [
    "CONTEXT_KEY",
    "CONTEXT_KEY_V2",
    "ZarrV2ArrayMetadataField",
    "ZarrV2ConsolidatedMetadataField",
    "ZarrV2GroupMetadataField",
    "ZarrV3ArrayMetadataField",
    "ZarrV3ConsolidatedMetadataField",
    "ZarrV3GroupMetadataField",
]
