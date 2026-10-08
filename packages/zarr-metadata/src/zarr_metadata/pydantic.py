"""Optional pydantic (v2) integration: field types over the core models.

Importing this module requires pydantic; the core package deliberately does
not depend on it, so this module is never imported by `zarr_metadata` itself.

Each exported name is an `Annotated` field type over the corresponding core
model class — the instances ARE the core classes, so values interoperate
freely with non-pydantic code (equality, isinstance, nesting). Validation
delegates to the library: a raw document routes through `from_json` (the
single source of truth for validation and normalization, so pydantic's
field-level coercion can never bypass it). A v3 field type reads extension
points in the scope pydantic's validation context holds, as pydantic hands
any validator its context: the context itself, when it is a `Context`, or
its `"zarr_metadata_context"` item, when it is a mapping; otherwise
`CORE_AND_EXTENSIONS`:

    TypeAdapter(zmp.ZarrV3ArrayMetadata).validate_python(document, context=SCOPE)
    ArrayManifest.model_validate(data, context={"zarr_metadata_context": SCOPE})

An existing model instance passes through unchanged, as pydantic's does, and
serialization emits the canonical document via `to_json`. A failed parse
surfaces as a pydantic `ValidationError` with one line error per problem,
as pydantic reports its own: its `type` the problem's `kind`, at the
problem's `loc` under the field's, with the `input` found there and the
problem's `ctx` (a message holding a ctx placeholder rides in the ctx as
`message`, as `_line_error` says). Annotate a field with these types, not the core classes,
which pydantic cannot build a schema for.

A node's attributes may hold `NaN`, `Infinity` or `-Infinity`, which the
models read and write as zarr-python does. Pydantic writes JSON by its own
rules, and by default writes such a number as `null`. A `BaseModel` holding
one of these fields keeps it with `ser_json_inf_nan="constants"` in its
`model_config`; a `TypeAdapter` over a field type takes no `config`, so
write its value with the model's `to_key_value` instead.

Usage:

    import zarr_metadata.pydantic as zmp

    class ArrayManifest(BaseModel):
        path: str
        metadata: zmp.ZarrV3ArrayMetadata

Static type checkers see each field type as its core model class, so
`manifest.metadata` is a `zarr_metadata.model.ZarrV3ArrayMetadata`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Final, LiteralString, Protocol, TypeVar, cast

from pydantic import BeforeValidator, InstanceOf, PlainSerializer, ValidationInfo
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

from zarr_metadata import model as _model
from zarr_metadata._json import MetadataValidationError, ValidationProblem, value_at
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
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Callable

_M = TypeVar("_M")


def _as_pydantic_raises(cls: type[_M], value: object, read: Callable[[], _M]) -> _M:
    """What `read` gives of `value`, or the `MetadataValidationError` it raises as a `pydantic_core.ValidationError`: one line error per problem, its type the problem's kind, at the problem's loc, with its input and ctx.

    A missing key's problem reports the object missing it, as pydantic's
    own `missing` error does; one that holds nothing else, what it found
    not being JSON a reader walks, reports what sits at its loc.
    """
    try:
        return read()
    except MetadataValidationError as error:
        line_errors = [_line_error(problem, value) for problem in error.problems]
        raise ValidationError.from_exception_data(cls.__name__, line_errors) from error


def _line_error(problem: ValidationProblem, value: object) -> InitErrorDetails:
    """`problem` as one of a `ValidationError`'s line errors: its type the problem's kind, at its loc, with its input and ctx, and its message as `msg`.

    pydantic renders a message as a template of its ctx, each `{key}` the
    ctx holds replaced, one key at a time in the ctx's order, with no way
    to escape one. A message that holds one -- a document value written
    `{expected}`, shown in it -- is handed whole as the ctx's `message`,
    last, where what it carries is scanned for no key after it, and the
    template is that placeholder, so `msg` is the problem's message
    whatever it holds; a ctx member of that name yields to it.
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
    """What a line error reports as its input: what the problem found; for a key that is missing, what `value` holds at the loc above, as pydantic's own `missing` reports the object; for what the problem could not hold, not being JSON a reader walks, what sits at its loc; and `value` itself where that is nothing either."""
    if problem.input is not UNSET:
        return problem.input
    if problem.kind == "missing_key":
        above = value_at(value, problem.loc[:-1]) if len(problem.loc) != 0 else UNSET
        return value if above is UNSET else above
    there = value_at(value, problem.loc)
    return value if there is UNSET else there


CONTEXT_KEY: Final = "zarr_metadata_context"
"""The item of a mapping pydantic's validation context is that holds the scope a v3 field type reads in."""


_Read_co = TypeVar("_Read_co", covariant=True)


class _Reads(Protocol[_Read_co]):
    def __call__(self, data: object, /, *, context: Context) -> _Read_co: ...


def _read_in_scope(
    cls: type[_M], read: _Reads[_M], default: Context
) -> Callable[[object, ValidationInfo], _M]:
    """A validator that passes instances of `cls` through and reads anything else in the scope the validation context holds, `default` when it holds none."""

    def coerce(value: object, info: ValidationInfo) -> _M:
        if isinstance(value, cls):
            return value
        return _as_pydantic_raises(
            cls, value, lambda: read(value, context=_scope(info.context, default))
        )

    return coerce


def _scope(context: object, default: Context) -> Context:
    """The scope a validation context holds: itself, a `Context`; its `CONTEXT_KEY` item; or, holding none, `default`: the format's own scope."""
    if isinstance(context, Context):
        return context
    if not isinstance(context, Mapping) or CONTEXT_KEY not in context:
        return default
    scope = cast("Mapping[object, object]", context)[CONTEXT_KEY]
    if not isinstance(scope, Context):
        msg = f"{CONTEXT_KEY}: the scope to read in is a Context, got {scope!r}"
        raise TypeError(msg)
    return scope


ZarrV3ArrayMetadata = Annotated[
    InstanceOf[_model.ZarrV3ArrayMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3ArrayMetadata, _model.ZarrV3ArrayMetadata.from_json, CORE_AND_EXTENSIONS
        ),
        json_schema_input_type=_ZarrV3ArrayMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV3ArrayMetadata.to_json, return_type=_ZarrV3ArrayMetadataSchema),
]
"""Field type for a v3 array metadata document (`zarr.json` content)."""

ZarrV2ArrayMetadata = Annotated[
    InstanceOf[_model.ZarrV2ArrayMetadata],
    BeforeValidator(
        _read_in_scope(_model.ZarrV2ArrayMetadata, _model.ZarrV2ArrayMetadata.from_json, CORE_V2),
        json_schema_input_type=_ZarrV2ArrayMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV2ArrayMetadata.to_json, return_type=_ZarrV2ArrayMetadataSchema),
]
"""Field type for a v2 array metadata document (merged `.zarray` + `.zattrs` form)."""

ZarrV3GroupMetadata = Annotated[
    InstanceOf[_model.ZarrV3GroupMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3GroupMetadata, _model.ZarrV3GroupMetadata.from_json, CORE_AND_EXTENSIONS
        ),
        json_schema_input_type=_ZarrV3GroupMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV3GroupMetadata.to_json, return_type=_ZarrV3GroupMetadataSchema),
]
"""Field type for a v3 group metadata document (`zarr.json` content)."""

ZarrV2GroupMetadata = Annotated[
    InstanceOf[_model.ZarrV2GroupMetadata],
    BeforeValidator(
        _read_in_scope(_model.ZarrV2GroupMetadata, _model.ZarrV2GroupMetadata.from_json, CORE_V2),
        json_schema_input_type=_ZarrV2GroupMetadataSchema,
    ),
    PlainSerializer(_model.ZarrV2GroupMetadata.to_json, return_type=_ZarrV2GroupMetadataSchema),
]
"""Field type for a v2 group metadata document (merged `.zgroup` + `.zattrs` form)."""

ZarrV3ConsolidatedMetadata = Annotated[
    InstanceOf[_model.ZarrV3ConsolidatedMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV3ConsolidatedMetadata,
            _model.ZarrV3ConsolidatedMetadata.from_json,
            CORE_AND_EXTENSIONS,
        ),
        json_schema_input_type=_ZarrV3ConsolidatedMetadataSchema,
    ),
    PlainSerializer(
        _model.ZarrV3ConsolidatedMetadata.to_json,
        return_type=_ZarrV3ConsolidatedMetadataSchema,
    ),
]
"""Field type for v3 inline consolidated metadata."""

ZarrV2ConsolidatedMetadata = Annotated[
    InstanceOf[_model.ZarrV2ConsolidatedMetadata],
    BeforeValidator(
        _read_in_scope(
            _model.ZarrV2ConsolidatedMetadata, _model.ZarrV2ConsolidatedMetadata.from_json, CORE_V2
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
    "ZarrV2ArrayMetadata",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2GroupMetadata",
    "ZarrV3ArrayMetadata",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3GroupMetadata",
]
