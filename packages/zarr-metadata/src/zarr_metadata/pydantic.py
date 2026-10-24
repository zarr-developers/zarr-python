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

A mapping key that starts like these keys but is not one of them raises
`TypeError`, so a misspelled key is not read as no scope. To fix a field's
scope whatever the validation context holds, annotate the model class
with `InScope`: `Annotated[ZarrV3ArrayMetadata, InScope(CONTEXT)]`.

For example:

    TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(document, context=CONTEXT)
    ArrayManifest.model_validate(data, context={"zarr_metadata_context": CONTEXT})
    Mixed.model_validate(data, context={"zarr_metadata_context": V3, "zarr_metadata_context_v2": V2})

An existing model instance passes through unchanged when no scope is
given. When one is, the instance is read again in it with `refined_in`, so
an instance that scope reads differently is refused. Serialization emits
the document as it was written, via `to_json`. A failed parse
raises a pydantic `ValidationError` with one line error per problem, in
the same format pydantic uses for its own errors. Each line error has the
problem's `kind` as its `type`, the problem's `loc` under the field's
`loc`, the `input` found there, and the problem's `ctx`. If the message
contains a ctx placeholder, the message is passed in the ctx as
`message`. Annotate a field with these types, not the
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

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    LiteralString,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
)

from pydantic import (
    BeforeValidator,
    GetCoreSchemaHandler,
    InstanceOf,
    PlainSerializer,
    ValidationInfo,
)
from pydantic_core import (
    CoreSchema,
    InitErrorDetails,
    PydanticCustomError,
    ValidationError,
    core_schema,
)

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
from zarr_metadata.definition._context import scoped
from zarr_metadata.definition._definition import Context
from zarr_metadata.v2.definition import CORE_V2, ZarrV2Context
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, ZarrV3Context

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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


_Context_contra = TypeVar("_Context_contra", bound=Context, contravariant=True)
_C = TypeVar("_C", bound=Context)


class _Reads(Protocol[_Read_co, _Context_contra]):
    def __call__(self, data: object, /, *, context: _Context_contra) -> _Read_co: ...


def _read_in_context(
    cls: type[_M], read: _Reads[_M, _C], refine: Callable[[_M, _C], _M], default: _C, key: str
) -> Callable[[object, ValidationInfo], _M]:
    """A validator that reads a value in the `Context` stored in the validation context under `key`, or in `default` if there is none. An instance of `cls` is read again in a scope the validation context gives, with `refine` (its `refined_in`), and passes through unchanged when none is given."""

    def coerce(value: object, info: ValidationInfo) -> _M:
        given = _context_of(info.context, default, key)
        if isinstance(value, cls):
            return value if given is None else refine(value, given)
        context = default if given is None else given
        return _as_pydantic_raises(cls, value, lambda: read(value, context=context))

    return coerce


def _read_in(
    cls: type[_M], read: _Reads[_M, _C], refine: Callable[[_M, _C], _M], context: _C
) -> Callable[[object], _M]:
    """A validator that reads a value in `context`, and an instance of `cls` again in `context`, with `refine` (its `refined_in`)."""

    def coerce(value: object) -> _M:
        if isinstance(value, cls):
            return refine(value, context)
        return _as_pydantic_raises(cls, value, lambda: read(value, context=context))

    return coerce


def _is_context(value: object) -> TypeGuard[ZarrV3Context | ZarrV2Context]:
    """Whether `value` is a scope of either format."""
    return isinstance(value, (ZarrV3Context, ZarrV2Context))


def _context_of(validation_context: object, default: _C, key: str) -> _C | None:
    """The `Context` a pydantic validation context gives for one format, of `default`'s class. If the validation context is a `Context`, it is the context for every field type of its format. If it is a mapping with a `key` item, that item is the context for this format. Otherwise it gives none.

    A bare `Context` of the other format raises `TypeError`, as it does in
    every reader. A model with fields of both formats must give each
    format's context under its own key. A mapping key that starts like
    this module's keys but is not one of them raises `TypeError` too, so a
    misspelled key is not read as no scope at all.
    """
    if _is_context(validation_context):
        return scoped(validation_context, default)
    if not is_object(validation_context):
        return None
    unknown = [
        name
        for name in validation_context
        if isinstance(name, str)
        and name.startswith("zarr_metadata")
        and name not in (CONTEXT_KEY, CONTEXT_KEY_V2)
    ]
    if len(unknown) != 0:
        msg = (
            f"the validation context gives a scope under {CONTEXT_KEY!r} or "
            f"{CONTEXT_KEY_V2!r}, got {', '.join(map(repr, unknown))}"
        )
        raise TypeError(msg)
    if key not in validation_context:
        return None
    context = validation_context[key]
    if not _is_context(context):
        msg = (
            f"{key}: the context to read in is a ZarrV3Context or a ZarrV2Context, got {context!r}"
        )
        raise TypeError(msg)
    return scoped(context, default)


ZarrV3ArrayMetadataField = Annotated[
    InstanceOf[_model.ZarrV3ArrayMetadata],
    BeforeValidator(
        _read_in_context(
            _model.ZarrV3ArrayMetadata,
            _model.ZarrV3ArrayMetadata.from_json,
            _model.ZarrV3ArrayMetadata.refined_in,
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
        _read_in_context(
            _model.ZarrV2ArrayMetadata,
            _model.ZarrV2ArrayMetadata.from_json,
            _model.ZarrV2ArrayMetadata.refined_in,
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
        _read_in_context(
            _model.ZarrV3GroupMetadata,
            _model.ZarrV3GroupMetadata.from_json,
            _model.ZarrV3GroupMetadata.refined_in,
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
        _read_in_context(
            _model.ZarrV2GroupMetadata,
            _model.ZarrV2GroupMetadata.from_json,
            _model.ZarrV2GroupMetadata.refined_in,
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
        _read_in_context(
            _model.ZarrV3ConsolidatedMetadata,
            _model.ZarrV3ConsolidatedMetadata.from_json,
            _model.ZarrV3ConsolidatedMetadata.refined_in,
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
        _read_in_context(
            _model.ZarrV2ConsolidatedMetadata,
            _model.ZarrV2ConsolidatedMetadata.from_json,
            _model.ZarrV2ConsolidatedMetadata.refined_in,
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

_FIELD_PARTS: Final[Mapping[type[Any], tuple[ZarrV3Context | ZarrV2Context, object]]] = (
    MappingProxyType(
        {
            _model.ZarrV3ArrayMetadata: (CORE_AND_EXTENSIONS, _ZarrV3ArrayMetadataSchema),
            _model.ZarrV2ArrayMetadata: (CORE_V2, _ZarrV2ArrayMetadataSchema),
            _model.ZarrV3GroupMetadata: (CORE_AND_EXTENSIONS, _ZarrV3GroupMetadataSchema),
            _model.ZarrV2GroupMetadata: (CORE_V2, _ZarrV2GroupMetadataSchema),
            _model.ZarrV3ConsolidatedMetadata: (
                CORE_AND_EXTENSIONS,
                _ZarrV3ConsolidatedMetadataSchema,
            ),
            _model.ZarrV2ConsolidatedMetadata: (CORE_V2, _ZarrV2ConsolidatedMetadataSchema),
        }
    )
)
"""For each model class, the default scope of its format and the pydantic schema of its document."""


@dataclass(frozen=True, slots=True)
class InScope:
    """A field type's scope, fixed: `Annotated[ZarrV3ArrayMetadata, InScope(CONTEXT)]` is a field of that model class read in `CONTEXT`, whatever the validation context holds.

    A document is read in `context`, and a model instance is read again
    in it with `refined_in`, so an instance that `context` reads
    differently is refused rather than passed through in its own scope.
    The field writes the document as written, and a type checker sees the
    model class. A scope of the other format raises `TypeError` when
    pydantic builds the schema.
    """

    context: ZarrV3Context | ZarrV2Context

    def __get_pydantic_core_schema__(
        self,
        source: Any,  # noqa: ANN401 - pydantic passes the annotated class
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        parts = _FIELD_PARTS.get(source)
        if parts is None:
            msg = f"InScope annotates a model class of zarr_metadata.model, got {source!r}"
            raise TypeError(msg)
        default, schema = parts
        cls: type[Any] = source
        document = handler.generate_schema(schema)
        return core_schema.no_info_before_validator_function(
            _read_in(cls, cls.from_json, cls.refined_in, scoped(self.context, default)),
            core_schema.is_instance_schema(cls),
            json_schema_input_schema=document,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.to_json, return_schema=document
            ),
        )


__all__ = [
    "CONTEXT_KEY",
    "CONTEXT_KEY_V2",
    "InScope",
    "ZarrV2ArrayMetadataField",
    "ZarrV2ConsolidatedMetadataField",
    "ZarrV2GroupMetadataField",
    "ZarrV3ArrayMetadataField",
    "ZarrV3ConsolidatedMetadataField",
    "ZarrV3GroupMetadataField",
]
