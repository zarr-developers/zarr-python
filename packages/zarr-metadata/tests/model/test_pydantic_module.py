"""Tests for `zarr_metadata.pydantic`, the optional pydantic field-type module.

The hand-rolled recipes in `test_pydantic.py` document how the integration
works; this module ships it. Instances are the CORE model classes (no
parallel hierarchy), so values interoperate freely with non-pydantic code.
"""

import json
import math
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import zarr_metadata.pydantic as zmp
from zarr_metadata.definition import AcceptedField, UnclaimedField
from zarr_metadata.json_schema import json_schema, node_metadata_json_schema_v3
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
)
from zarr_metadata.problems import ValidationProblem
from zarr_metadata.typed_json import JSONValue
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, ZarrV3Context

V3_ARRAY_DOC = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
V2_ARRAY_DOC = dict(ZarrV2ArrayMetadata.create_default(shape=(4,), chunks=(2,)).to_json())
V3_GROUP_DOC = {"zarr_format": 3, "node_type": "group", "attributes": {"a": 1}}
V2_GROUP_DOC = {"zarr_format": 2, "attributes": {"a": 1}}
V3_CONSOLIDATED_DOC = {
    "kind": "inline",
    "must_understand": False,
    "metadata": {"a": dict(V3_ARRAY_DOC)},
}
V2_CONSOLIDATED_DOC = {
    "zarr_consolidated_format": 1,
    "metadata": {".zgroup": {"zarr_format": 2}},
}

FIELD_CASES = [
    pytest.param(zmp.ZarrV3ArrayMetadataField, ZarrV3ArrayMetadata, V3_ARRAY_DOC, id="array-v3"),
    pytest.param(zmp.ZarrV2ArrayMetadataField, ZarrV2ArrayMetadata, V2_ARRAY_DOC, id="array-v2"),
    pytest.param(zmp.ZarrV3GroupMetadataField, ZarrV3GroupMetadata, V3_GROUP_DOC, id="group-v3"),
    pytest.param(zmp.ZarrV2GroupMetadataField, ZarrV2GroupMetadata, V2_GROUP_DOC, id="group-v2"),
    pytest.param(
        zmp.ZarrV3ConsolidatedMetadataField,
        ZarrV3ConsolidatedMetadata,
        V3_CONSOLIDATED_DOC,
        id="consolidated-v3",
    ),
    pytest.param(
        zmp.ZarrV2ConsolidatedMetadataField,
        ZarrV2ConsolidatedMetadata,
        V2_CONSOLIDATED_DOC,
        id="consolidated-v2",
    ),
]


@pytest.mark.parametrize(("field_type", "model_cls", "doc"), FIELD_CASES)
def test_field_type_validates_and_dumps_canonically(
    field_type: object, model_cls: type, doc: dict[str, object]
) -> None:
    """Each field type parses its raw document into the CORE model class,
    passes existing instances through unchanged, and dumps the canonical
    document via to_json."""
    adapter = TypeAdapter(field_type)
    model = adapter.validate_python(doc)
    assert type(model) is model_cls
    assert adapter.validate_python(model) is model
    assert adapter.dump_python(model) == model.to_json()


def test_core_instances_interoperate() -> None:
    """A core model instance (e.g. returned by zarr-python) drops straight
    into a pydantic field — the reason the module ships Annotated aliases over
    the core classes rather than pydantic-aware subclasses."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadataField

    core = ZarrV3ArrayMetadata.from_json(V3_ARRAY_DOC)
    manifest = Manifest(metadata=core)
    assert manifest.metadata is core


def test_validation_error_carries_problems() -> None:
    """A defective document fails with the library's loc-annotated messages."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadataField

    doc = dict(V3_ARRAY_DOC)
    del doc["chunk_key_encoding"]
    with pytest.raises(ValidationError) as raised:
        Manifest.model_validate({"metadata": doc})
    (error,) = raised.value.errors()
    assert (error["type"], error["loc"], error["msg"]) == (
        "missing_key",
        ("metadata", "chunk_key_encoding"),
        "missing required key",
    )
    # For a missing key, the input is the object that lacks the key, as in pydantic's own errors.
    assert error["input"] == doc


def test_json_schema_generation() -> None:
    """model_json_schema works, describing the document form each field accepts."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadataField

    schema = Manifest.model_json_schema()
    metadata_schema = schema["$defs"]["ZarrV3ArrayMetadataJSON"]
    assert schema["properties"]["metadata"]["$ref"] == "#/$defs/ZarrV3ArrayMetadataJSON"
    assert metadata_schema["required"] == [
        "zarr_format",
        "node_type",
        "data_type",
        "shape",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
    ]
    assert metadata_schema["properties"]["zarr_format"] == {
        "const": 3,
        "title": "Zarr Format",
        "type": "integer",
    }


def test_json_schema_generation_emits_no_warnings() -> None:
    """Consumers can generate every public integration schema without warning filters."""
    field_types = (
        zmp.ZarrV3ArrayMetadataField,
        zmp.ZarrV2ArrayMetadataField,
        zmp.ZarrV3GroupMetadataField,
        zmp.ZarrV2GroupMetadataField,
        zmp.ZarrV3ConsolidatedMetadataField,
        zmp.ZarrV2ConsolidatedMetadataField,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for field_type in field_types:
            TypeAdapter(field_type).json_schema()


def test_v2_recursive_structured_dtype_is_in_pydantic_schema() -> None:
    """The schema accepts nested structured dtypes supported by the v2 specification."""
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    doc["dtype"] = [["outer", [["inner", "<i4"]]]]
    doc["fill_value"] = None
    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadataField)

    assert adapter.validate_python(doc).dtype.to_json() == (("outer", (("inner", "<i4"),)),)
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


def _assert_runtime_and_schema_reject(
    field_type: object, document: Mapping[str, JSONValue]
) -> None:
    adapter = TypeAdapter(field_type)
    with pytest.raises(ValidationError):
        adapter.validate_python(document)
    assert list(Draft202012Validator(adapter.json_schema()).iter_errors(document))


def test_v3_array_schema_rejects_empty_codecs() -> None:
    """The generated schema mirrors the runtime non-empty codec pipeline rule."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc["codecs"] = []

    _assert_runtime_and_schema_reject(zmp.ZarrV3ArrayMetadataField, doc)


def test_array_schemas_reject_negative_dimensions() -> None:
    """Both array schemas mirror the runtime non-negative dimension rule."""
    for field_type, source in (
        (zmp.ZarrV3ArrayMetadataField, V3_ARRAY_DOC),
        (zmp.ZarrV2ArrayMetadataField, V2_ARRAY_DOC),
    ):
        doc = json.loads(json.dumps(source))
        doc["shape"] = [-1]
        _assert_runtime_and_schema_reject(field_type, doc)


def test_v2_array_schema_allows_empty_filters() -> None:
    """The v2 schema, like the runtime, accepts an empty list, because the spec says "a list ... or null" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79)."""
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    doc["filters"] = []
    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadataField)

    assert adapter.validate_python(doc).filters == ()
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


@pytest.mark.parametrize(
    "field", ["data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"]
)
def test_v3_array_schema_rejects_false_at_every_extension_point(field: str) -> None:
    """No extension point of an array document can set `must_understand` to false, in the schema or at runtime."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    entry = {"name": "example", "must_understand": False}
    doc[field] = [entry] if field in ("codecs", "storage_transformers") else entry

    _assert_runtime_and_schema_reject(zmp.ZarrV3ArrayMetadataField, doc)


def test_metadata_field_schema_rejects_unknown_members() -> None:
    """Named-configuration envelopes are closed in both runtime and schema validation."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc["codecs"] = [{"name": "bytes", "unexpected": 1}]

    _assert_runtime_and_schema_reject(zmp.ZarrV3ArrayMetadataField, doc)


@pytest.mark.parametrize(
    ("field_type", "source"),
    [
        (zmp.ZarrV2GroupMetadataField, V2_GROUP_DOC),
        (zmp.ZarrV2ConsolidatedMetadataField, V2_CONSOLIDATED_DOC),
    ],
)
def test_v2_schema_rejects_unknown_document_members(
    field_type: object, source: dict[str, object]
) -> None:
    """Closed v2 merged documents expose their runtime boundary in JSON Schema."""
    doc = json.loads(json.dumps(source))
    doc["unexpected"] = 1

    _assert_runtime_and_schema_reject(field_type, doc)


def test_v2_array_schema_allows_unknown_document_members() -> None:
    """The v2 array document is open ("SHOULD be ignored", https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92) in both runtime and schema validation. An unknown member is kept as written."""
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    doc["unexpected"] = 1
    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadataField)

    assert adapter.validate_python(doc).to_json()["unexpected"] == 1
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


def test_v3_array_schema_allows_unknown_extension_fields() -> None:
    """Schema constraints do not close the v3 top-level extension namespace."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc["vendor_extension"] = {"anything": [1, 2]}
    adapter = TypeAdapter(zmp.ZarrV3ArrayMetadataField)

    held = adapter.validate_python(doc).extra_fields["vendor_extension"]
    assert held.to_json() == {"vendor_extension": {"anything": (1, 2)}}
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


def test_json_roundtrip() -> None:
    """model_dump_json output re-validates to an equal pydantic model."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadataField

    manifest = Manifest.model_validate({"metadata": V3_ARRAY_DOC})
    assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_core_package_does_not_import_pydantic() -> None:
    """Importing zarr_metadata (in a fresh interpreter) must not import
    pydantic: the integration is opt-in via zarr_metadata.pydantic."""
    import subprocess
    import sys

    code = "import sys, zarr_metadata; assert 'pydantic' not in sys.modules, 'leaked'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_non_finite_attribute_is_kept_by_constants_serialization() -> None:
    """Pydantic writes a non-finite number as `null` unless the enclosing
    model writes it the way the store does. The module docstring explains how."""

    class Manifest(BaseModel):
        model_config = ConfigDict(ser_json_inf_nan="constants")
        metadata: zmp.ZarrV3GroupMetadataField

    document = b'{"metadata": {"zarr_format": 3, "node_type": "group", "attributes": {"x": NaN}}}'
    written = Manifest.model_validate_json(document).model_dump_json()
    held = Manifest.model_validate_json(written).metadata.attributes["x"]
    assert isinstance(held, float)
    assert math.isnan(held)


# --- the scope a v3 field type reads in --------------------------------------

_WITH_ZSTD = ZarrV3ArrayMetadata.create_default(
    shape=(4,),
    codecs=({"name": "bytes"}, {"name": "zstd", "configuration": {"level": 3, "checksum": False}}),
).to_json()


@pytest.mark.parametrize(
    ("context", "read"),
    [
        (None, True),
        (CORE, False),
        ({zmp.CONTEXT_KEY: CORE}, False),
        ({"another validator's": 1}, True),
    ],
    ids=["none", "a-scope", "a-mapping-holding-one", "a-mapping-holding-none"],
)
def test_a_v3_field_type_reads_in_the_scope_the_validation_context_holds(
    context: object, read: bool
) -> None:
    """The field type reads in the scope from the validation context, which pydantic passes to every validator. `zstd` is an extension, so `CORE` leaves it unclaimed."""
    model = TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(_WITH_ZSTD, context=context)
    assert isinstance(model.codecs[1], AcceptedField) is read


class _Bound(BaseModel):
    """A model whose field is read in `CORE`, whatever the validation context holds."""

    metadata: Annotated[ZarrV3ArrayMetadata, zmp.InScope(CORE)]


def test_a_field_bound_to_a_scope_reads_in_that_scope() -> None:
    """`Annotated[ZarrV3ArrayMetadata, InScope(CORE)]` reads a document in `CORE`, whatever the validation context holds, and reads an instance again in `CORE`: an instance read in `CORE` with `zstd` unclaimed stays as it is. The field writes the document as written, and a type checker sees the model class."""
    for context in (None, CORE_AND_EXTENSIONS, {zmp.CONTEXT_KEY: CORE_AND_EXTENSIONS}):
        read = _Bound.model_validate({"metadata": _WITH_ZSTD}, context=context).metadata
        assert read.context == CORE
        assert isinstance(read.codecs[1], UnclaimedField)
    held = ZarrV3ArrayMetadata(_WITH_ZSTD, context=CORE)
    assert _Bound(metadata=held).metadata == held
    assert _Bound(metadata=held).model_dump()["metadata"] == held.to_json()


def test_error_a_field_bound_to_a_scope_refuses_an_instance_it_reads_differently() -> None:
    """An instance read where `zstd` is claimed would lose that meaning in `CORE`, so a field bound to `CORE` refuses it with a `ValidationError`, rather than passing it through in its own scope."""
    with pytest.raises(ValidationError):
        _Bound(metadata=ZarrV3ArrayMetadata(_WITH_ZSTD, context=CORE_AND_EXTENSIONS))


def test_error_a_field_bound_to_a_scope_of_the_other_format() -> None:
    """A field of a v3 model bound to a v2 scope is refused when pydantic builds its schema, as every v3 reader refuses a v2 scope."""
    with pytest.raises(TypeError, match="v3 document is read in a scope of that format"):
        TypeAdapter(Annotated[ZarrV3ArrayMetadata, zmp.InScope(CORE_V2)])


def test_error_a_scope_given_to_validation_reads_an_instance_in_it() -> None:
    """A scope given in the validation context reads an instance again in that scope, as a bound field does, so an instance that would lose a meaning there is refused. With no scope given, an instance passes through unchanged."""
    adapter = TypeAdapter(zmp.ZarrV3ArrayMetadataField)
    instance = ZarrV3ArrayMetadata(_WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    assert adapter.validate_python(instance) is instance
    with pytest.raises(ValidationError):
        adapter.validate_python(instance, context=CORE)


_READ_IN_SCOPE = [
    pytest.param(
        lambda value, context: TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(
            value, context=context
        ),
        id="validation-context",
    ),
    pytest.param(
        lambda value, context: TypeAdapter(
            Annotated[ZarrV3ArrayMetadata, zmp.InScope(context)]
        ).validate_python(value),
        id="in-scope",
    ),
]


@pytest.mark.parametrize("validate", _READ_IN_SCOPE)
def test_error_an_instance_a_scope_finds_problems_in_reports_each_one(validate: Any) -> None:  # noqa: ANN401 - a validating function
    """An instance read again in a scope that claims more can show problems its own scope did not check. pydantic gets one line error per problem, with the problem's kind as its type, at its location, and the value found there as its input, as for a document."""
    document: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
    document["codecs"] = (
        *document["codecs"],
        {"name": "gzip", "configuration": {"level": 99}},
        {"name": "zstd", "configuration": {"level": "x", "checksum": False}},
    )
    instance = ZarrV3ArrayMetadata(document, context=ZarrV3Context())
    with pytest.raises(ValidationError) as raised:
        validate(instance, CORE_AND_EXTENSIONS)
    assert [(line["type"], line["loc"], line["input"]) for line in raised.value.errors()] == [
        ("invalid_value", ("codecs", 1, "configuration", "level"), 99),
        ("invalid_type", ("codecs", 2, "configuration", "level"), "x"),
    ]


@pytest.mark.parametrize("validate", _READ_IN_SCOPE)
def test_error_an_instance_a_scope_reads_differently_reports_each_conflict(validate: Any) -> None:  # noqa: ANN401 - a validating function
    """An instance read where `zstd` is claimed would lose that meaning in `CORE`. pydantic gets one `scope_conflict` line error per conflict, at the field it is in, with that field as written as its input, and a message that names the codec."""
    instance = ZarrV3ArrayMetadata(_WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    with pytest.raises(ValidationError) as raised:
        validate(instance, CORE)
    (line,) = raised.value.errors()
    assert (line["type"], line["loc"]) == ("scope_conflict", ("codecs", 1))
    assert line["input"] == instance.to_json()["codecs"][1]
    assert "zstd" in line["msg"]


def test_error_in_scope_takes_a_scope() -> None:
    """`InScope` fixes a scope, so `None`, which elsewhere means the default scope, is a `TypeError` when it is built."""
    with pytest.raises(TypeError, match="InScope takes a ZarrV3Context or a ZarrV2Context"):
        zmp.InScope(None)  # pyright: ignore[reportArgumentType]


def test_error_a_validation_context_names_a_key_of_this_package_it_does_not_have() -> None:
    """A mapping validation context with a key that starts like this package's keys, but is not one of them, is a `TypeError`, so a misspelled key is not read as no scope at all."""
    with pytest.raises(TypeError, match="zarr_metadata_contxt"):
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(
            _WITH_ZSTD, context={"zarr_metadata_contxt": CORE}
        )


def test_error_a_validation_context_holds_a_scope_that_is_not_one() -> None:
    with pytest.raises(TypeError, match=zmp.CONTEXT_KEY):
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(
            _WITH_ZSTD, context={zmp.CONTEXT_KEY: "CORE"}
        )


def test_each_problem_is_a_line_error_of_pydantic_s() -> None:
    # Problems are reported the way pydantic reports its own errors: one line
    # error per problem, with the problem's kind as its type, at the problem's
    # loc under the field's loc, with the input found there and the expected
    # values. A message that contains JSON is kept unchanged.
    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadataField

    doc = {
        **V3_ARRAY_DOC,
        "fill_value": {"a": 1},
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 12}},
        ],
    }
    with pytest.raises(ValidationError) as raised:
        Manifest.model_validate({"metadata": doc})
    errors = raised.value.errors()
    assert [(e["type"], e["loc"], e["input"]) for e in errors] == [
        ("invalid_type", ("metadata", "fill_value"), {"a": 1}),
        ("invalid_value", ("metadata", "codecs", 1, "configuration", "level"), 12),
    ]
    assert errors[0]["msg"] == 'expected an integer, got {"a": 1}'
    assert errors[1].get("ctx") == {"ge": 0, "le": 9}


def test_a_message_holding_a_ctx_placeholder_is_reported_as_it_is() -> None:
    # pydantic renders a message as a template over its ctx, with no way to
    # escape a placeholder. A document value written as `{expected}` would be
    # shown as the ctx's `expected`. So such a message is passed whole in the
    # ctx instead.
    doc = {**V3_ARRAY_DOC, "zarr_format": "{expected}"}
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(doc)
    (error,) = raised.value.errors()
    assert (error["type"], error["loc"], error["msg"]) == (
        "invalid_type",
        ("zarr_format",),
        'expected 3, got "{expected}"',
    )
    assert error.get("ctx") == {"expected": (3,), "message": 'expected 3, got "{expected}"'}


def test_a_ctx_member_named_message_yields_to_a_message_holding_a_placeholder() -> None:
    # The `message` key that carries the message goes last in the ctx, so no
    # ctx key after it is substituted into the carried text. When no message
    # needs to be carried, a ctx member named `message` is kept.
    ctx = {"message": "theirs", "expected": 3}
    problem = ValidationProblem(("a",), 'got "{expected}"', "invalid_value", ctx=ctx)
    error = ValidationError.from_exception_data("T", [zmp._line_error(problem, {"a": 1})]).errors()[
        0
    ]
    assert error["msg"] == 'got "{expected}"'
    assert error.get("ctx") == {"expected": 3, "message": 'got "{expected}"'}
    assert list(error.get("ctx", {})) == ["expected", "message"]
    plain = ValidationProblem(("a",), "got 1", "invalid_value", ctx=ctx)
    error = ValidationError.from_exception_data("T", [zmp._line_error(plain, {"a": 1})]).errors()[0]
    assert (error["msg"], error.get("ctx")) == ("got 1", ctx)


def test_a_line_error_for_what_a_problem_could_not_hold_reports_what_sits_there() -> None:
    # A problem holds no input when the value is not JSON that a reader walks
    # (here, an `AcceptedField` built by hand and placed among the codecs). The
    # line error then reports that object. For a missing key, the line error
    # reports the object that lacks the key.
    smuggled = AcceptedField(
        json="gzip", name="gzip", definition=GZIP_CODEC, configuration={"level": 1}
    )
    doc = {
        **V3_ARRAY_DOC,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, smuggled],
    }
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(doc)
    (error,) = raised.value.errors()
    assert (error["type"], error["loc"]) == ("invalid_type", ("codecs", 1))
    assert error["input"] is smuggled


def test_the_pydantic_schema_refuses_a_null_consolidated_metadata_as_the_reader_does() -> None:
    # The schema and the reader give the same result for `null` here: both refuse it.
    schema = TypeAdapter(zmp.ZarrV3GroupMetadataField).json_schema()
    document = {"zarr_format": 3, "node_type": "group", "consolidated_metadata": None}
    assert not Draft202012Validator(schema).is_valid(document)
    with pytest.raises(ValidationError):
        TypeAdapter(zmp.ZarrV3GroupMetadataField).validate_python(document)


def test_a_line_error_for_a_document_past_the_cap_renders() -> None:
    # The depth problem holds no input. The line error reports the subtree
    # at its loc. pydantic renders it like any other input: truncated in
    # `str` and whole in `json`.
    deep: dict[str, object] = {}
    for _ in range(2_000):
        deep = {"a": deep}
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(
            {**V3_ARRAY_DOC, "attributes": deep}
        )
    (error,) = raised.value.errors()
    assert (error["type"], len(error["loc"])) == ("invalid_value", 256)
    assert "nested deeper" in str(raised.value)
    assert json.loads(raised.value.json())[0]["type"] == "invalid_value"


@pytest.mark.parametrize("field", ["Foo/bar", "", "r*", {"name": "Int8"}])
def test_the_pydantic_schema_names_an_extension_as_the_reader_does(field: object) -> None:
    # The schema that pydantic generates checks a name the same way the reader does.
    schema = TypeAdapter(zmp.ZarrV3ArrayMetadataField).json_schema()
    # Use arrays as lists, as JSON parses them, because `jsonschema` does not accept a tuple.
    document: dict[str, Any] = json.loads(json.dumps(V3_ARRAY_DOC))
    assert Draft202012Validator(schema).is_valid(document)
    named: dict[str, Any] = {**document, "data_type": field}
    assert not Draft202012Validator(schema).is_valid(named)
    with pytest.raises(ValidationError):
        TypeAdapter(zmp.ZarrV3ArrayMetadataField).validate_python(named)


def test_v2_field_types_read_in_the_validation_contexts_scope() -> None:
    """The v2 field types read a document in the scope that the validation context holds. The context is either a `Context` itself or holds one in its `zarr_metadata_context_v2` item. When it holds none, they read in `CORE_V2`. The v3 field types work the same way."""
    from zarr_metadata.definition import UnclaimedField
    from zarr_metadata.v2.data_type.scalar import UINT_DATA_TYPE_V2
    from zarr_metadata.v2.definition import CORE_V2, ZarrV2Context

    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadataField)
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    assert adapter.validate_python(doc).context is CORE_V2
    small = ZarrV2Context(data_types=[UINT_DATA_TYPE_V2])
    read = adapter.validate_python({**doc, "compressor": {"id": "zlib"}}, context=small)
    assert read.context is small
    assert isinstance(read.compressor, UnclaimedField)
    held = adapter.validate_python(doc, context={"zarr_metadata_context_v2": small})
    assert held.context is small
    group = TypeAdapter(zmp.ZarrV2GroupMetadataField).validate_python(V2_GROUP_DOC, context=small)
    assert group.context is small


def test_each_format_reads_in_its_own_context_key() -> None:
    """A mapping context gives each format's scope under its own key: `zarr_metadata_context` for v3 and `zarr_metadata_context_v2` for v2. So a v3 scope given for the v3 fields leaves the v2 fields in `CORE_V2`. A bare `Context` is the scope for every field type."""
    from zarr_metadata.definition import UnclaimedField
    from zarr_metadata.v2.data_type.scalar import UINT_DATA_TYPE_V2
    from zarr_metadata.v2.definition import CORE_V2, ZarrV2Context
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadataField)
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    small = ZarrV2Context(data_types=[UINT_DATA_TYPE_V2])
    v3_only = adapter.validate_python(doc, context={"zarr_metadata_context": CORE_AND_EXTENSIONS})
    assert v3_only.context is CORE_V2
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {**doc, "fill_value": "garbage"}, context={"zarr_metadata_context": CORE_AND_EXTENSIONS}
        )
    own = adapter.validate_python(doc, context={"zarr_metadata_context_v2": small})
    assert own.context is small
    bare = adapter.validate_python({**doc, "compressor": {"id": "zlib"}}, context=small)
    assert bare.context is small
    assert isinstance(bare.compressor, UnclaimedField)
    assert zmp.CONTEXT_KEY_V2 == "zarr_metadata_context_v2"


_TESTS = Path(__file__).parent.parent
_NODE_SCHEMA_V3 = node_metadata_json_schema_v3(context=ZarrV3Context())
_SCHEMA_PAIRS: dict[str, tuple[object, Mapping[str, Any], str]] = {
    "v3-array": (zmp.ZarrV3ArrayMetadataField, _NODE_SCHEMA_V3, "v3/array"),
    "v3-group": (zmp.ZarrV3GroupMetadataField, _NODE_SCHEMA_V3, "v3/group"),
    "v2-array": (zmp.ZarrV2ArrayMetadataField, json_schema(ZarrV2ArrayMetadataJSON), "v2/array"),
    "v2-group": (zmp.ZarrV2GroupMetadataField, json_schema(ZarrV2GroupMetadataJSON), "v2/group"),
}


def _variants(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The document, and copies of it that break one rule each."""
    yield document
    for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
        if isinstance(document.get(key), dict):
            yield {**document, key: {**document[key], "name": "Not A Name"}}
            yield {**document, key: {**document[key], "must_understand": False}}
    if "codecs" in document:
        yield {**document, "codecs": []}
    for key in ("shape", "chunks"):
        if key in document:
            yield {**document, key: [-1] * len(document[key])}
    yield {**document, "zarr_format": 7}


@pytest.mark.parametrize(
    ("field_type", "schema", "corpus"), list(_SCHEMA_PAIRS.values()), ids=list(_SCHEMA_PAIRS)
)
def test_the_pydantic_schema_agrees_with_the_package_schema(
    field_type: object, schema: Mapping[str, Any], corpus: str
) -> None:
    """Pydantic needs its own copies of the document types to write a JSON Schema, because it cannot embed a schema with its own `$defs`. Each fixture document, and each copy of it that breaks one rule, is accepted by the pydantic field type's schema exactly when `zarr_metadata.json_schema` accepts it, so the copies cannot drift."""
    ours = Draft202012Validator(schema)
    theirs = Draft202012Validator(TypeAdapter(field_type).json_schema())
    documents = sorted((_TESTS / corpus).glob("*.json"))
    assert documents != []
    for path in documents:
        for document in _variants(json.loads(path.read_text())):
            assert theirs.is_valid(document) == ours.is_valid(document), (path.name, document)
