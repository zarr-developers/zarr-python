"""Where an extension's rules are located: in the document as it is written, whatever the rules yield."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from zarr_metadata import ValidationProblem
from zarr_metadata.definition import RefusedField, resolve
from zarr_metadata.v2.definition import CORE_V2_CONTEXT, ZarrV2DataTypeDefinition
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import CORE_CONTEXT, CodecDefinition, DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr_metadata.definition import Nested
    from zarr_metadata.typed_json import Loc


@dataclasses.dataclass(frozen=True)
class _UnknownKeyAt:
    """Rules that yield one `unknown_key` at `loc`, wherever an extension author puts it."""

    loc: Loc

    def __call__(
        self, configuration: Mapping[str, object], nested: Nested
    ) -> Iterator[ValidationProblem]:
        yield ValidationProblem(self.loc, "unexpected", "unknown_key")


def _wants_an_endian(
    configuration: Mapping[str, object], nested: Nested
) -> Iterator[ValidationProblem]:
    """Rules that always want an `endian`, and find a problem with the configuration as a whole."""
    if "endian" not in configuration:
        yield ValidationProblem(("endian",), "expected an endian", "missing_key")
    yield ValidationProblem((), "the whole configuration", "invalid_value")


@pytest.mark.parametrize("loc", [(), (0,), ("extra",)], ids=["root", "index", "key"])
def test_error_a_rule_of_a_name_that_encodes_its_configuration_is_a_wrong_value_of_the_name(
    loc: Loc,
) -> None:
    """A name such as `r16` or `<f4` encodes its configuration, so each problem its definition's rules find is a wrong value of the name, at the field, whatever kind and location the rules give it."""
    rules = _UnknownKeyAt(loc)
    v3 = CORE_CONTEXT.update(
        data_types={
            **CORE_CONTEXT.data_types,
            "r*": dataclasses.replace(RAW_BYTES_DATA_TYPE, rules=rules),
        }
    )
    v2 = CORE_V2_CONTEXT.update(
        data_types={
            **CORE_V2_CONTEXT.data_types,
            "float": dataclasses.replace(CORE_V2_CONTEXT.data_types["float"], rules=rules),
        }
    )
    for name, kind, context, at in [
        ("r16", DataTypeDefinition, v3, ("data_type",)),
        ("<f4", ZarrV2DataTypeDefinition, v2, ("dtype",)),
    ]:
        resolved, problems = resolve(name, kind, context, at)
        assert isinstance(resolved, RefusedField)
        assert [(p.loc, p.kind) for p in problems] == [(at, "invalid_value")]


@pytest.mark.parametrize(
    ("codec", "no_endian", "whole"),
    [
        ("bytes", ("codecs", 0, "configuration"), ("codecs", 0)),
        ({"name": "bytes"}, ("codecs", 0, "configuration"), ("codecs", 0)),
        (
            {"name": "bytes", "configuration": {}},
            ("codecs", 0, "configuration", "endian"),
            ("codecs", 0, "configuration"),
        ),
    ],
    ids=["name", "object", "empty-configuration"],
)
def test_error_a_rule_of_a_codec_is_located_where_its_document_writes_it(
    codec: object, no_endian: Loc, whole: Loc
) -> None:
    """A codec that writes no configuration has an empty one the document does not hold. A member its rules find missing is the configuration itself, missing, and a problem with the configuration as a whole is the codec's."""
    context = CORE_CONTEXT.update(
        codecs={
            **CORE_CONTEXT.codecs,
            "bytes": dataclasses.replace(BYTES_CODEC, rules=_wants_an_endian),
        }
    )
    _, problems = resolve(codec, CodecDefinition, context, ("codecs", 0))
    assert [(p.loc, p.kind) for p in problems] == [
        (no_endian, "missing_key"),
        (whole, "invalid_value"),
    ]
