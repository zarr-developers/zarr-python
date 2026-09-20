"""Tests for rule registration.

The registry exists so that a rule cannot be defined without being run.
These tests cover the three ways that could still fail: a rule declaring
dependencies no such document has, an entity whose rules were never
imported, and a document rule set assembled from something other than
the registry.
"""

from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

import pytest

import zarr_metadata.rules._entities as entities
from zarr_metadata.rules import (
    ZARR_V2_ARRAY_RULES,
    ZARR_V3_ARRAY_RULES,
    ZARR_V3_GROUP_RULES,
    Rule,
)
from zarr_metadata.rules._registry import (
    dispatched_fields,
    document_rule,
    entity_rule,
    register_document_type,
    registered_entities,
)
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    RAW_BYTES_FAMILY,
)
from zarr_metadata.v3._shape import modelled_entities

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.rules._spec import ArrayParts
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME

# Entities the package models that carry no *composition* rule — nothing
# about them depends on the document or on the codec chain. Several still
# have value constraints (`blosc`'s clevel range, `gzip`'s and `zstd`'s
# level ranges); those are refinements of the type and live with it in
# `v3._shape`, not here. Listed by hand, keyed by extension point, so that
# adding a codec is a deliberate choice between "write a rule" and "record
# that composition says nothing", never a silent omission.
_RULE_FREE = frozenset(
    {
        (CODECS, "blosc"),
        (CODECS, "crc32c"),
        (CODECS, "gzip"),
        (CODECS, "scale_offset"),
        (CODECS, "zstd"),
        (CHUNK_KEY_ENCODING, "default"),
        (CHUNK_KEY_ENCODING, "v2"),
        (DATA_TYPE, "bool"),
        (DATA_TYPE, "int8"),
        (DATA_TYPE, "int16"),
        (DATA_TYPE, "int32"),
        (DATA_TYPE, "int64"),
        (DATA_TYPE, "uint8"),
        (DATA_TYPE, "uint16"),
        (DATA_TYPE, "uint32"),
        (DATA_TYPE, "uint64"),
        (DATA_TYPE, "float16"),
        (DATA_TYPE, "float32"),
        (DATA_TYPE, "float64"),
        (DATA_TYPE, "complex64"),
        (DATA_TYPE, "complex128"),
        (DATA_TYPE, RAW_BYTES_FAMILY),
        (DATA_TYPE, "bytes"),
        (DATA_TYPE, "string"),
    }
)


def test_every_shape_modelled_entity_is_accounted_for() -> None:
    # Every shape-modelled entity either
    # carries rules or is recorded as deliberately rule-free.
    assert modelled_entities() == registered_entities() | _RULE_FREE


def test_every_shape_modelled_field_has_a_dispatcher() -> None:
    # Regression: shapes existed for four extension points but dispatchers
    # for only two, so `entity_rule` accepted registrations at `data_type`
    # and `chunk_key_encoding` whose rules then silently never ran — the
    # exact silent-pass failure the registry exists to prevent. A rule can
    # only fire at a field something dispatches.
    shape_modelled = {field for field, _ in modelled_entities()}
    assert shape_modelled <= dispatched_fields()


def test_every_field_with_rules_has_a_dispatcher() -> None:
    assert {field for field, _ in registered_entities()} <= dispatched_fields()


def test_rule_free_entities_really_have_no_rules() -> None:
    # Guards the exclusion list itself: an entity cannot be listed as
    # rule-free while quietly carrying rules.
    assert registered_entities() & _RULE_FREE == frozenset()


def test_rules_are_keyed_by_extension_point_not_name() -> None:
    # `bytes` is a core codec and a registered extension data type; a rule
    # for one must never fire on the other, so the key carries the field.
    assert {(CODECS, "bytes"), (DATA_TYPE, "bytes")} <= modelled_entities()
    assert (CODECS, "bytes") in registered_entities()
    assert (DATA_TYPE, "bytes") not in registered_entities()


def test_every_entity_module_is_imported() -> None:
    # The package auto-imports its modules; this asserts the discovery
    # actually ran, so a new module cannot sit unimported and inert.
    module_names = {info.name for info in pkgutil.iter_modules(entities.__path__)}
    assert len(module_names) != 0
    for name in module_names:
        assert f"{entities.__name__}.{name}" in __import__("sys").modules


@pytest.mark.parametrize("rules", [ZARR_V3_ARRAY_RULES, ZARR_V2_ARRAY_RULES, ZARR_V3_GROUP_RULES])
def test_rule_sets_are_non_empty(rules: tuple[object, ...]) -> None:
    assert len(rules) != 0


def test_error_document_rule_requiring_an_unknown_key() -> None:
    # A rule whose dependency is misspelled can never fire, and a rule
    # that never fires is indistinguishable from one that always passes.
    with pytest.raises(ValueError, match="could never fire"):

        @document_rule(ZARR_V3_ARRAY, frozenset({"shapee"}))
        def _misspelled(document: object) -> tuple[()]:  # pragma: no cover - never runs
            return ()


def test_error_entity_rule_requiring_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="could never fire"):

        @entity_rule(ZARR_V3_ARRAY, CHUNK_GRID, "regular", requires=frozenset({"shapee"}))
        def _misspelled(
            configuration: Mapping[str, object],
            document: Mapping[str, object],
            incoming: ArrayParts | None,
        ) -> tuple[()]:  # pragma: no cover - refused at registration
            return ()


def test_error_entity_rule_for_an_unmodelled_entity() -> None:
    # Entity rules read configuration members by name, so a rule for an
    # entity with no shape validator could never fire.
    with pytest.raises(ValueError, match="no shape validator"):

        @entity_rule(ZARR_V3_ARRAY, CHUNK_GRID, "hilbert")
        def _unmodelled(
            configuration: Mapping[str, object],
            document: Mapping[str, object],
            incoming: ArrayParts | None,
        ) -> tuple[()]:  # pragma: no cover - refused at registration
            return ()


def test_error_entity_rule_for_name_modelled_only_at_another_extension_point() -> None:
    # `regular` has a chunk-grid shape, but no codec shape. Name-only lookup
    # would accept this registration and later interpret codec metadata using
    # the chunk-grid schema.
    with pytest.raises(ValueError, match="no shape validator"):

        @entity_rule(ZARR_V3_ARRAY, CODECS, "regular")
        def _wrong_extension_point(
            configuration: Mapping[str, object],
            document: Mapping[str, object],
            incoming: ArrayParts | None,
        ) -> tuple[()]:  # pragma: no cover - refused at registration
            return ()


def test_error_rule_for_an_unregistered_document_type() -> None:
    with pytest.raises(LookupError, match="unknown document type"):

        @document_rule("zarr_v9_array", frozenset())
        def _orphan(document: object) -> tuple[()]:  # pragma: no cover - never runs
            return ()


def test_register_document_type_accepts_declared_extension_keys() -> None:
    register_document_type("test_doc", frozenset({"a"}), extension_keys=frozenset({"b"}))

    @document_rule("test_doc", frozenset({"a", "b"}))
    def _uses_both(document: object) -> tuple[()]:
        return ()

    assert _uses_both.requires == frozenset({"a", "b"})


def test_error_entity_rule_reads_an_unmodelled_member() -> None:
    with pytest.raises(ValueError, match="does not model"):

        @entity_rule(ZARR_V3_ARRAY, CODECS, GZIP_CODEC_NAME, reads=frozenset({"nosuchmember"}))
        def _unmodelled_member(
            configuration: Mapping[str, object],
            document: Mapping[str, object],
            incoming: ArrayParts | None,
        ) -> tuple[ValidationProblem, ...]:  # pragma: no cover - never registered
            return ()


def test_error_entity_rule_reads_an_optional_member() -> None:
    # Only a required member is safe to subscript: an absent optional one is
    # reported by nothing, so the rule would raise out of a validator.
    with pytest.raises(ValueError, match="reads_optional"):

        @entity_rule(ZARR_V3_ARRAY, CODECS, BYTES_CODEC_NAME, reads=frozenset({"endian"}))
        def _subscripts_an_optional_member(
            configuration: Mapping[str, object],
            document: Mapping[str, object],
            incoming: ArrayParts | None,
        ) -> tuple[ValidationProblem, ...]:  # pragma: no cover - never registered
            return ()


# Every document rule, by the layer it belongs to. A field rule reads one
# top-level field; a composition rule spans several. Listed by hand so that
# adding one is a deliberate choice, the way `_RULE_FREE` makes "this entity
# has no composition rule" a deliberate choice.
_FIELD_RULES = frozenset(
    {
        "_check_data_type_spelling",
        "_check_data_type_shape",
        "_check_chunk_key_encoding_shape",
        "_check_chunk_grid_shape",
        "check_codec_pipeline_order",
        "check_codec_shapes",
        "check_chunk_grid_shape",
        "_dispatch_chunk_grid_entity_rules",
        "_dispatch_data_type_entity_rules",
        "_dispatch_chunk_key_encoding_entity_rules",
        "_dispatch_codecs_entity_rules",
        "check_consolidated_entries",
    }
)
_COMPOSITION_RULES = frozenset(
    {"_check_fill_matches_dtype", "check_dimension_names_length", "check_chunks_match_shape"}
)


@pytest.mark.parametrize(
    "rules",
    [ZARR_V3_ARRAY_RULES, ZARR_V2_ARRAY_RULES, ZARR_V3_GROUP_RULES],
    ids=["v3-array", "v2-array", "v3-group"],
)
def test_document_rules_are_classified_by_what_they_read(rules: tuple[Rule, ...]) -> None:
    # The classification is not decoration: a rule reading one field is a
    # value constraint on that field, and could in principle move down to
    # the layer that owns the field. One spanning fields cannot.
    for rule in rules:
        name = rule.check.__name__
        assert name in _FIELD_RULES | _COMPOSITION_RULES, f"{name} is classified nowhere"
        if name in _FIELD_RULES:
            assert len(rule.requires) == 1, (
                f"{name} is a field rule but reads {sorted(rule.requires)}"
            )
        else:
            assert len(rule.requires) >= 2, f"{name} is a composition rule but reads one field"
