"""Test that the curated front-door names are accessible from the top-level zarr_metadata package."""

import importlib
import pkgutil
import re
from typing import Literal, get_args, get_origin

import zarr_metadata as zm


def _group_rank(s: str) -> int:
    """RUF022 groups `__all__` as: SCREAMING_SNAKE (0), then TitleCase (1), then dunders (2).

    The exact intra-group ordering is ruff's own natural sort and is enforced by
    ruff itself (pre-commit + CI); this test only asserts the grouping, not the
    fragile tie-breaking, so it can't drift out of sync with ruff's implementation.
    """
    if s.startswith("__") and s.endswith("__"):
        return 2
    stripped = re.sub(r"[\d_]", "", s)
    return 0 if stripped.isupper() else 1


EXPECTED = [
    # Category A — metadata-document types
    "ZarrV2ArrayMetadataJSON",
    "ZarrV2ArrayMetadataJSONPartial",
    "ZarrV2ZArrayJSON",
    "ZarrV2GroupMetadataJSON",
    "ZarrV2GroupMetadataJSONPartial",
    "ZarrV2ZGroupJSON",
    "ZarrV2ConsolidatedMetadataJSON",
    "ZarrV2ZAttrsJSON",
    "ZarrV2CodecMetadata",
    "ZarrV3ArrayMetadataJSON",
    "ZarrV3ArrayMetadataJSONPartial",
    "ZarrV3ExtensionField",
    "ZarrV3GroupMetadataJSON",
    "ZarrV3GroupMetadataJSONPartial",
    "ZarrV3ConsolidatedMetadataJSON",
    "ZarrV3NamedConfigJSON",
    "ZarrV3MetadataFieldJSON",
    "JSONValue",
    # Category A' — metadata models (in-memory dataclasses over the documents)
    "ZarrV2ArrayMetadata",
    "ZarrV2ArrayMetadataPartial",
    "ZarrV3ArrayMetadata",
    "ZarrV3ArrayMetadataPartial",
    "ZarrV2GroupMetadata",
    "ZarrV2GroupMetadataPartial",
    "ZarrV3GroupMetadata",
    "ZarrV3GroupMetadataPartial",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3NamedConfig",
    "ZarrV3MetadataField",
    "ValidationProblem",
    "MetadataValidationError",
    "ProblemKind",
    "UNSET",
    # Store keys — the names the documents are persisted under. Defined in the
    # v2/v3 spec modules, re-exported through `zarr_metadata.model`.
    "ZARR_V2_ARRAY_METADATA_STORE_KEY",
    "ZarrV2ArrayMetadataStoreKey",
    "ZARR_V2_GROUP_METADATA_STORE_KEY",
    "ZarrV2GroupMetadataStoreKey",
    "ZARR_V2_ATTRIBUTES_STORE_KEY",
    "ZarrV2AttributesStoreKey",
    "ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY",
    "ZarrV2ConsolidatedMetadataStoreKey",
    "ZARR_V3_ARRAY_METADATA_STORE_KEY",
    "ZarrV3ArrayMetadataStoreKey",
    "ZARR_V3_GROUP_METADATA_STORE_KEY",
    "ZarrV3GroupMetadataStoreKey",
    # Not a store key: v3 consolidated metadata is embedded in the group's own
    # `zarr.json`, so it has no paired Literal alias.
    "ZARR_V3_CONSOLIDATED_METADATA_KEY",
    # v2 data-type encoding union
    "ZarrV2DataTypeMetadata",
    # Category B — codec canonical unions
    "BloscCodecMetadata",
    "BytesCodecMetadata",
    "CastValueCodecMetadata",
    "Crc32cCodecMetadata",
    "GzipCodecMetadata",
    "ScaleOffsetCodecMetadata",
    "ShardingIndexedCodecMetadata",
    "TransposeCodecMetadata",
    "ZstdCodecMetadata",
    # Category C — grid/key canonical unions
    "RegularChunkGridMetadata",
    "RectilinearChunkGridMetadata",
    "DefaultChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingMetadata",
    # Category D — dtype trios
    # bool
    "BoolDataTypeName",
    "BOOL_DATA_TYPE_NAME",
    "BoolFillValue",
    # int8/16/32/64
    "Int8DataTypeName",
    "INT8_DATA_TYPE_NAME",
    "Int8FillValue",
    "Int16DataTypeName",
    "INT16_DATA_TYPE_NAME",
    "Int16FillValue",
    "Int32DataTypeName",
    "INT32_DATA_TYPE_NAME",
    "Int32FillValue",
    "Int64DataTypeName",
    "INT64_DATA_TYPE_NAME",
    "Int64FillValue",
    # uint8/16/32/64 (actual casing is Uint, not UInt)
    "Uint8DataTypeName",
    "UINT8_DATA_TYPE_NAME",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "UINT16_DATA_TYPE_NAME",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "UINT32_DATA_TYPE_NAME",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "UINT64_DATA_TYPE_NAME",
    "Uint64FillValue",
    # float16/32/64
    "Float16DataTypeName",
    "FLOAT16_DATA_TYPE_NAME",
    "Float16FillValue",
    "Float32DataTypeName",
    "FLOAT32_DATA_TYPE_NAME",
    "Float32FillValue",
    "Float64DataTypeName",
    "FLOAT64_DATA_TYPE_NAME",
    "Float64FillValue",
    # complex64/128
    "Complex64DataTypeName",
    "COMPLEX64_DATA_TYPE_NAME",
    "Complex64FillValue",
    "Complex128DataTypeName",
    "COMPLEX128_DATA_TYPE_NAME",
    "Complex128FillValue",
    # bytes
    "BytesDataTypeName",
    "BYTES_DATA_TYPE_NAME",
    "BytesFillValue",
    # string
    "StringDataTypeName",
    "STRING_DATA_TYPE_NAME",
    "StringFillValue",
    # numpy_datetime64
    "NumpyDatetime64DataTypeName",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NumpyDatetime64FillValue",
    # numpy_timedelta64
    "NumpyTimedelta64DataTypeName",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NumpyTimedelta64FillValue",
    # struct
    "StructDataTypeName",
    "STRUCT_DATA_TYPE_NAME",
    "StructFillValue",
    # raw (no _DATA_TYPE_NAME constant)
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    # Category E — constant+Literal pairs
    "ZARR_V2_ARRAY_ORDER",
    "ZarrV2ArrayOrder",
    "ZARR_V2_ARRAY_DIMENSION_SEPARATOR",
    "ZarrV2ArrayDimensionSeparator",
    "ENDIANNESS",
    "Endianness",
    "BYTES_CODEC_NAME",
    "BytesCodecName",
    "BLOSC_CODEC_NAME",
    "BloscCodecName",
    "BLOSC_CNAME",
    "BloscCName",
    "BLOSC_SHUFFLE",
    "BloscShuffle",
    "CAST_ROUNDING_MODE",
    "CastRoundingMode",
    "CAST_OUT_OF_RANGE_MODE",
    "CastOutOfRangeMode",
    "CAST_VALUE_CODEC_NAME",
    "CastValueCodecName",
    "CRC32C_CODEC_NAME",
    "Crc32cCodecName",
    "GZIP_CODEC_NAME",
    "GzipCodecName",
    "SCALE_OFFSET_CODEC_NAME",
    "ScaleOffsetCodecName",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "SHARDING_INDEXED_CODEC_NAME",
    "ShardingIndexedCodecName",
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodecName",
    "ZSTD_CODEC_NAME",
    "ZstdCodecName",
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGridName",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridName",
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DefaultChunkKeyEncodingName",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncodingSeparator",
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2ChunkKeyEncodingName",
    "V2_CHUNK_KEY_ENCODING_SEPARATOR",
    "V2ChunkKeyEncodingSeparator",
    "NUMPY_TIME_UNIT",
    "NumpyTimeUnit",
]


def test_front_door_names_public() -> None:
    missing = [n for n in EXPECTED if n not in zm.__all__ or not hasattr(zm, n)]
    assert not missing, f"missing from top-level API: {missing}"


def test_front_door_is_exactly_expected() -> None:
    """`__all__` must contain exactly the curated names (plus `__version__`).

    Guards against a name being promoted to the front door without a
    corresponding, deliberate entry in `EXPECTED` — i.e. an accidental
    addition to the public API surface.
    """
    assert set(zm.__all__) - {"__version__"} == set(EXPECTED)


def test_all_is_grouped_and_unique() -> None:
    ranks = [_group_rank(n) for n in zm.__all__]
    assert ranks == sorted(ranks), "`__all__` groups out of order (SCREAMING, TitleCase, dunder)"
    assert len(zm.__all__) == len(set(zm.__all__))


# --- naming grammar ---------------------------------------------------------

# Core document/model names: the format version comes first (`ZarrV2` /
# `ZarrV3`), then the CamelCase entity, then an optional role suffix
# (`JSON`, `JSONPartial`, `Partial`, `StoreKey`) — validated loosely here
# because `JSON` decomposes into single-letter words under any strict
# word-splitting regex.
_CORE_NAME = re.compile(r"^ZarrV[23](?:[A-Z][a-z0-9]*)+$")

# Zarr v3 extension-entity names: the registered entity comes first (`Blosc`,
# `Uint8`, ... — `V2` here is the *entity name* of the v2-compatibility chunk
# key encoding, not a format-version marker, which is always spelled
# `ZarrV2`/`ZarrV3`), followed by exactly one role suffix.
_EXTENSION_ROLES = (
    "CodecConfiguration",
    "CodecMetadata",
    "CodecName",
    "CodecObject",
    "ChunkGridConfiguration",
    "ChunkGridMetadata",
    "ChunkGridName",
    "ChunkGridObject",
    "ChunkKeyEncodingConfiguration",
    "ChunkKeyEncodingMetadata",
    "ChunkKeyEncodingName",
    "ChunkKeyEncodingObject",
    "ChunkKeyEncodingSeparator",
    "DataTypeName",
    "FillValue",
    "Configuration",
    "Component",
)
_EXTENSION_NAME = re.compile(r"^(?:[A-Z][a-z0-9]*)+?(?:" + "|".join(_EXTENSION_ROLES) + r")$")

# Standalone vocabulary: scalar Literal aliases, structural helper shapes, and
# the validation diagnostics. Closed by hand — a new name belongs here only if
# it is genuinely role-less; anything document- or entity-shaped must fit the
# grammars above instead.
_STANDALONE_VOCAB = frozenset(
    {
        "Base64Bytes",
        "BloscCName",
        "BloscShuffle",
        "CastOutOfRangeMode",
        "CastRoundingMode",
        "CodecKind",
        "Endianness",
        "HexFloat16",
        "HexFloat32",
        "HexFloat64",
        "JSONValue",
        "Invalid",
        "MetadataValidationError",
        "NumpyDatetime64",
        "NumpyTimeUnit",
        "NumpyTimedelta64",
        "ProblemKind",
        "RectilinearDimSpec",
        "Rule",
        "RuleCheck",
        "ScalarMap",
        "ScalarMapEntry",
        "ShardingIndexLocation",
        "Struct",
        "StructField",
        "Valid",
        "ValidationProblem",
        "ValidationResult",
    }
)


def _iter_module_names() -> set[str]:
    """Every public module in the package, including the top-level namespace."""
    module_names = {"zarr_metadata"}
    for info in pkgutil.walk_packages(zm.__path__, prefix="zarr_metadata."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            module_names.add(info.name)
    return module_names


def _public_type_names() -> set[tuple[str, str]]:
    """Every (module, CamelCase name) pair exported via a public `__all__`."""
    out: set[tuple[str, str]] = set()
    for module_name in _iter_module_names():
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            if name.startswith("_") or name.isupper() or name.islower():
                continue
            out.add((module_name, name))
    return out


def test_public_type_names_comply_with_naming_grammar() -> None:
    """Every public type name parses against the package naming grammar:
    version-first core names, entity-plus-role extension names, or the closed
    standalone vocabulary."""
    exported = _public_type_names()
    violations = [
        f"{module}.{name}"
        for module, name in sorted(exported)
        if name not in _STANDALONE_VOCAB
        and not _CORE_NAME.match(name)
        and not _EXTENSION_NAME.match(name)
    ]
    assert not violations, f"names outside the naming grammar: {violations}"


def test_standalone_vocab_is_not_stale() -> None:
    """Every allowlisted vocabulary name is still actually exported."""
    exported_names = {name for _, name in _public_type_names()}
    assert exported_names >= _STANDALONE_VOCAB


def test_promoted_pairs_drift() -> None:
    """Each promoted runtime constant holds exactly the values of the `Literal`
    type it manifests, so the two cannot drift apart."""
    pairs = [
        (zm.ENDIANNESS, zm.Endianness),
        (zm.BLOSC_CNAME, zm.BloscCName),
        (zm.BLOSC_SHUFFLE, zm.BloscShuffle),
        (zm.SHARDING_INDEX_LOCATION, zm.ShardingIndexLocation),
        (zm.NUMPY_TIME_UNIT, zm.NumpyTimeUnit),
        (zm.CAST_ROUNDING_MODE, zm.CastRoundingMode),
        (zm.CAST_OUT_OF_RANGE_MODE, zm.CastOutOfRangeMode),
        (zm.ZARR_V2_ARRAY_ORDER, zm.ZarrV2ArrayOrder),
        (zm.ZARR_V2_ARRAY_DIMENSION_SEPARATOR, zm.ZarrV2ArrayDimensionSeparator),
        (zm.DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR, zm.DefaultChunkKeyEncodingSeparator),
        (zm.V2_CHUNK_KEY_ENCODING_SEPARATOR, zm.V2ChunkKeyEncodingSeparator),
    ]
    for const, lit in pairs:
        assert set(const) == set(get_args(lit))


def constant_name_for(type_name: str) -> str:
    """Derive a constant's name from the name of the type it manifests.

    The transformation is purely syntactic: split at each lowercase-to-uppercase
    boundary and before an uppercase run that starts a new word, then uppercase.
    Digit runs stay glued to the token they follow (`Uint8` -> `UINT8`,
    `Crc32c` -> `CRC32C`), because a digit boundary in CamelCase does not mark a
    word boundary in the spec vocabulary these names model.

    Consecutive capitals do not split, so acronym-adjacent names derive badly:
    `ZarrV2ZArrayJSON` -> `ZARR_V2ZARRAY_JSON` and `...JSONPartial` ->
    `...JSONPARTIAL`. Every such name in the package today is a `TypedDict` or
    `TypeAliasType` that backs no constant, so none reaches this function — but
    a future `Literal` spelled that way would silently be held to a bad name.
    Splitting acronyms correctly needs a vocabulary, not a regex, so the rule
    stays syntactic and this stays a known limit.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])", "_", type_name).upper()


def _literal_backed_constants() -> list[tuple[str, str, str]]:
    """Every (module, constant, type) triple where a module-level SCREAMING_SNAKE
    constant holds exactly the values of a `Literal` type in the same module.

    Pairing is by value, not by proximity: a constant manifests the type whose
    members it enumerates. Constants with no such type (extension-field keys,
    key sets, canonical bit patterns) are exempt from the naming rule and are
    simply absent from the result.
    """
    out: list[tuple[str, str, str]] = []
    for module_name in _iter_module_names():
        module = importlib.import_module(module_name)
        literals = {
            name: frozenset(get_args(obj))
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and not name.isupper()
            and get_origin(obj) is Literal
            and get_args(obj)
        }
        if not literals:
            continue
        for const_name, value in vars(module).items():
            if const_name.startswith("_") or not const_name.isupper():
                continue
            if isinstance(value, tuple):
                members = frozenset(value)
            elif isinstance(value, str):
                members = frozenset({value})
            else:
                # Unhashable non-value constants (rule sets, factory
                # registries) back no Literal type and carry no signal.
                continue
            if not all(isinstance(m, str) for m in members):
                continue
            matches = [t for t, args in literals.items() if args == members]
            # A single unambiguous type means this constant manifests it. Ties
            # (two Literals with identical members) carry no signal about which
            # name the constant should take, so they are skipped.
            if len(matches) == 1:
                out.append((module_name, const_name, matches[0]))
    return out


def _value_tied_constants() -> set[str]:
    """Constants whose manifested type is ambiguous because two or more `Literal`
    types in the same module share its exact members.

    These are invisible to the derivation check, so they are surfaced here and
    counted, rather than silently dropped inside the pairing helper."""
    tied: set[str] = set()
    for module_name in _iter_module_names():
        module = importlib.import_module(module_name)
        literals = [
            frozenset(get_args(obj))
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and not name.isupper()
            and get_origin(obj) is Literal
            and get_args(obj)
        ]
        for const_name, value in vars(module).items():
            if const_name.startswith("_") or not const_name.isupper():
                continue
            if isinstance(value, tuple):
                members = frozenset(value)
            elif isinstance(value, str):
                members = frozenset({value})
            else:
                # Unhashable non-value constants (rule sets, factory
                # registries) back no Literal type and carry no signal.
                continue
            if not all(isinstance(m, str) for m in members):
                continue
            if sum(1 for args in literals if args == members) > 1:
                tied.add(f"{module_name}.{const_name}")
    return tied


# Constants whose `Literal` type cannot be identified by value because another
# `Literal` in the same module has identical members. Pairing is by value, so a
# tie carries no signal about which name the constant should take. These are
# checked by eye; the count below fails if the tied set grows silently.
KNOWN_VALUE_TIES = 9


def test_constant_names_derive_from_their_type_names() -> None:
    """Every `Literal`-backed constant whose type can be identified by value has
    a name that is the mechanical transform of that type's name.

    Constants tied to more than one identically-valued `Literal` are exempt (see
    `KNOWN_VALUE_TIES`), as are constants in private modules and those backing
    no `Literal` at all — so this pins the rule for most of the package, not all
    of it."""
    pairs = _literal_backed_constants()
    assert pairs, "found no Literal-backed constants to check"
    violations = [
        f"{module}: {const} should be {constant_name_for(type_name)} (manifests {type_name})"
        for module, const, type_name in pairs
        if const != constant_name_for(type_name)
    ]
    assert not violations, "constants whose names do not derive from their type:\n" + "\n".join(
        violations
    )


def test_value_tied_constants_are_a_known_set() -> None:
    """The derivation check cannot see constants whose type is ambiguous by
    value. Pin how many there are, so the exempt set cannot grow unnoticed and
    quietly shrink the rule's coverage."""
    tied = _value_tied_constants()
    assert len(tied) == KNOWN_VALUE_TIES, (
        f"value-tied constants changed (expected {KNOWN_VALUE_TIES}, got {len(tied)}); "
        f"these are unchecked by the derivation rule and must be named by hand:\n"
        + "\n".join(sorted(tied))
    )
