"""
Type-level shape validation for known metadata-field entities.

One validator per extension-point entity this package defines, exact with
respect to the entity's declared TypedDicts: a value yields no problems
exactly when it is an instance of the canonical metadata type, so a
verdict here means the same thing the type does.

Two package-wide conventions qualify "exact":

- `int`-annotated fields mean JSON integers, so JSON booleans are
  rejected even though `bool` is an `int` subtype in Python's type
  system (matching the fill-value rules' treatment of integers).
- Judgments are at the canonical data level: JSON arrays are tuples, as
  the TypedDicts declare. Normalize a freshly-`json.loads`-ed document
  (e.g. with a model-layer parser) before asking for shape verdicts.

Value judgments beyond the types — permutation contents, shard geometry,
cross-field consistency — belong to the composition rule layer, not here.

Unknown names are not judged (extension openness): the `validate_known_*`
functions answer `None` for entities this package has no types for, no
problems for a valid known entity, and problems otherwise.

Key sets are derived from the TypedDicts' `__annotations__` /
`__required_keys__` rather than restated by hand, so those entries cannot
drift from the canonical types; only the per-field value checks are
written out. The exception is `_BARE_DATA_TYPE_NAMES`: the core scalar
data types have no TypedDict to derive from — their whole metadata is a
name — so that list is hand-written, and `tests/test_registry_drift.py`
ties it to the modules that define those names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import (
    ValidationProblem,
    is_json,
    is_metadata_field_v3,
)
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    RAW_BYTES_FAMILY,
    ExtensionPointField,
    canonical_name,
)
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID_NAME,
    RectilinearChunkGridConfiguration,
    RectilinearChunkGridObject,
)
from zarr_metadata.v3.chunk_grid.regular import (
    REGULAR_CHUNK_GRID_NAME,
    RegularChunkGridConfiguration,
    RegularChunkGridObject,
)
from zarr_metadata.v3.chunk_key_encoding.default import (
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
    DefaultChunkKeyEncodingConfiguration,
    DefaultChunkKeyEncodingObject,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2_CHUNK_KEY_ENCODING_NAME,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
    V2ChunkKeyEncodingConfiguration,
    V2ChunkKeyEncodingObject,
)
from zarr_metadata.v3.codec.blosc import (
    BLOSC_CNAME,
    BLOSC_CODEC_NAME,
    BLOSC_SHUFFLE,
    BloscCodecConfiguration,
    BloscCodecObject,
)
from zarr_metadata.v3.codec.bytes import (
    BYTES_CODEC_NAME,
    ENDIANNESS,
    BytesCodecConfiguration,
    BytesCodecObject,
)
from zarr_metadata.v3.codec.cast_value import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC_NAME,
    CastValueCodecConfiguration,
    CastValueCodecObject,
    ScalarMap,
)
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME, Crc32cCodecObject, Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME, GzipCodecConfiguration, GzipCodecObject
from zarr_metadata.v3.codec.scale_offset import (
    SCALE_OFFSET_CODEC_NAME,
    ScaleOffsetCodecConfiguration,
    ScaleOffsetCodecObject,
)
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEX_LOCATION,
    SHARDING_INDEXED_CODEC_NAME,
    ShardingIndexedCodecConfiguration,
    ShardingIndexedCodecObject,
)
from zarr_metadata.v3.codec.transpose import (
    TRANSPOSE_CODEC_NAME,
    TransposeCodecConfiguration,
    TransposeCodecObject,
)
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME, ZstdCodecConfiguration, ZstdCodecObject
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_datetime64 import (
    NUMPY_DATETIME64_DATA_TYPE_NAME,
    NumpyDatetime64,
    NumpyDatetime64Configuration,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NUMPY_TIME_UNIT,
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    NumpyTimedelta64,
    NumpyTimedelta64Configuration,
)
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.struct import (
    STRUCT_DATA_TYPE_NAME,
    Struct,
    StructConfiguration,
    StructField,
)
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE_NAME

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata.model._validation import ProblemKind

    _FieldChecker = Callable[[object, tuple[str | int, ...]], tuple[ValidationProblem, ...]]


def entity_name(value: object) -> str | None:
    """The `name` of a metadata-field entry in any spelling, or None.

    A bare string is its own name; an object's name is its `name` member.
    Anything else (or a mapping without a string `name`) has no name and
    is not interpretable as a known entity.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    name = cast("Mapping[object, object]", value).get("name")
    return name if isinstance(name, str) else None


def _problems(
    loc: tuple[str | int, ...], message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    return (ValidationProblem(loc, message, kind),)


def _check_json_int(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if isinstance(value, bool) or not isinstance(value, int):
        return _problems(loc, f"expected an integer, got {value!r}")
    return ()


def _check_json_bool(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, bool):
        return _problems(loc, f"expected a boolean, got {value!r}")
    return ()


def _literal(allowed: tuple[str, ...]) -> _FieldChecker:
    def check(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
        if value not in allowed:
            return _problems(loc, f"expected one of {allowed!r}, got {value!r}", "invalid_value")
        return ()

    return check


def _check_int_tuple(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return _problems(loc, f"expected an array (tuple) of integers, got {value!r}")
    items = cast("tuple[object, ...]", value)
    return tuple(
        problem
        for index, item in enumerate(items)
        for problem in _check_json_int(item, (*loc, index))
    )


def _check_json_value(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if not is_json(value):
        return _problems(loc, "expected a JSON value")
    return ()


def _check_metadata_field(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    if not is_metadata_field_v3(value):
        return _problems(loc, "expected a metadata field (bare name or name/configuration object)")
    return ()


def _check_field_tuple(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return _problems(loc, f"expected an array (tuple) of metadata fields, got {value!r}")
    items = cast("tuple[object, ...]", value)
    return tuple(
        problem
        for index, item in enumerate(items)
        for problem in _check_metadata_field(item, (*loc, index))
    )


_STRUCT_FIELD_KEYS: Final = frozenset(StructField.__annotations__)


def _check_data_type_field(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """A nested `data_type` position: structural shape plus its known shape.

    `cast_value`'s target type and a struct field's type are data types
    like any other, so they get the judgment a top-level `data_type` gets.
    Without this the same value is accepted in one position and rejected
    in another — a bare `"numpy.datetime64"` is invalid at the top level
    (its configuration is required) and was silently fine inside a struct.
    Recurses naturally: a struct of structs is judged all the way down.
    """
    problems = _check_metadata_field(value, loc)
    if len(problems) != 0:
        return problems
    found = validate_known_entity_metadata(DATA_TYPE, value)
    return () if found is None else _prefixed_at(loc, found)


def _prefixed_at(
    loc: tuple[str | int, ...], problems: tuple[ValidationProblem, ...]
) -> tuple[ValidationProblem, ...]:
    return tuple(
        ValidationProblem((*loc, *problem.loc), problem.message, problem.kind)
        for problem in problems
    )


def _check_struct_fields(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return _problems(loc, f"expected an array (tuple) of struct fields, got {value!r}")
    problems: list[ValidationProblem] = []
    for index, item in enumerate(cast("tuple[object, ...]", value)):
        item_loc = (*loc, index)
        if not isinstance(item, Mapping):
            problems.extend(_problems(item_loc, f"expected an object, got {item!r}"))
            continue
        field = cast("Mapping[object, object]", item)
        for key in field:
            if not isinstance(key, str) or key not in _STRUCT_FIELD_KEYS:
                problems.extend(_problems(item_loc, f"unexpected key {key!r}", "unknown_key"))
        for key in sorted(_STRUCT_FIELD_KEYS - field.keys()):
            problems.extend(_problems((*item_loc, key), "missing required key", "missing_key"))
        if "name" in field and not isinstance(field["name"], str):
            problems.extend(_problems((*item_loc, "name"), "expected a string"))
        if "data_type" in field:
            problems.extend(_check_data_type_field(field["data_type"], (*item_loc, "data_type")))
    return tuple(problems)


_SCALAR_MAP_KEYS: Final = frozenset(ScalarMap.__annotations__)


def _check_scalar_map_entries(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return _problems(loc, f"expected an array (tuple) of [old, new] pairs, got {value!r}")
    problems: list[ValidationProblem] = []
    for index, item in enumerate(cast("tuple[object, ...]", value)):
        if not isinstance(item, tuple) or len(cast("tuple[object, ...]", item)) != 2:
            problems.extend(_problems((*loc, index), f"expected an [old, new] pair, got {item!r}"))
            continue
        for position, scalar in enumerate(cast("tuple[object, ...]", item)):
            problems.extend(_check_json_value(scalar, (*loc, index, position)))
    return tuple(problems)


def _check_scalar_map(value: object, loc: tuple[str | int, ...]) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, Mapping):
        return _problems(loc, f"expected an object, got {value!r}")
    mapping = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
    for key in mapping:
        if not isinstance(key, str) or key not in _SCALAR_MAP_KEYS:
            problems.extend(_problems((*loc,), f"unexpected key {key!r}", "unknown_key"))
    for key in _SCALAR_MAP_KEYS:
        if key in mapping:
            problems.extend(_check_scalar_map_entries(mapping[key], (*loc, key)))
    return tuple(problems)


def _check_rectilinear_dim_spec(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, bool) and isinstance(value, int):
        return ()
    if not isinstance(value, tuple):
        return _problems(
            loc,
            f"expected an integer or an array of integers / [value, count] pairs, got {value!r}",
        )
    problems: list[ValidationProblem] = []
    for index, item in enumerate(cast("tuple[object, ...]", value)):
        if not isinstance(item, bool) and isinstance(item, int):
            continue
        if isinstance(item, tuple) and len(cast("tuple[object, ...]", item)) == 2:
            problems.extend(
                problem
                for position, part in enumerate(cast("tuple[object, ...]", item))
                for problem in _check_json_int(part, (*loc, index, position))
            )
            continue
        problems.extend(
            _problems((*loc, index), f"expected an integer or a [value, count] pair, got {item!r}")
        )
    return tuple(problems)


def _check_rectilinear_dim_specs(
    value: object, loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return _problems(loc, f"expected an array (tuple) of dimension specs, got {value!r}")
    return tuple(
        problem
        for index, item in enumerate(cast("tuple[object, ...]", value))
        for problem in _check_rectilinear_dim_spec(item, (*loc, index))
    )


@dataclass(frozen=True, slots=True)
class _EntityShape:
    """Shape facts for one known entity name, derived from its TypedDicts."""

    object_keys: frozenset[str]
    configuration_required: bool
    config_keys: frozenset[str]
    config_required: frozenset[str]
    config_checkers: Mapping[str, _FieldChecker]


def _shape(
    object_type: type,
    configuration_type: type,
    checkers: Mapping[str, _FieldChecker],
) -> _EntityShape:
    config_keys = frozenset(configuration_type.__annotations__)
    if frozenset(checkers) != config_keys:
        raise AssertionError(  # pragma: no cover - registry construction guard
            f"checkers {sorted(checkers)} do not cover configuration keys {sorted(config_keys)}"
        )
    return _EntityShape(
        object_keys=frozenset(object_type.__annotations__),
        configuration_required="configuration"
        in cast("frozenset[str]", object_type.__required_keys__),  # type: ignore[attr-defined]
        config_keys=config_keys,
        config_required=cast(
            "frozenset[str]",
            configuration_type.__required_keys__,  # type: ignore[attr-defined]
        ),
        config_checkers=dict(checkers),
    )


def _bare_shape() -> _EntityShape:
    """Shape of an entity that takes no configuration.

    Both spellings are valid: the spec makes `{"name": ...}` the base form
    and permits the bare short-hand when no configuration is required. The
    object form is therefore accepted with an absent or empty
    `configuration`, and any member inside one is an unknown key.

    Keyword arguments deliberately: this is a six-field record whose flags
    are easy to transpose positionally.
    """
    return _EntityShape(
        object_keys=frozenset({"name", "configuration", "must_understand"}),
        configuration_required=False,
        config_keys=frozenset(),
        config_required=frozenset(),
        config_checkers={},
    )


_CODEC_SHAPES: Final[Mapping[str, _EntityShape]] = {
    BLOSC_CODEC_NAME: _shape(
        BloscCodecObject,
        BloscCodecConfiguration,
        {
            "cname": _literal(BLOSC_CNAME),
            "clevel": _check_json_int,
            "shuffle": _literal(BLOSC_SHUFFLE),
            "blocksize": _check_json_int,
            "typesize": _check_json_int,
        },
    ),
    BYTES_CODEC_NAME: _shape(
        BytesCodecObject, BytesCodecConfiguration, {"endian": _literal(ENDIANNESS)}
    ),
    CAST_VALUE_CODEC_NAME: _shape(
        CastValueCodecObject,
        CastValueCodecConfiguration,
        {
            "data_type": _check_data_type_field,
            "rounding": _literal(CAST_ROUNDING_MODE),
            "out_of_range": _literal(CAST_OUT_OF_RANGE_MODE),
            "scalar_map": _check_scalar_map,
        },
    ),
    CRC32C_CODEC_NAME: _shape(Crc32cCodecObject, Empty, {}),
    GZIP_CODEC_NAME: _shape(GzipCodecObject, GzipCodecConfiguration, {"level": _check_json_int}),
    SCALE_OFFSET_CODEC_NAME: _shape(
        ScaleOffsetCodecObject,
        ScaleOffsetCodecConfiguration,
        {"offset": _check_json_value, "scale": _check_json_value},
    ),
    SHARDING_INDEXED_CODEC_NAME: _shape(
        ShardingIndexedCodecObject,
        ShardingIndexedCodecConfiguration,
        {
            "chunk_shape": _check_int_tuple,
            "codecs": _check_field_tuple,
            "index_codecs": _check_field_tuple,
            "index_location": _literal(SHARDING_INDEX_LOCATION),
        },
    ),
    TRANSPOSE_CODEC_NAME: _shape(
        TransposeCodecObject, TransposeCodecConfiguration, {"order": _check_int_tuple}
    ),
    ZSTD_CODEC_NAME: _shape(
        ZstdCodecObject,
        ZstdCodecConfiguration,
        {"level": _check_json_int, "checksum": _check_json_bool},
    ),
}

_CHUNK_GRID_SHAPES: Final[Mapping[str, _EntityShape]] = {
    REGULAR_CHUNK_GRID_NAME: _shape(
        RegularChunkGridObject, RegularChunkGridConfiguration, {"chunk_shape": _check_int_tuple}
    ),
    RECTILINEAR_CHUNK_GRID_NAME: _shape(
        RectilinearChunkGridObject,
        RectilinearChunkGridConfiguration,
        {
            # "inline" is the sole member of the kind Literal; the type
            # exports no constant tuple for it.
            "kind": _literal(("inline",)),
            "chunk_shapes": _check_rectilinear_dim_specs,
        },
    ),
}

_CHUNK_KEY_ENCODING_SHAPES: Final[Mapping[str, _EntityShape]] = {
    DEFAULT_CHUNK_KEY_ENCODING_NAME: _shape(
        DefaultChunkKeyEncodingObject,
        DefaultChunkKeyEncodingConfiguration,
        {"separator": _literal(DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR)},
    ),
    V2_CHUNK_KEY_ENCODING_NAME: _shape(
        V2ChunkKeyEncodingObject,
        V2ChunkKeyEncodingConfiguration,
        {"separator": _literal(V2_CHUNK_KEY_ENCODING_SEPARATOR)},
    ),
}

_BARE_DATA_TYPE_NAMES: Final = (
    BOOL_DATA_TYPE_NAME,
    INT8_DATA_TYPE_NAME,
    INT16_DATA_TYPE_NAME,
    INT32_DATA_TYPE_NAME,
    INT64_DATA_TYPE_NAME,
    UINT8_DATA_TYPE_NAME,
    UINT16_DATA_TYPE_NAME,
    UINT32_DATA_TYPE_NAME,
    UINT64_DATA_TYPE_NAME,
    FLOAT16_DATA_TYPE_NAME,
    FLOAT32_DATA_TYPE_NAME,
    FLOAT64_DATA_TYPE_NAME,
    COMPLEX64_DATA_TYPE_NAME,
    COMPLEX128_DATA_TYPE_NAME,
    RAW_BYTES_FAMILY,
    BYTES_DATA_TYPE_NAME,
    STRING_DATA_TYPE_NAME,
)

_DATA_TYPE_SHAPES: Final[Mapping[str, _EntityShape]] = {
    **{name: _bare_shape() for name in _BARE_DATA_TYPE_NAMES},
    NUMPY_DATETIME64_DATA_TYPE_NAME: _shape(
        NumpyDatetime64,
        NumpyDatetime64Configuration,
        {"unit": _literal(NUMPY_TIME_UNIT), "scale_factor": _check_json_int},
    ),
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME: _shape(
        NumpyTimedelta64,
        NumpyTimedelta64Configuration,
        {"unit": _literal(NUMPY_TIME_UNIT), "scale_factor": _check_json_int},
    ),
    STRUCT_DATA_TYPE_NAME: _shape(Struct, StructConfiguration, {"fields": _check_struct_fields}),
}


def _validate_known_entity(
    value: object, name: str, shape: _EntityShape, entity: str
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not an instance of `name`'s canonical type.

    Locations are relative to the entry itself (`("configuration", key)`
    etc.); callers prefix the entry's position in its document.
    """
    if isinstance(value, str):
        if not shape.configuration_required:
            return ()
        return _problems(
            (),
            f"{entity} {name!r} requires a configuration and has no bare short-hand "
            f"form; use {{'name': {name!r}, 'configuration': {{...}}}}",
            "invalid_value",
        )
    if not isinstance(value, Mapping):
        return _problems((), f"expected a bare name or an object, got {value!r}")
    mapping = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
    for key in mapping:
        if not isinstance(key, str) or key not in shape.object_keys:
            problems.extend(_problems((), f"unexpected key {key!r}", "unknown_key"))
    if "must_understand" in mapping:
        problems.extend(_check_json_bool(mapping["must_understand"], ("must_understand",)))
    if "configuration" not in mapping:
        if shape.configuration_required:
            problems.extend(
                _problems(
                    ("configuration",),
                    f"{entity} {name!r} requires a 'configuration' object",
                    "missing_key",
                )
            )
        return tuple(problems)
    configuration = mapping["configuration"]
    if not isinstance(configuration, Mapping):
        problems.extend(_problems(("configuration",), f"expected an object, got {configuration!r}"))
        return tuple(problems)
    config = cast("Mapping[object, object]", configuration)
    for key in config:
        if not isinstance(key, str) or key not in shape.config_keys:
            problems.extend(_problems(("configuration",), f"unexpected key {key!r}", "unknown_key"))
    for key in sorted(shape.config_required - {k for k in config if isinstance(k, str)}):
        problems.extend(
            _problems(
                ("configuration", key),
                f"configuration for {entity} {name!r} is missing required key {key!r}",
                "missing_key",
            )
        )
    for key, checker in shape.config_checkers.items():
        if key in config:
            problems.extend(checker(config[key], ("configuration", key)))
    return tuple(problems)


def validate_known_codec_metadata(value: object) -> tuple[ValidationProblem, ...] | None:
    """Shape problems for a known-name codec entry, or None if not judged.

    None means the entry has no interpretable name or its name is not a
    codec this package defines (extension openness: unknown entities are
    not ours to judge). An empty list means `value` is an instance of the
    named codec's canonical metadata type.
    """
    name = entity_name(value)
    if name is None:
        return None
    shape = _CODEC_SHAPES.get(name)
    if shape is None:
        return None
    return _validate_known_entity(value, name, shape, "codec")


def validate_known_chunk_grid_metadata(value: object) -> tuple[ValidationProblem, ...] | None:
    """Shape problems for a known-name chunk grid entry, or None if not judged."""
    name = entity_name(value)
    if name is None:
        return None
    shape = _CHUNK_GRID_SHAPES.get(name)
    if shape is None:
        return None
    return _validate_known_entity(value, name, shape, "chunk grid")


_ENTITY_SHAPES: Final[Mapping[ExtensionPointField, Mapping[str, _EntityShape]]] = {
    DATA_TYPE: _DATA_TYPE_SHAPES,
    CODECS: _CODEC_SHAPES,
    CHUNK_GRID: _CHUNK_GRID_SHAPES,
    CHUNK_KEY_ENCODING: _CHUNK_KEY_ENCODING_SHAPES,
}


def validate_known_entity_metadata(
    field: ExtensionPointField, value: object
) -> tuple[ValidationProblem, ...] | None:
    """Shape problems for an entity known at `field`, or None if not judged."""
    name = entity_name(value)
    if name is None:
        return None
    shape = _ENTITY_SHAPES.get(field, {}).get(canonical_name(field, name))
    if shape is None:
        return None
    return _validate_known_entity(value, name, shape, field.replace("_", " ").rstrip("s"))


def modelled_entities() -> frozenset[tuple[ExtensionPointField, str]]:
    """Every `(extension point, name)` with a shape validator."""
    return frozenset((field, name) for field, shapes in _ENTITY_SHAPES.items() for name in shapes)


def blocking_problems(
    problems: Sequence[ValidationProblem],
) -> tuple[ValidationProblem, ...]:
    """The problems that prevent interpreting an entity's fields.

    `unknown_key` problems do not: a member this package does not model
    says nothing about the members it does. Rules use this so a single
    unrecognized key cannot silently suppress every other judgment about
    the same entity — the extra key is still reported, and the geometry
    checks still run.
    """
    return tuple(problem for problem in problems if problem.kind != "unknown_key")


__all__ = [
    "blocking_problems",
    "entity_name",
    "modelled_entities",
    "validate_known_chunk_grid_metadata",
    "validate_known_codec_metadata",
    "validate_known_entity_metadata",
]
