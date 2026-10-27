"""The strategies against the definitions they draw for, and against what they ought to reach.

Every draw is read in `CORE_AND_EXTENSIONS` and must read without a
problem, by the definition it was drawn for: the field alone, a fill
value beside its data type, a chunk grid over the shape it was drawn
for, a codec handed the chunk it was drawn for. A strategy that only
drew valid metadata could draw almost none, so reach is checked too,
with targets worked out from each configuration's TypedDict rather than
from the strategies: every optional key present and absent, every
`Literal` value, and for a definition with a simplest spelling, a
configuration it simplifies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin

import pytest
from hypothesis import event, find, given, settings
from hypothesis import strategies as st

from zarr_metadata.testing.strategies import (
    DEFAULT_STRATEGIES,
    ChunkKeyEncodingStrategy,
    CodecStrategy,
    Spellings,
    Strategies,
    chunk_grid_configurations,
    codec_configurations,
    data_type_configurations,
    fields,
    fill_values,
)
from zarr_metadata.typed_json import typeddict_keys
from zarr_metadata.v3._definition import as_kind, kind_of, unchanged
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.data_type._float import FloatSpecialFillValue
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    Chunk,
    ChunkGridDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Definition,
    ZarrV3MetadataFieldJSON,
    canonicalize,
    chunk_grid_lengths,
    fill_value_problems,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue

_HOLDING_FIELDS = frozenset({SHARDING_INDEXED_CODEC, CAST_VALUE_CODEC, STRUCT_DATA_TYPE})
"""The definitions whose configurations hold other fields, which have no strategy yet."""

_IN_SCOPE = frozenset(
    definition for table in CORE_AND_EXTENSIONS.tables.values() for definition in table.values()
)
_CORE_DATA_TYPES = frozenset(CORE.tables[DataTypeDefinition].values())

_DEFINITIONS = sorted(DEFAULT_STRATEGIES, key=lambda definition: definition.name)
_DATA_TYPES = [d for d in _DEFINITIONS if isinstance(d, DataTypeDefinition)]
_CHUNK_GRIDS = [d for d in _DEFINITIONS if isinstance(d, ChunkGridDefinition)]
_CODECS = [d for d in _DEFINITIONS if isinstance(d, CodecDefinition)]

_OPEN_FOR_SCALE_OFFSET = frozenset(
    {"complex64", "complex128", "numpy.datetime64", "numpy.timedelta64"}
)
"""Data types `scale_offset` is drawn for none of, though its rules refuse none: its registry entry leaves them open."""

_SHAPES = st.lists(st.integers(0, 12), max_size=4).map(tuple)


def _id(value: object) -> str:
    if isinstance(value, Definition):
        return f"{type(value).__name__}:{value.name}"
    return str(value)


def _read(definition: Definition[Any], field: object) -> Any:  # noqa: ANN401
    """`field` read in `CORE_AND_EXTENSIONS`, asserted to read without a problem by `definition`."""
    resolved, problems = resolve(field, as_kind(kind_of(definition)), CORE_AND_EXTENSIONS)
    assert problems == ()
    assert resolved.resolution == "read"
    assert resolved.definition is definition
    return resolved


def test_every_definition_has_a_strategy() -> None:
    """Each definition in scope has a strategy, but those that hold other fields; and none is out of scope."""
    assert _IN_SCOPE - frozenset(DEFAULT_STRATEGIES) == _HOLDING_FIELDS
    assert frozenset(DEFAULT_STRATEGIES) <= _IN_SCOPE


@pytest.mark.parametrize("spellings", get_args(Spellings))
@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
@given(data=st.data())
def test_fields_read(
    definition: Definition[Any], spellings: Spellings, data: st.DataObject
) -> None:
    """A drawn field reads by its definition, through JSON too; a canonical one is its own simplest spelling."""
    field = data.draw(fields(definition, spellings=spellings))
    _read(definition, field)
    _read(definition, json.loads(json.dumps(field)))
    if definition in _CORE_DATA_TYPES:
        assert isinstance(field, str)
    if spellings == "canonical":
        canonical, _ = canonicalize(field, as_kind(kind_of(definition)), CORE_AND_EXTENSIONS)
        assert canonical == field


@pytest.mark.parametrize("definition", _DATA_TYPES, ids=_id)
@given(data=st.data())
def test_fill_values_fit_their_data_type(
    definition: DataTypeDefinition[Any], data: st.DataObject
) -> None:
    configuration = data.draw(data_type_configurations(definition))
    data_type = _read(definition, data.draw(fields(definition, st.just(configuration))))
    value = data.draw(fill_values(definition, configuration))
    assert fill_value_problems(data_type, value) == ()
    assert fill_value_problems(data_type, json.loads(json.dumps(value))) == ()


@pytest.mark.parametrize("definition", _CHUNK_GRIDS, ids=_id)
@given(data=st.data(), shape=_SHAPES)
def test_chunk_grids_fit_their_shape(
    definition: ChunkGridDefinition[Any], data: st.DataObject, shape: tuple[int, ...]
) -> None:
    configurations = chunk_grid_configurations(definition, shape)
    chunk_grid = _read(definition, data.draw(fields(definition, configurations)))
    lengths, problems = chunk_grid_lengths(chunk_grid, shape)
    assert problems == ()
    # "Chunk sizes must be greater than zero", which the regular grid's rules do not yet refuse.
    assert all(0 not in axis for axis in lengths if axis is not None)


@st.composite
def _chunks(draw: st.DrawFn) -> Chunk:
    """A chunk of some lengths, of a data type drawn from the strategies."""
    definition = draw(st.sampled_from(_DATA_TYPES))
    data_type = _read(definition, draw(fields(definition)))
    lengths = draw(st.lists(st.integers(1, 8), max_size=4))
    return Chunk(tuple(frozenset({length}) for length in lengths), data_type)


@pytest.mark.parametrize("definition", _CODECS, ids=_id)
@given(data=st.data(), chunk=_chunks())
def test_codecs_take_their_chunk(
    definition: CodecDefinition[Any], data: st.DataObject, chunk: Chunk
) -> None:
    """A codec drawn for a chunk takes it; one drawn for none is one the spec does not give it."""
    assert chunk.data_type is not None
    assert chunk.data_type.definition is not None
    data_type_name = chunk.data_type.definition.name
    configurations = codec_configurations(definition, chunk)
    if configurations is None:
        event(f"takes no {data_type_name}")
        if definition is SCALE_OFFSET_CODEC and data_type_name in _OPEN_FOR_SCALE_OFFSET:
            return
        codec = _read(definition, data.draw(fields(definition)))
        assert list(definition.chunk_rules(codec.configuration, codec.nested, chunk)) != []
        return
    codec = _read(definition, data.draw(fields(definition, configurations)))
    assert list(definition.chunk_rules(codec.configuration, codec.nested, chunk)) == []


Target = tuple[Literal["present", "absent"], str] | tuple[Literal["value"], str, object]
FieldForm = Literal["bare", "object", "configured", "must_understand"]


def _configuration_targets(
    definition: Definition[Any],
) -> list[Target | tuple[Literal["simplified"]]]:
    """What a strategy for `definition` ought to reach, by its configuration's TypedDict."""
    targets: list[Target | tuple[Literal["simplified"]]] = []
    for key, (annotation, required) in typeddict_keys(definition.configuration).members.items():
        if not required:
            targets.extend([("present", key), ("absent", key)])
        if get_origin(annotation) is Literal:
            targets.extend(("value", key, value) for value in get_args(annotation))
    if definition.canonical is not unchanged:
        targets.append(("simplified",))
    return targets


def _field_forms(definition: Definition[Any]) -> list[FieldForm]:
    """The envelopes the spec allows a field of `definition` to be written in."""
    if definition in _CORE_DATA_TYPES:
        return []  # only ever bare, which `test_fields_read` asserts
    if definition.requires_configuration:
        return ["configured", "must_understand"]
    return ["bare", "object", "configured", "must_understand"]


def _form_of(field: ZarrV3MetadataFieldJSON) -> set[FieldForm]:
    if isinstance(field, str):
        return {"bare"}
    forms: set[FieldForm] = {"configured" if "configuration" in field else "object"}
    if "must_understand" in field:
        forms.add("must_understand")
    return forms


def _hits(
    definition: Definition[Any],
    target: Target | tuple[Literal["simplified"]],
    configuration: Mapping[str, JSONValue],
) -> bool:
    match target:
        case ("present", key):
            return key in configuration
        case ("absent", key):
            return key not in configuration
        case ("value", key, value):
            return configuration.get(key) == value
        case ("simplified",):
            return definition.canonical(configuration) != configuration


_REACH = settings(max_examples=2000, database=None)


@pytest.mark.parametrize(
    ("definition", "target"),
    [(d, target) for d in _DEFINITIONS for target in _configuration_targets(d)],
    ids=_id,
)
def test_configurations_reach(
    definition: Definition[Any], target: Target | tuple[Literal["simplified"]]
) -> None:
    configurations = fields(definition).map(lambda field: _read(definition, field).configuration)
    find(
        configurations,
        lambda configuration: _hits(definition, target, configuration),
        settings=_REACH,
    )


@pytest.mark.parametrize(
    ("definition", "form"),
    [(d, form) for d in _DEFINITIONS for form in _field_forms(d)],
    ids=_id,
)
def test_fields_reach(definition: Definition[Any], form: FieldForm) -> None:
    find(fields(definition), lambda field: form in _form_of(field), settings=_REACH)


def test_an_added_strategy_is_drawn_from() -> None:
    """A strategy joined to the defaults, as an extension author adds one, is the one drawn from."""
    level_three = CodecStrategy(configurations=lambda _chunk: st.just({"level": 3}))
    strategies = DEFAULT_STRATEGIES | {GZIP_CODEC: level_three}
    field = find(fields(GZIP_CODEC, strategies=strategies), lambda _field: True, settings=_REACH)
    assert _read(GZIP_CODEC, field).configuration == {"level": 3}


def test_a_definition_without_a_strategy_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no strategy for 'sharding_indexed'"):
        fields(SHARDING_INDEXED_CODEC)


def test_a_strategy_of_another_kind_is_a_type_error() -> None:
    strategies: Strategies = {GZIP_CODEC: ChunkKeyEncodingStrategy(configurations=st.just({}))}
    with pytest.raises(TypeError, match="draws configurations of a ChunkKeyEncodingDefinition"):
        codec_configurations(GZIP_CODEC, strategies=strategies)


def test_a_codec_strategy_refusing_an_unknown_chunk_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"Chunk\(\), a chunk nothing is known of"):
        CodecStrategy(configurations=lambda _chunk: None)


def test_a_refused_draw_in_the_canonical_spelling_is_a_value_error() -> None:
    refused = fields(GZIP_CODEC, st.just({"level": 99}), spellings="canonical")
    with pytest.raises(ValueError, match="which is refused"):
        find(refused, lambda _field: True, settings=settings(database=None))


def _has_pair(configuration: Mapping[str, Any]) -> bool:
    return any(
        isinstance(entry, tuple)
        for dimension in configuration["chunk_shapes"]
        if isinstance(dimension, tuple)
        for entry in cast("tuple[object, ...]", dimension)
    )


def test_rectilinear_edges_reach_a_run_written_as_a_pair() -> None:
    """The `[size, count]` spelling, which no TypedDict target asks for: it is one arm of a union."""
    find(chunk_grid_configurations(RECTILINEAR_CHUNK_GRID), _has_pair, settings=_REACH)


FillValueSpelling = Literal[
    "number", "named", "hex lowercase", "hex uppercase", "array", "base64", "NaT"
]

_FILL_VALUE_SPELLINGS: list[tuple[DataTypeDefinition[Any], FillValueSpelling]] = [
    *(
        (definition, spelling)
        for definition in (
            FLOAT16_DATA_TYPE,
            FLOAT32_DATA_TYPE,
            FLOAT64_DATA_TYPE,
            COMPLEX64_DATA_TYPE,
            COMPLEX128_DATA_TYPE,
        )
        for spelling in get_args(FillValueSpelling)[:4]
    ),
    (BYTES_DATA_TYPE, "array"),
    (BYTES_DATA_TYPE, "base64"),
    (NUMPY_DATETIME64_DATA_TYPE, "number"),
    (NUMPY_DATETIME64_DATA_TYPE, "NaT"),
    (NUMPY_TIMEDELTA64_DATA_TYPE, "number"),
    (NUMPY_TIMEDELTA64_DATA_TYPE, "NaT"),
]
"""The spellings the specs give a fill value, by the data types whose fill values take them."""


def _spellings_of(value: object) -> set[FillValueSpelling]:
    """How a fill value is spelled; a complex one, how either component is."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, int | float):
        return {"number"}
    if isinstance(value, tuple):
        parts = cast("tuple[object, ...]", value)
        if all(isinstance(part, int) for part in parts):
            return {"array"}
        return {spelling for part in parts for spelling in _spellings_of(part)}
    if not isinstance(value, str):
        return set()
    if value == "NaT":
        return {"NaT"}
    if value in get_args(FloatSpecialFillValue):
        return {"named"}
    if value.startswith("0x"):
        digits = value[2:]
        return {
            *(("hex lowercase",) if any(c in "abcdef" for c in digits) else ()),
            *(("hex uppercase",) if any(c in "ABCDEF" for c in digits) else ()),
        }
    return {"base64"}


@pytest.mark.parametrize(("definition", "spelling"), _FILL_VALUE_SPELLINGS, ids=_id)
def test_fill_values_reach(
    definition: DataTypeDefinition[Any], spelling: FillValueSpelling
) -> None:
    values = data_type_configurations(definition).flatmap(
        lambda configuration: fill_values(definition, configuration)
    )
    find(values, lambda value: spelling in _spellings_of(value), settings=_REACH)
