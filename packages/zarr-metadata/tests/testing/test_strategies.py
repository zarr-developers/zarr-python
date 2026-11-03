"""The strategies against the definitions they draw for, and against what they ought to reach.

Every draw is read in `ZARR_PYTHON_CONTEXT` and must read without a
problem, by the definition it was drawn for: the declaration alone, a fill
value beside its data type, a chunk grid over the shape it was drawn
for, a codec handed the chunk it was drawn for, and a pipeline handed
the chunks of a grid, as `read_pipeline` reads one. A strategy that only
drew valid metadata could draw almost none, so reach is checked too,
with targets worked out from each configuration's TypedDict rather than
from the strategies: every optional key present and absent, every
`Literal` value, and for a definition with a simplest spelling, a
configuration it simplifies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args, get_origin

import pytest
from hypothesis import event, find, given, settings
from hypothesis import strategies as st

from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.definition import (
    AcceptedDeclaration,
    Definition,
    EmptyConfiguration,
    fill_value_problems,
    resolve,
)
from zarr_metadata.definition._definition import as_kind, canonicalize, kind_of, unchanged
from zarr_metadata.testing._draw import scope_with
from zarr_metadata.testing._strategy import EMPTY_CONFIGURATIONS
from zarr_metadata.testing.strategies import (
    DEFAULT_STRATEGIES,
    ChunkKeyEncodingStrategy,
    CodecStrategy,
    DataTypeStrategy,
    Spellings,
    Strategies,
    chunk_grid_configurations,
    codec_configurations,
    data_type_configurations,
    declarations,
    fill_values,
    pipelines,
)
from zarr_metadata.v2.definition import CORE_V2_CONTEXT
from zarr_metadata.v3._definition import (
    chunk_grid_lengths,
    no_pipelines,
    storage_of,
    variable_length,
)
from zarr_metadata.v3._pipeline import read_pipeline
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
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
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE_CONTEXT,
    ZARR_PYTHON_CONTEXT,
    Chunk,
    ChunkGridDefinition,
    CodecDefinition,
    DataTypeDefinition,
    ZarrV3Context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3 import ZarrV3DeclarationJSON
    from zarr_metadata.v3._definition import Lengths

_IN_SCOPE = frozenset(
    (
        *ZARR_PYTHON_CONTEXT.data_types.values(),
        *ZARR_PYTHON_CONTEXT.chunk_grids.values(),
        *ZARR_PYTHON_CONTEXT.chunk_key_encodings.values(),
        *ZARR_PYTHON_CONTEXT.codecs.values(),
        *ZARR_PYTHON_CONTEXT.storage_transformers.values(),
    )
)
"""The definitions at the spec's extension points that `ZARR_PYTHON_CONTEXT` holds; members of a document it defines are not declarations."""
_CORE_DATA_TYPES = frozenset(CORE_CONTEXT.data_types.values())

_DEFINITIONS = sorted(DEFAULT_STRATEGIES, key=lambda definition: definition.name)
_DATA_TYPES = [d for d in _DEFINITIONS if isinstance(d, DataTypeDefinition)]
_CHUNK_GRIDS = [d for d in _DEFINITIONS if isinstance(d, ChunkGridDefinition)]
_CODECS = [d for d in _DEFINITIONS if isinstance(d, CodecDefinition)]

T = TypeVar("T")

_SHAPES = st.lists(st.integers(0, 12), max_size=4).map(tuple)


def _id(value: object) -> str:
    if isinstance(value, Definition):
        return f"{type(value).__name__}:{value.name}"
    return str(value)


def _read(definition: Definition[Any], declaration: object) -> Any:  # noqa: ANN401
    """`declaration` read in `ZARR_PYTHON_CONTEXT`, asserted to read without a problem by `definition`."""
    resolved, problems = resolve(declaration, as_kind(kind_of(definition)), ZARR_PYTHON_CONTEXT)
    assert problems == ()
    assert isinstance(resolved, AcceptedDeclaration)
    assert resolved.definition is definition
    return resolved


def test_every_definition_has_a_strategy() -> None:
    """Each definition in scope has a strategy, and none is out of scope."""
    assert frozenset(DEFAULT_STRATEGIES) == _IN_SCOPE


@pytest.mark.parametrize("spellings", get_args(Spellings))
@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
@given(data=st.data())
def test_declarations_read(
    definition: Definition[Any], spellings: Spellings, data: st.DataObject
) -> None:
    """A drawn declaration reads by its definition, through JSON too; a canonical one is its own simplest spelling."""
    declaration = data.draw(declarations(definition, spellings=spellings))
    _read(definition, declaration)
    _read(definition, json.loads(json.dumps(declaration)))
    if definition in _CORE_DATA_TYPES:
        assert isinstance(declaration, str)
    if spellings == "canonical":
        canonical, _ = canonicalize(declaration, as_kind(kind_of(definition)), ZARR_PYTHON_CONTEXT)
        assert canonical == declaration


@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
@given(data=st.data())
def test_a_changed_draw_changes_no_other(definition: Definition[Any], data: st.DataObject) -> None:
    """Each draw is a fresh value: a test that changes the configuration it drew does not change the next draw, which Hypothesis would report as flaky."""
    declaration = data.draw(declarations(definition))
    if isinstance(declaration, dict) and "configuration" in declaration:
        configuration = cast("dict[str, JSONValue]", declaration["configuration"])
        assert "changed" not in configuration
        configuration["changed"] = True


@pytest.mark.parametrize("definition", _DATA_TYPES, ids=_id)
@given(data=st.data())
def test_fill_values_fit_their_data_type(
    definition: DataTypeDefinition[Any], data: st.DataObject
) -> None:
    configuration = data.draw(data_type_configurations(definition))
    data_type = _read(definition, data.draw(declarations(definition, st.just(configuration))))
    value = data.draw(fill_values(definition, configuration))
    assert fill_value_problems(data_type, value) == ()
    assert fill_value_problems(data_type, json.loads(json.dumps(value))) == ()


@pytest.mark.parametrize("definition", _CHUNK_GRIDS, ids=_id)
@given(data=st.data(), shape=_SHAPES)
def test_chunk_grids_fit_their_shape(
    definition: ChunkGridDefinition[Any], data: st.DataObject, shape: tuple[int, ...]
) -> None:
    configurations = chunk_grid_configurations(definition, shape)
    chunk_grid = _read(definition, data.draw(declarations(definition, configurations)))
    lengths, problems = chunk_grid_lengths(chunk_grid, shape)
    assert problems == ()
    # "Chunk sizes must be greater than zero", which the regular grid's rules do not yet refuse.
    assert all(0 not in axis for axis in lengths if axis is not None)


@st.composite
def _chunks(draw: st.DrawFn) -> Chunk:
    """A chunk of a data type drawn from the strategies, or of none known, of lengths known along some axes, all, or none, or of a grid drawn over an array of some shape.

    A rectilinear grid's chunks take several lengths along an axis, which
    a codec drawn for them must take each of.
    """
    lengths: Lengths | None
    if draw(st.booleans()):
        shape = draw(_SHAPES)
        grid = draw(st.sampled_from(_CHUNK_GRIDS))
        chunk_grid = _read(grid, draw(declarations(grid, chunk_grid_configurations(grid, shape))))
        lengths, _ = chunk_grid_lengths(chunk_grid, shape)
    else:
        axis = st.none() | st.integers(1, 8).map(lambda length: frozenset({length}))
        lengths = draw(st.none() | st.lists(axis, max_size=4).map(tuple))
    definition = draw(st.none() | st.sampled_from(_DATA_TYPES))
    data_type = None if definition is None else _read(definition, draw(declarations(definition)))
    return Chunk(lengths, data_type)


def _handed_problems(codec: Any, chunk: Chunk) -> list[object]:  # noqa: ANN401
    """What is wrong with `codec`, a codec field read, handed `chunk`: by its chunk rules, and in each pipeline it holds."""
    definition = codec.definition
    problems: list[object] = list(definition.chunk_rules(codec.configuration, codec.nested, chunk))
    handed = definition.pipelines(codec.configuration, codec.nested, chunk)
    for member, inner in handed.items():
        held = [codec.nested[(member, index)] for index in range(len(codec.configuration[member]))]
        problems.extend(read_pipeline(held, inner)[1])
    return problems


def _holds_pipelines(definition: CodecDefinition[Any]) -> bool:
    return definition.pipelines is not no_pipelines


def _encoders_refuse(data: st.DataObject, chunk: Chunk) -> bool:
    """Whether every array -> bytes codec that holds no pipelines refuses a chunk of `chunk`'s data type."""
    of_data_type = Chunk(None, chunk.data_type)
    encoders = [d for d in _CODECS if d.kind == "array_bytes" and not _holds_pipelines(d)]
    for encoder in encoders:
        codec = _read(encoder, data.draw(declarations(encoder)))
        if list(encoder.chunk_rules(codec.configuration, codec.nested, of_data_type)) == []:
            return False
    return True


@pytest.mark.parametrize("definition", _CODECS, ids=_id)
@given(data=st.data(), chunk=_chunks())
def test_codecs_take_their_chunk(
    definition: CodecDefinition[Any], data: st.DataObject, chunk: Chunk
) -> None:
    """A codec drawn for a chunk takes it; one drawn for none is one the spec does not give it, which needs the data type known.

    A shard drawn for none is one whose inner chunks no codec encodes.
    """
    configurations = codec_configurations(definition, chunk)
    if configurations is None:
        assert chunk.data_type is not None
        assert chunk.data_type.definition is not None
        data_type_name = chunk.data_type.definition.name
        event(f"takes no {data_type_name}")
        if _holds_pipelines(definition):
            assert _encoders_refuse(data, chunk)
            return
        codec = _read(definition, data.draw(declarations(definition)))
        assert list(definition.chunk_rules(codec.configuration, codec.nested, chunk)) != []
        return
    codec = _read(definition, data.draw(declarations(definition, configurations)))
    assert _handed_problems(codec, chunk) == []


_MIXED_LENGTHS = st.lists(
    st.frozensets(st.integers(1, 24), min_size=1, max_size=3), max_size=3
).map(tuple)
"""Several lengths along each axis, as a rectilinear grid gives: one inner length must divide them all."""


@given(data=st.data(), lengths=_MIXED_LENGTHS)
def test_shards_divide_every_length(
    data: st.DataObject, lengths: tuple[frozenset[int], ...]
) -> None:
    chunk = Chunk(lengths)
    configurations = codec_configurations(SHARDING_INDEXED_CODEC, chunk)
    assert configurations is not None
    codec = _read(
        SHARDING_INDEXED_CODEC, data.draw(declarations(SHARDING_INDEXED_CODEC, configurations))
    )
    assert _handed_problems(codec, chunk) == []


def test_shards_divide_a_long_shard() -> None:
    """A length of 2**30 has its divisors found without counting to it."""
    chunk = Chunk((frozenset({2**30}),))
    configurations = codec_configurations(SHARDING_INDEXED_CODEC, chunk)
    assert configurations is not None
    find(
        configurations,
        lambda configuration: configuration["chunk_shape"][0] > 2**15,
        settings=_REACH,
    )


def test_a_shard_whose_index_nothing_encodes_is_none() -> None:
    """Without a codec of static size to encode the index, no shard is drawn."""
    without_bytes = {d: s for d, s in DEFAULT_STRATEGIES.items() if d is not BYTES_CODEC}
    assert codec_configurations(SHARDING_INDEXED_CODEC, strategies=without_bytes) is None


def test_a_cast_from_a_data_type_without_a_strategy_draws_numbers() -> None:
    """Scalars of a chunk's data type that no strategy draws for are small numbers, which it holds."""
    data_type = _read(INT8_DATA_TYPE, "int8")
    strategies: Strategies = {
        CAST_VALUE_CODEC: DEFAULT_STRATEGIES[CAST_VALUE_CODEC],
        FLOAT32_DATA_TYPE: DEFAULT_STRATEGIES[FLOAT32_DATA_TYPE],
    }
    configurations = codec_configurations(
        CAST_VALUE_CODEC, Chunk(None, data_type), strategies=strategies
    )
    assert configurations is not None
    find(configurations, lambda configuration: "scalar_map" in configuration, settings=_REACH)


@st.composite
def _partly_known_chunks(draw: st.DrawFn) -> Chunk:
    """A chunk some of which nothing says: its lengths along an axis or at all, or its data type, read by nothing in scope."""
    chunk = draw(_chunks())
    lengths = draw(
        st.sampled_from(
            [
                chunk.lengths,
                None,
                None if chunk.lengths is None else tuple(None for _ in chunk.lengths),
            ]
        )
    )
    data_type = draw(st.sampled_from([chunk.data_type, None, _UNREAD_DATA_TYPE]))
    return Chunk(lengths, data_type)


_UNREAD_DATA_TYPE = resolve({"name": "an.unread.type"}, DataTypeDefinition, ZARR_PYTHON_CONTEXT)[0]
"""A data type field that nothing in scope claims: a chunk of it says nothing of its values."""


@given(data=st.data(), chunk=_chunks() | _partly_known_chunks())
def test_pipelines_read(data: st.DataObject, chunk: Chunk) -> None:
    """A pipeline drawn for a chunk reads as one handed it, through JSON too; none is drawn only for a chunk no codec encodes."""
    drawn = pipelines(chunk)
    if drawn is None:
        assert chunk.data_type is not None
        event(f"no pipeline for {chunk.data_type.json!r}")
        assert storage_of(chunk.data_type) == "variable_length"
        return
    pipeline = data.draw(drawn)
    for written in (pipeline, json.loads(json.dumps(pipeline))):
        codecs = [resolve(field, CodecDefinition, ZARR_PYTHON_CONTEXT) for field in written]
        assert [problems for _, problems in codecs] == [()] * len(codecs)
        _, problems = read_pipeline([codec for codec, _ in codecs], chunk)
        assert problems == ()


Target = tuple[Literal["present", "absent"], str] | tuple[Literal["value"], str, object]
DeclarationForm = Literal["bare", "object", "configured", "must_understand"]


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


def _declaration_forms(definition: Definition[Any]) -> list[DeclarationForm]:
    """The envelopes the spec allows a declaration of `definition` to be written in."""
    if definition in _CORE_DATA_TYPES:
        return []  # only ever bare, which `test_declarations_read` asserts
    if definition.requires_configuration:
        return ["configured", "must_understand"]
    return ["bare", "object", "configured", "must_understand"]


def _form_of(declaration: ZarrV3DeclarationJSON) -> set[DeclarationForm]:
    if isinstance(declaration, str):
        return {"bare"}
    forms: set[DeclarationForm] = {"configured" if "configuration" in declaration else "object"}
    if "must_understand" in declaration:
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
_ONCE = settings(database=None)


@pytest.mark.parametrize(
    ("definition", "target"),
    [(d, target) for d in _DEFINITIONS for target in _configuration_targets(d)],
    ids=_id,
)
def test_configurations_reach(
    definition: Definition[Any], target: Target | tuple[Literal["simplified"]]
) -> None:
    configurations = declarations(definition).map(
        lambda declaration: _read(definition, declaration).configuration
    )
    find(
        configurations,
        lambda configuration: _hits(definition, target, configuration),
        settings=_REACH,
    )


@pytest.mark.parametrize(
    ("definition", "form"),
    [(d, form) for d in _DEFINITIONS for form in _declaration_forms(d)],
    ids=_id,
)
def test_declarations_reach(definition: Definition[Any], form: DeclarationForm) -> None:
    find(
        declarations(definition), lambda declaration: form in _form_of(declaration), settings=_REACH
    )


def test_an_added_strategy_is_drawn_from() -> None:
    """A strategy joined to the defaults, as an extension author adds one, is the one drawn from."""
    level_three = CodecStrategy(configurations=lambda _chunk, _strategies: st.just({"level": 3}))
    strategies = DEFAULT_STRATEGIES | {GZIP_CODEC: level_three}
    declaration = find(
        declarations(GZIP_CODEC, strategies=strategies), lambda _declaration: True, settings=_REACH
    )
    assert _read(GZIP_CODEC, declaration).configuration == {"level": 3}


_SHORT_REPR = 10_000
"""Well short of the 30,000 characters past which Hypothesis warns about a strategy's repr.

It writes that repr into the note on any error raised while drawing,
its own rejections among them, so a longer one fails a suite that turns
warnings into errors.
"""


@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
def test_reprs_stay_short(definition: Definition[Any]) -> None:
    """However the strategies are passed: a plain mapping is as long as every strategy in it."""
    plain: Strategies = dict(DEFAULT_STRATEGIES)
    assert len(repr(declarations(definition, strategies=plain))) < _SHORT_REPR
    assert len(repr(pipelines(strategies=plain))) < _SHORT_REPR


def test_a_definition_without_a_strategy_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no strategy for 'gzip'"):
        declarations(GZIP_CODEC, strategies={})


def test_a_strategy_of_another_kind_is_a_type_error() -> None:
    # Unannotated, as an extension author writes it: a mapping keyed by a
    # `CodecDefinition` is a `Strategies`.
    strategies = {
        GZIP_CODEC: ChunkKeyEncodingStrategy(configurations=lambda _strategies: st.just({}))
    }
    with pytest.raises(TypeError, match="draws configurations of a ChunkKeyEncodingDefinition"):
        codec_configurations(GZIP_CODEC, strategies=strategies)


def test_a_codec_strategy_refusing_an_unknown_chunk_is_a_value_error() -> None:
    refusing = CodecStrategy(configurations=lambda _chunk, _strategies: None)
    with pytest.raises(ValueError, match=r"Chunk\(\), a chunk nothing is known of"):
        declarations(GZIP_CODEC, strategies=DEFAULT_STRATEGIES | {GZIP_CODEC: refusing})


def test_fill_values_of_a_field_no_definition_reads_is_a_lookup_error() -> None:
    """A struct handed a field type that no definition in the strategies reads has no fill values to draw for it."""
    without_float32 = {d: s for d, s in DEFAULT_STRATEGIES.items() if d is not FLOAT32_DATA_TYPE}
    configuration: Any = {"fields": ({"name": "a", "data_type": "float32"},)}
    with pytest.raises(LookupError, match="no definition in `strategies` reads"):
        fill_values(STRUCT_DATA_TYPE, configuration, strategies=without_float32)


def test_a_refused_nested_draw_is_a_value_error() -> None:
    """A field a configuration holds is read as it is drawn, so a strategy that draws a refused one is caught there."""
    refused = DataTypeStrategy(
        configurations=lambda _strategies: st.just({"stray": 1}),
        fill_values=lambda _configuration, _strategies: st.just(0.0),
    )
    strategies: Strategies = {
        CAST_VALUE_CODEC: DEFAULT_STRATEGIES[CAST_VALUE_CODEC],
        NUMPY_DATETIME64_DATA_TYPE: refused,
    }
    with pytest.raises(ValueError, match="read in the strategies. scope, is refused"):
        find(
            declarations(CAST_VALUE_CODEC, strategies=strategies),
            lambda _field: True,
            settings=_ONCE,
        )


@pytest.mark.parametrize("spellings", get_args(Spellings))
def test_a_refused_draw_is_a_value_error(spellings: Spellings) -> None:
    refused = declarations(GZIP_CODEC, st.just({"level": 99}), spellings=spellings)
    with pytest.raises(ValueError, match="which is refused"):
        find(refused, lambda _field: True, settings=_ONCE)


_BFLOAT16 = DataTypeDefinition(name="bfloat16", configuration=EmptyConfiguration)
"""A number type this package does not define, as an extension author would."""


@given(data=st.data())
def test_scale_offset_takes_a_data_type_the_package_does_not_define(data: st.DataObject) -> None:
    """Values of a type the package does not define cannot be drawn, so the configuration is the identity, which its rules take."""
    data_type, problems = resolve(
        "bfloat16", DataTypeDefinition, ZarrV3Context(data_types=[_BFLOAT16])
    )
    assert problems == ()
    chunk = Chunk(None, data_type)
    configurations = codec_configurations(SCALE_OFFSET_CODEC, chunk)
    assert configurations is not None
    configuration = data.draw(configurations)
    assert configuration == {}
    assert list(SCALE_OFFSET_CODEC.chunk_rules(configuration, {}, chunk)) == []


def test_a_definition_no_v3_scope_holds_is_a_type_error() -> None:
    zlib = CORE_V2_CONTEXT.codecs["zlib"]
    with pytest.raises(TypeError, match="which no slot of a Zarr v3 scope holds"):
        declarations(zlib, st.just({}))


@pytest.mark.parametrize(
    "definition", [FLOAT16_DATA_TYPE, FLOAT32_DATA_TYPE, FLOAT64_DATA_TYPE], ids=_id
)
@pytest.mark.parametrize("member", ["offset", "scale"])
def test_scale_offset_reaches_a_value_written_in_hex(
    definition: DataTypeDefinition[Any], member: str
) -> None:
    """A float's `offset` and `scale` are fill values of its type, which the bits in hex spell too."""
    chunk = Chunk(None, _read(definition, definition.name))
    configurations = codec_configurations(SCALE_OFFSET_CODEC, chunk)
    assert configurations is not None
    find(
        configurations,
        lambda configuration: isinstance(configuration.get(member), str),
        settings=_REACH,
    )


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


def _codecs_in(pipeline: tuple[Any, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """Each codec's name in `pipeline`, with the members of the shards it sits inside."""
    found: list[tuple[str, tuple[str, ...]]] = []
    for field in pipeline:
        if isinstance(field, str):
            found.append((field, ()))
            continue
        found.append((field["name"], ()))
        if field["name"] != SHARDING_INDEXED_CODEC.name:
            continue
        configuration: Mapping[str, Any] = field["configuration"]
        for member in ("codecs", "index_codecs"):
            found.extend(
                (inner, (member, *within)) for inner, within in _codecs_in(configuration[member])
            )
    return found


def _drawn(strategy: st.SearchStrategy[T] | None) -> st.SearchStrategy[T]:
    """`strategy`, or nothing when there is none."""
    return st.nothing() if strategy is None else strategy


def _drawn_pipelines() -> st.SearchStrategy[tuple[Any, ...]]:
    return _chunks().flatmap(lambda chunk: _drawn(pipelines(chunk)))


def _shards() -> st.SearchStrategy[Any]:
    """Shards drawn for the chunks of a grid."""
    return _chunks().flatmap(
        lambda chunk: _drawn(codec_configurations(SHARDING_INDEXED_CODEC, chunk)).flatmap(
            lambda configuration: declarations(SHARDING_INDEXED_CODEC, st.just(configuration))
        )
    )


@pytest.mark.parametrize("definition", _CODECS, ids=_id)
def test_pipelines_reach_every_codec(definition: CodecDefinition[Any]) -> None:
    find(
        _drawn_pipelines(),
        lambda pipeline: any(name == definition.name for name, _ in _codecs_in(pipeline)),
        settings=_REACH,
    )


@pytest.mark.parametrize(
    ("definition", "member"),
    [
        (d, member)
        for d in _CODECS
        if d is not SHARDING_INDEXED_CODEC
        for member in ("codecs", "index_codecs")
        if member == "codecs" or d.size == "static"
    ],
    ids=_id,
)
def test_shards_hold_every_codec_their_pipeline_takes(
    definition: CodecDefinition[Any], member: str
) -> None:
    """A shard's pipelines reach each codec they take: its index codecs, only those of static size."""
    find(
        _shards(),
        lambda shard: (definition.name, (member,)) in _codecs_in((shard,)),
        settings=_REACH,
    )


def test_shards_reach_a_shard_inside() -> None:
    find(
        _shards(),
        lambda shard: (SHARDING_INDEXED_CODEC.name, ("codecs",)) in _codecs_in((shard,)),
        settings=_REACH,
    )


_VARIABLE_LENGTH = frozenset(d for d in _DATA_TYPES if d.storage is variable_length)
"""The data types whose values vary in size, which a struct's fields may not be."""


@pytest.mark.parametrize(
    "definition", [d for d in _DATA_TYPES if d not in _VARIABLE_LENGTH], ids=_id
)
def test_struct_fields_reach_every_fixed_size_data_type(
    definition: DataTypeDefinition[Any],
) -> None:
    """A struct's field types reach each data type of fixed size, a struct among them."""
    find(
        data_type_configurations(STRUCT_DATA_TYPE),
        lambda configuration: any(
            _read_data_type(member["data_type"]).definition is definition
            for member in configuration["fields"]
        ),
        settings=_REACH,
    )


def _read_data_type(field: object) -> Any:  # noqa: ANN401
    resolved, problems = resolve(field, DataTypeDefinition, ZARR_PYTHON_CONTEXT)
    assert problems == ()
    return resolved


_REAL_NUMBERS = [
    d
    for d in _DATA_TYPES
    if d.name
    in {
        *(f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64)),
        *(f"float{bits}" for bits in (16, 32, 64)),
        "numpy.datetime64",
        "numpy.timedelta64",
    }
]
"""The data types `cast_value` casts to: integers and floats, and the numpy times, which are signed 64-bit integers."""


@pytest.mark.parametrize("definition", _REAL_NUMBERS, ids=_id)
def test_cast_value_reaches_every_real_number_data_type(
    definition: DataTypeDefinition[Any],
) -> None:
    find(
        declarations(CAST_VALUE_CODEC).map(lambda field: _read(CAST_VALUE_CODEC, field)),
        lambda codec: codec.nested[("data_type",)].definition is definition,
        settings=_REACH,
    )


@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_cast_value_reaches_scalars_in_each_direction(direction: str) -> None:
    find(
        _drawn(codec_configurations(CAST_VALUE_CODEC)),
        lambda configuration: len(configuration.get("scalar_map", {}).get(direction, ())) != 0,
        settings=_REACH,
    )


@given(declaration=declarations(CAST_VALUE_CODEC))
def test_cast_value_draws_only_real_number_data_types(declaration: ZarrV3DeclarationJSON) -> None:
    """The reach test above says each of these is drawn; this says nothing else is."""
    target = _read(CAST_VALUE_CODEC, declaration).nested[("data_type",)]
    assert target.definition in _REAL_NUMBERS


_PLAIN_GZIP = CodecDefinition(
    name=GZIP_CODEC.name, configuration=EmptyConfiguration, kind="bytes_bytes", size="dynamic"
)
"""A codec named as one the defaults hold, with a configuration of its own."""

_WITH_PLAIN_GZIP = DEFAULT_STRATEGIES | {
    _PLAIN_GZIP: CodecStrategy(configurations=lambda _chunk, _strategies: EMPTY_CONFIGURATIONS)
}


@given(declaration=declarations(_PLAIN_GZIP, st.just({})))
def test_a_definition_named_as_a_default_is_read_as_itself(
    declaration: ZarrV3DeclarationJSON,
) -> None:
    """It claims its name in the scope its draw is read in, though the strategies hold another, whose rules refuse an empty configuration."""
    scope = scope_with(_PLAIN_GZIP, DEFAULT_STRATEGIES)
    read, problems = resolve(declaration, CodecDefinition, scope)
    assert problems == ()
    assert isinstance(read, AcceptedDeclaration)
    assert read.definition is _PLAIN_GZIP


def test_a_definition_shadowed_in_the_strategies_is_read_as_itself() -> None:
    """Asked for by name, it claims that name over the one that replaced it."""
    declaration = find(
        declarations(GZIP_CODEC, strategies=_WITH_PLAIN_GZIP),
        lambda _declaration: True,
        settings=_ONCE,
    )
    assert _read(GZIP_CODEC, declaration).definition is GZIP_CODEC


@given(pipeline=_drawn(pipelines(strategies=_WITH_PLAIN_GZIP)))
def test_a_later_definition_replaces_a_default_of_its_name(pipeline: tuple[Any, ...]) -> None:
    """A pipeline holds the later of two codecs with one name, so none has the earlier's `level`."""
    gzips = [
        field for field in pipeline if isinstance(field, dict) and field["name"] == GZIP_CODEC.name
    ]
    assert [field.get("configuration", {}) for field in gzips] == [{}] * len(gzips)
