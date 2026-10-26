"""Reading codec pipelines: each codec together with the chunk it receives.

`read_pipeline` checks a pipeline's order, and checks each codec against
the chunk it receives. The v3 array validators read a document's
`codecs` as the pipeline that the chunks of its grid, with its data
type, pass through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import TypedDict

from zarr_metadata.definition import (
    EmptyConfiguration,
    Nested,
    ResolvedField,
    resolve,
)
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
    read_array_metadata_v3,
)
from zarr_metadata.v3._pipeline import (
    read_pipeline,
)
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    Chunk,
    CodecDefinition,
    DataTypeDefinition,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
)

if TYPE_CHECKING:
    from zarr_metadata import JSONValue, ValidationProblem
    from zarr_metadata.v3.definition import Stage


def _tiled(configuration: EmptyConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
    """Each chunk twice as long along every axis, of the same data type."""
    if chunk.lengths is None:
        return Chunk(None, chunk.data_type)
    doubled = tuple(
        None if axis is None else frozenset(2 * length for length in axis) for axis in chunk.lengths
    )
    return Chunk(doubled, chunk.data_type)


ACME_TILE = CodecDefinition(
    name="acme.tile",
    configuration=EmptyConfiguration,
    kind="array_array",
    size="static",
    transition=_tiled,
)
"""An array -> array codec that declares its output chunk."""

ACME_SHUFFLE = CodecDefinition(
    name="acme.shuffle", configuration=EmptyConfiguration, kind="array_array", size="static"
)
"""An array -> array codec that does not declare its output chunk."""

CONTEXT = CORE_AND_EXTENSIONS.update(
    codecs=[*CORE_AND_EXTENSIONS.codecs.values(), ACME_TILE, ACME_SHUFFLE]
)

UINT8 = resolve("uint8", DataTypeDefinition, CONTEXT)[0]

CHUNK = Chunk((frozenset({4}), frozenset({2})), UINT8)
"""Chunks of 4 by 2 `uint8` values."""

GZIP: JSONValue = {"name": "gzip", "configuration": {"level": 1}}

LITTLE: JSONValue = {"name": "bytes", "configuration": {"endian": "little"}}


def _transpose(*order: int) -> JSONValue:
    return {"name": "transpose", "configuration": {"order": list(order)}}


def _cast(data_type: JSONValue, **members: JSONValue) -> JSONValue:
    return {"name": "cast_value", "configuration": {"data_type": data_type, **members}}


def _scale(**members: JSONValue) -> JSONValue:
    return {"name": "scale_offset", "configuration": members}


def _of(data_type: JSONValue) -> Chunk:
    """Chunks of 4 by 2 values of `data_type`."""
    return Chunk(CHUNK.lengths, resolve(data_type, DataTypeDefinition, CONTEXT)[0])


def _struct(*field_types: JSONValue) -> JSONValue:
    fields: list[JSONValue] = [
        {"name": f"f{index}", "data_type": dt} for index, dt in enumerate(field_types)
    ]
    return {"name": "struct", "configuration": {"fields": fields}}


def _read(
    codecs: list[JSONValue], chunk: Chunk
) -> tuple[tuple[Stage, ...], tuple[ValidationProblem, ...]]:
    """The stages, and every problem: first each codec's own problems at its location, then the pipeline's."""
    read = [
        resolve(codec, CodecDefinition, CONTEXT, (index,)) for index, codec in enumerate(codecs)
    ]
    stages, found = read_pipeline([resolved for resolved, _ in read], chunk)
    return stages, (*(problem for _, own in read for problem in own), *found)


def _problems(
    codecs: list[JSONValue], chunk: Chunk = CHUNK
) -> list[tuple[tuple[str | int, ...], str]]:
    return [(problem.loc, problem.kind) for problem in _read(codecs, chunk)[1]]


def _lengths(*axes: set[int]) -> tuple[frozenset[int], ...]:
    return tuple(frozenset(axis) for axis in axes)


@pytest.mark.parametrize(
    ("codecs", "chunk", "incoming"),
    [
        (["bytes"], CHUNK, [CHUNK]),
        # Each array -> array codec outputs the chunk its transition gives.
        # transpose permutes the axes; two transposes restore them.
        (
            [_transpose(1, 0), "bytes"],
            CHUNK,
            [CHUNK, Chunk(_lengths({2}, {4}), UINT8)],
        ),
        (
            [_transpose(1, 0), _transpose(1, 0), "bytes"],
            CHUNK,
            [CHUNK, Chunk(_lengths({2}, {4}), UINT8), CHUNK],
        ),
        (["acme.tile", "bytes"], CHUNK, [CHUNK, Chunk(_lengths({8}, {4}), UINT8)]),
        # After the array -> bytes codec, each codec receives bytes.
        (["bytes", GZIP, "crc32c"], CHUNK, [CHUNK, None, None]),
        # A codec that does not declare its output gives the next codec a
        # chunk with unknown shape and data type.
        (["acme.shuffle", "bytes"], CHUNK, [CHUNK, Chunk()]),
        # So does a codec that nothing in scope claims, which might be of any
        # kind. A pipeline that holds one may have it as its array -> bytes codec.
        (["zfpy", "bytes"], CHUNK, [CHUNK, Chunk()]),
        (["zfpy"], CHUNK, [CHUNK]),
        (["bytes", "zfpy"], CHUNK, [CHUNK, None]),
        # A chunk with unknown properties stays unknown, and no codec rejects it.
        ([_transpose(1, 0, 2), "bytes"], Chunk(), [Chunk(), Chunk()]),
        # `bytes` requires an `endian` for values of several bytes, and accepts
        # any for single-byte values. The spec says nothing about wider raw bits.
        ([LITTLE], _of("float32"), [_of("float32")]),
        ([LITTLE], _of("uint8"), [_of("uint8")]),
        (["bytes"], _of(_struct("int8", "uint8")), [_of(_struct("int8", "uint8"))]),
        (["bytes"], _of("r16"), [_of("r16")]),
        # cast_value outputs its data type, and the next codec is checked
        # against it. scale_offset outputs the same chunk it receives.
        ([_cast("float32"), LITTLE], _of("int16"), [_of("int16"), _of("float32")]),
        (
            [_cast("uint8", out_of_range="wrap"), "bytes"],
            _of("float32"),
            [_of("float32"), _of("uint8")],
        ),
        (
            [_cast("float32"), _scale(offset=0.5, scale=2), _cast("int8"), "bytes"],
            _of("int8"),
            [_of("int8"), _of("float32"), _of("float32"), _of("int8")],
        ),
        (
            [_cast("uint8", scalar_map={"encode": [["NaN", 0]], "decode": [[0, "NaN"]]}), "bytes"],
            _of("float32"),
            [_of("float32"), _of("uint8")],
        ),
        # A cast to a data type that nothing in scope claims outputs that
        # field, which says nothing about the values.
        ([_cast("acme.decimal"), "bytes"], _of("float32"), [_of("float32"), _of("acme.decimal")]),
        # After a codec that nothing in scope claims, a cast still declares the
        # data type it outputs. An arithmetic codec that receives an unknown
        # chunk is not rejected.
        (
            ["zfpy", _cast("float32"), LITTLE],
            CHUNK,
            [CHUNK, Chunk(), Chunk(None, _of("float32").data_type)],
        ),
        ([_scale(offset=1), "vlen-utf8"], Chunk(), [Chunk(), Chunk()]),
        # Where the specs disagree or say nothing, a data type is not checked.
        # The numpy time types say they accept any codec of 64-bit integers,
        # and scale_offset does not say whether it accepts complex numbers.
        (
            [_cast("int64"), LITTLE],
            _of({"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}}),
            [
                _of(
                    {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}}
                ),
                _of("int64"),
            ],
        ),
        ([_scale(offset=[1, 0]), LITTLE], _of("complex64"), [_of("complex64"), _of("complex64")]),
    ],
)
def test_every_pipeline_hands_each_codec_the_chunk_the_one_before_hands_on(
    codecs: list[JSONValue], chunk: Chunk, incoming: list[Chunk | None]
) -> None:
    stages, problems = _read(codecs, chunk)
    assert problems == ()
    assert [stage.incoming for stage in stages] == incoming


@pytest.mark.parametrize(
    ("codecs", "index"),
    [
        ([GZIP, "bytes"], 1),
        (["bytes", _transpose(0, 1)], 1),
        (["bytes", "crc32c", _transpose(0, 1)], 2),
        (["bytes", GZIP, "bytes"], 2),
    ],
)
def test_error_a_codec_out_of_order(codecs: list[JSONValue], index: int) -> None:
    # The problem is located at the codec.
    assert _problems(codecs) == [((index,), "invalid_value")]


def test_error_a_codec_out_of_order_whose_configuration_is_not_an_object() -> None:
    # Its name still identifies the codec.
    assert _problems([{"name": "gzip", "configuration": 5}, "bytes"]) == [
        ((0, "configuration"), "invalid_type"),
        ((1,), "invalid_value"),
    ]


def test_error_a_second_array_to_bytes_codec() -> None:
    assert _problems(["bytes", "bytes"]) == [((1,), "invalid_value")]


@pytest.mark.parametrize("codecs", [[], [GZIP], [_transpose(0, 1)]])
def test_error_a_pipeline_without_an_array_to_bytes_codec(codecs: list[JSONValue]) -> None:
    assert _problems(codecs) == [((), "invalid_value")]


def test_error_a_transpose_order_with_another_number_of_axes_than_its_chunk() -> None:
    # The problem is located in the codec's configuration.
    assert _problems([_transpose(1, 0, 2), "bytes"]) == [
        ((0, "configuration", "order"), "invalid_value")
    ]
    # It is checked against the chunk it receives from the codec before it,
    # which here still has two axes.
    assert _problems([_transpose(1, 0), _transpose(0), "bytes"]) == [
        ((1, "configuration", "order"), "invalid_value")
    ]


def test_error_a_codec_with_a_problem_of_its_own_hands_on_a_chunk_nothing_is_known_of() -> None:
    # The reader does not guess what such a codec does, so the next codec
    # receives an unknown chunk and is not checked against it.
    stages, problems = _read([_transpose(0, 0), "bytes"], CHUNK)
    assert [(problem.loc, problem.kind) for problem in problems] == [
        ((0, "configuration", "order"), "invalid_value")
    ]
    assert [stage.incoming for stage in stages] == [CHUNK, Chunk()]


@pytest.mark.parametrize(
    "data_type",
    [
        "float32",
        "int16",
        _struct("int8", "float32"),
        {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
    ],
)
def test_error_a_bytes_codec_without_endian_handed_values_of_several_bytes(
    data_type: JSONValue,
) -> None:
    assert _problems(["bytes"], _of(data_type)) == [((0, "configuration", "endian"), "missing_key")]


def test_error_a_bytes_codec_without_endian_after_a_cast_past_a_codec_nothing_claims() -> None:
    # The cast declares the data type it outputs, whatever it receives.
    assert _problems(["zfpy", _cast("float32"), "bytes"]) == [
        ((2, "configuration", "endian"), "missing_key")
    ]


@pytest.mark.parametrize("data_type", ["string", "bytes"])
def test_error_a_bytes_codec_handed_values_that_vary_in_size(data_type: JSONValue) -> None:
    assert _problems([LITTLE], _of(data_type)) == [((0, "configuration"), "invalid_value")]


@pytest.mark.parametrize(
    ("scalar_map", "loc"),
    [
        # The input of encoding and the output of decoding are values of the
        # data type the cast receives.
        ({"encode": [[300, 0]]}, ("scalar_map", "encode", 0, 0)),
        ({"decode": [[0, -129]]}, ("scalar_map", "decode", 0, 1)),
    ],
)
def test_error_a_scalar_the_cast_maps_from_that_is_not_of_the_data_type_it_is_handed(
    scalar_map: JSONValue, loc: tuple[str | int, ...]
) -> None:
    codecs = [_cast("uint8", scalar_map=scalar_map), "bytes"]
    assert _problems(codecs, _of("int8")) == [((0, "configuration", *loc), "invalid_value")]


@pytest.mark.parametrize("codec", [_cast("float32"), _scale()])
@pytest.mark.parametrize("data_type", ["bool", "string", _struct("int8")])
def test_error_an_arithmetic_codec_handed_values_that_are_no_numbers(
    codec: JSONValue, data_type: JSONValue
) -> None:
    # The codec comes before an array -> bytes codec that nothing in scope
    # claims, which is not checked.
    assert _problems([codec, "vlen-utf8"], _of(data_type)) == [
        ((0, "configuration"), "invalid_value")
    ]


def test_error_a_codec_handed_on_what_another_could_not_take_is_judged_too() -> None:
    # scale_offset outputs the chunk it receives, which the bytes codec
    # after it cannot accept either.
    assert _problems([_scale(), LITTLE], _of("string")) == [
        ((0, "configuration"), "invalid_value"),
        ((1, "configuration"), "invalid_value"),
    ]


def test_error_a_cast_handed_complex_numbers() -> None:
    assert _problems([_cast("float32"), "vlen-utf8"], _of("complex64")) == [
        ((0, "configuration"), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("members", "data_type", "found"),
    [
        ({"offset": 0.5}, "int16", [((0, "configuration", "offset"), "invalid_type")]),
        ({"scale": 70000}, "int16", [((0, "configuration", "scale"), "invalid_value")]),
        ({"offset": "nan"}, "float32", [((0, "configuration", "offset"), "invalid_value")]),
    ],
)
def test_error_a_scale_or_offset_that_is_not_a_value_of_the_data_type_it_is_handed(
    members: dict[str, JSONValue], data_type: str, found: list[tuple[tuple[str | int, ...], str]]
) -> None:
    assert _problems([_scale(**members), LITTLE], _of(data_type)) == found


def test_error_a_transition_that_gives_something_else() -> None:
    lying = CodecDefinition(
        name="acme.lying",
        configuration=EmptyConfiguration,
        kind="array_array",
        size="static",
        transition=lambda configuration, nested, chunk: "a chunk",  # pyright: ignore[reportArgumentType]
    )
    codec = resolve(
        "acme.lying", CodecDefinition, CONTEXT.update(codecs=[*CONTEXT.codecs.values(), lying])
    )[0]
    with pytest.raises(TypeError, match="'acme.lying': its transition gives a Chunk"):
        read_pipeline([codec], CHUNK)


def test_error_a_transition_that_raises_says_whose_it_is() -> None:
    def transition(configuration: EmptyConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
        raise KeyError("axis")

    raising = CodecDefinition(
        name="acme.raising",
        configuration=EmptyConfiguration,
        kind="array_array",
        size="static",
        transition=transition,
    )
    codec = resolve(
        "acme.raising", CodecDefinition, CONTEXT.update(codecs=[*CONTEXT.codecs.values(), raising])
    )[0]
    with pytest.raises(KeyError) as raised:
        read_pipeline([codec], CHUNK, ("codecs",))
    assert raised.value.__notes__ == [
        "raised by the transition of 'acme.raising', reading ('codecs', 0, 'configuration')"
    ]


@pytest.mark.parametrize(
    ("lengths", "data_type", "match"),
    [
        ((4, 2), None, "a chunk's lengths are"),
        (None, "float32", "a chunk's data type is"),
        # A field read as a codec, though nothing claims it.
        (None, resolve("acme.t", CodecDefinition, CONTEXT)[0], "a chunk's data type is"),
    ],
)
def test_error_a_transition_that_builds_a_chunk_of_something_else(
    lengths: object, data_type: object, match: str
) -> None:
    # A chunk checks its own contents, so the bug is in the transition.
    def transition(configuration: EmptyConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
        return Chunk(lengths, data_type)  # pyright: ignore[reportArgumentType]

    odd = CodecDefinition(
        name="acme.odd",
        configuration=EmptyConfiguration,
        kind="array_array",
        size="static",
        transition=transition,
    )
    codec = resolve(
        "acme.odd", CodecDefinition, CONTEXT.update(codecs=[*CONTEXT.codecs.values(), odd])
    )[0]
    with pytest.raises(TypeError, match=match) as raised:
        read_pipeline([codec], CHUNK, ("codecs",))
    assert raised.value.__notes__ == [
        "raised by the transition of 'acme.odd', reading ('codecs', 0, 'configuration')"
    ]


@pytest.mark.parametrize(
    ("kind", "member"),
    [
        ("bytes_bytes", "chunk_rules"),
        ("bytes_bytes", "transition"),
        ("bytes_bytes", "pipelines"),
        ("array_bytes", "transition"),
    ],
)
def test_error_a_codec_hook_nothing_would_ask(kind: str, member: str) -> None:
    # A bytes -> bytes codec receives bytes, and only an array -> array
    # codec outputs a chunk, so these hooks would never be called.
    hook = {member: lambda configuration, nested, chunk: ()}
    with pytest.raises(TypeError, match=f"'acme.x': {member}, "):
        CodecDefinition(
            name="acme.x",
            configuration=EmptyConfiguration,
            kind=kind,  # pyright: ignore[reportArgumentType]
            size="static",
            **hook,  # pyright: ignore[reportArgumentType]
        )


def test_an_array_document_s_codecs_are_each_judged_against_the_chunk_they_are_handed() -> None:
    # scale_offset's offset is checked against the float32 that the first
    # cast outputs, not against the array's int8.
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
    document["data_type"] = "int8"
    document["codecs"] = [_cast("float32"), _scale(offset=0.5), _cast("int8"), "bytes"]
    assert read_array_metadata_v3(document).problems == ()


def test_error_a_transpose_of_another_rank_than_the_array_under_a_grid_nothing_claims() -> None:
    # A chunk has an extent for each dimension of the array, whatever the
    # grid says about the lengths.
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(4, 4)).to_json())
    document["chunk_grid"] = {"name": "acme.grid", "configuration": {}}
    document["codecs"] = [_transpose(0), "bytes"]
    assert [(p.loc, p.kind) for p in read_array_metadata_v3(document).problems] == [
        (("codecs", 0, "configuration", "order"), "invalid_value")
    ]


def test_error_an_array_document_s_codecs_out_of_order_or_not_fitting_its_chunks() -> None:
    # The problems are located in the document's codecs. The chunks come
    # from the grid, with the document's shape and data type.
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(4, 4)).to_json())
    document["data_type"] = "int16"
    document["codecs"] = [_transpose(0), "bytes", "crc32c", _transpose(0, 1)]
    assert [(p.loc, p.kind) for p in read_array_metadata_v3(document).problems] == [
        (("codecs", 3), "invalid_value"),
        (("codecs", 0, "configuration", "order"), "invalid_value"),
        (("codecs", 1, "configuration", "endian"), "missing_key"),
    ]


class AcmeHolderConfiguration(TypedDict, closed=True):
    codecs: tuple[ZarrV3CodecField, ...]
    types: tuple[ZarrV3DataTypeField, ...]


def _holder(
    pipelines: object, types: JSONValue = ("uint8",)
) -> ResolvedField[CodecDefinition[Any]]:
    """A codec that holds a pipeline of codecs and a list of data types, `types`, and whose `pipelines` hook is `pipelines`."""
    holder = CodecDefinition(
        name="acme.holder",
        configuration=AcmeHolderConfiguration,
        kind="array_bytes",
        size="dynamic",
        pipelines=pipelines,  # pyright: ignore[reportArgumentType]
    )
    field = {"name": "acme.holder", "configuration": {"codecs": [LITTLE], "types": types}}
    return resolve(
        field, CodecDefinition, CONTEXT.update(codecs=[*CONTEXT.codecs.values(), holder])
    )[0]


@pytest.mark.parametrize("given", ["a chunk", {"codecs": "a chunk"}, {("codecs",): Chunk()}])
def test_error_pipelines_that_give_something_else(given: object) -> None:
    with pytest.raises(TypeError, match="'acme.holder': its pipelines give a mapping of members"):
        read_pipeline([_holder(lambda configuration, nested, chunk: given)], CHUNK)


@pytest.mark.parametrize(
    ("member", "types"),
    [
        ("nowhere", ("uint8",)),
        ("types", ("uint8",)),
        # Data types that nothing in scope claims are still read as data types.
        ("types", ("acme.t",)),
    ],
)
def test_error_pipelines_that_name_a_member_holding_no_codecs(
    member: str, types: JSONValue
) -> None:
    holder = _holder(lambda configuration, nested, chunk: {member: Chunk()}, types)
    with pytest.raises(TypeError, match=f"its pipelines name {member!r}, which holds no list"):
        read_pipeline([holder], CHUNK)


def test_error_pipelines_that_raise_say_whose_they_are() -> None:
    holder = _holder(lambda configuration, nested, chunk: {}["codecs"])
    with pytest.raises(KeyError) as raised:
        read_pipeline([holder], CHUNK, ("codecs",))
    assert raised.value.__notes__ == [
        "raised by the pipelines of 'acme.holder', reading ('codecs', 0, 'configuration')"
    ]


def test_error_a_stage_s_inner_pipelines_are_read_only() -> None:
    """The pipelines of a shard's inner codecs held by a stage cannot be changed in place. A reading that holds them stays equal to itself after pickling and deep-copying."""
    import copy
    import pickle

    shard: JSONValue = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [2],
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"},
            ],
            "index_location": "end",
        },
    }
    stages, _ = _read([shard], CHUNK)
    stage = stages[0]
    assert "codecs" in stage.inner
    with pytest.raises(TypeError):
        stage.inner["codecs"] = ()  # pyright: ignore[reportIndexIssue]
    for again in (pickle.loads(pickle.dumps(stages)), copy.deepcopy(stages)):
        assert again == stages
