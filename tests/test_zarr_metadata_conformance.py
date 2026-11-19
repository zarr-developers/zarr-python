"""zarr-python's verdict on the array documents zarr-metadata's strategies draw.

zarr-metadata's strategies draw documents that the spec allows. Each codec in them
is drawn for the data type and the chunk the codecs before it hand on, so new codecs
and nestings are covered as soon as zarr-metadata defines them.

zarr-python refuses one kind of valid document on purpose: one whose fill value the
codecs cannot encode, because an unwritten chunk holds the fill value. A document
refused for any other reason is a divergence from the spec.

`_DIVERGENCES` lists the known divergences. The strategies here leave out each one
they can, and a strict xfail test below pins each one, so that fixing it fails that
test and prompts removing the exclusion.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import importlib.util
import json
import traceback
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Final

import pytest

# Not installed in the `min_deps` env: see the `conformance` dependency group.
pytest.importorskip("zarr_metadata.testing")

from hypothesis import event, given, reject
from hypothesis import strategies as st
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.testing.strategies import (
    DEFAULT_STRATEGIES,
    ChunkGridStrategy,
    DataTypeStrategy,
    array_documents,
)
from zarr_metadata.v3.chunk_grid import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.codec import CAST_VALUE_CODEC
from zarr_metadata.v3.data_type import BYTES_DATA_TYPE, RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import ChunkGridDefinition, CodecDefinition, DataTypeDefinition

import zarr
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.registry import get_codec_class

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from zarr_metadata.testing.strategies import Configurations, Strategies

_HAS_CAST_VALUE_RS: Final = importlib.util.find_spec("cast_value_rs") is not None

_DIVERGENCES: Final = {
    "data type spellings": (
        "A data type written as an object, such as `{'name': 'int8'}` or one with "
        "`must_understand`, is refused. Documents are drawn in canonical spelling."
    ),
    "raw bits": "The core `r*` data types are not implemented. None are drawn.",
    "bytes fill value": (
        "A `bytes` fill value written as an array of integers is refused. Only the "
        "base64 spelling is drawn."
    ),
    "sharding behind transpose": (
        "A sharding codec after a `transpose` is validated against the untransposed "
        "chunk. Such documents are rejected from the draw."
    ),
    "cast_value hex scalar map": (
        "A `scalar_map` value written as a hex string is refused while the fill value "
        "is encoded, so the property counts it as a fill value rejection. Not excluded."
    ),
    # zarr-metadata's side: it follows the spec as of zarr-extensions 4da7b37a, before
    # https://github.com/zarr-developers/zarr-extensions/pull/78 required at least one
    # edge. zarr-python refuses an empty list, as the spec now says.
    "empty rectilinear edges": (
        "zarr-metadata accepts an empty list of edge lengths for an axis of length 0. "
        "No such grid is drawn."
    ),
    "expanded rectilinear runs": (
        "zarr-python expands each run-length pair of rectilinear edges into a list, so a "
        "valid run of many chunks exhausts memory. No grid listing more than "
        "`_MAX_LISTED_CHUNKS` chunks along an axis is drawn."
    ),
}

_MAX_LISTED_CHUNKS: Final = 10_000


def _base64_fill_values(
    _configuration: Mapping[str, Any], _strategies: Strategies
) -> st.SearchStrategy[str]:
    return st.binary(max_size=16).map(lambda data: base64.standard_b64encode(data).decode())


def _listed_chunks(dimension: Any) -> int:
    """How many chunks a dimension's edges list, counting each run; 0 for a bare length."""
    if isinstance(dimension, int):
        return 0
    return sum(edge[1] if isinstance(edge, tuple | list) else 1 for edge in dimension)


def _readable(configuration: Mapping[str, Any]) -> bool:
    """Whether zarr-python reads the rectilinear grid without a known divergence."""
    return all(
        dimension != () and _listed_chunks(dimension) <= _MAX_LISTED_CHUNKS
        for dimension in configuration["chunk_shapes"]
    )


def _readable_grids(
    configurations: Callable[[tuple[int, ...] | None, Strategies], Configurations],
    shape: tuple[int, ...] | None,
    strategies: Strategies,
) -> Configurations:
    return configurations(shape, strategies).filter(_readable)


def _supported(strategies: Strategies) -> Strategies:
    """`strategies` without the known divergences in `_DIVERGENCES`."""
    out = {
        definition: strategy
        for definition, strategy in strategies.items()
        if definition is not RAW_BYTES_DATA_TYPE
        and (_HAS_CAST_VALUE_RS or definition is not CAST_VALUE_CODEC)
    }
    bytes_strategy = out.get(BYTES_DATA_TYPE)
    if isinstance(bytes_strategy, DataTypeStrategy):
        out[BYTES_DATA_TYPE] = dataclasses.replace(bytes_strategy, fill_values=_base64_fill_values)
    grid_strategy = out.get(RECTILINEAR_CHUNK_GRID)
    if isinstance(grid_strategy, ChunkGridStrategy):
        out[RECTILINEAR_CHUNK_GRID] = dataclasses.replace(
            grid_strategy, configurations=partial(_readable_grids, grid_strategy.configurations)
        )
    return out


_VALUE_DATA_TYPES: Final = frozenset(
    {"int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float64"}
)
_VALUE_CODECS: Final = frozenset(
    {"bytes", "cast_value", "scale_offset", "sharding_indexed", "transpose"}
)


def _value_transforming(strategies: Strategies) -> Strategies:
    """Integer and float data types, regular grids, and the codecs that change values.

    In this slice a fill value is often one the codecs cannot encode, which is where
    zarr-python's one deliberate refusal is decided.
    """
    return {
        definition: strategy
        for definition, strategy in strategies.items()
        if not (
            isinstance(definition, DataTypeDefinition) and definition.name not in _VALUE_DATA_TYPES
        )
        and not (isinstance(definition, CodecDefinition) and definition.name not in _VALUE_CODECS)
        and not (isinstance(definition, ChunkGridDefinition) and definition.name != "regular")
    }


_DOCUMENTS: Final = {
    "every-definition": array_documents(
        spellings="canonical", strategies=_supported(DEFAULT_STRATEGIES)
    ),
    "value-transforming": array_documents(
        shapes=st.lists(st.integers(1, 16), min_size=1, max_size=2).map(tuple),
        spellings="canonical",
        strategies=_value_transforming(_supported(DEFAULT_STRATEGIES)),
    ),
}


def _codec_name(codec: str | Mapping[str, Any]) -> str:
    return codec if isinstance(codec, str) else codec["name"]


def _is_array_array(codec: str | Mapping[str, Any]) -> bool:
    return issubclass(get_codec_class(_codec_name(codec)), ArrayArrayCodec)


def _shard_behind_transpose(codecs: tuple[Any, ...]) -> bool:
    """Whether the array's own codecs hold a transpose and a sharding codec, which follows it.

    Only the array's own codecs: inside a shard, each codec is validated against the
    chunk the codecs before it leave.
    """
    names = [_codec_name(codec) for codec in codecs]
    return "transpose" in names and "sharding_indexed" in names


def _spliced(codecs: tuple[Any, ...]) -> list[Any]:
    """The codecs up to the array -> bytes codec, each shard replaced by the codecs it holds.

    The codecs inside a shard see the fill value as the codecs before the shard leave
    it, so these codecs can encode the same fill values as `codecs`.
    """
    out: list[Any] = []
    for codec in codecs:
        if _is_array_array(codec):
            out.append(codec)
        elif _codec_name(codec) == "sharding_indexed":
            return out + _spliced(codec["configuration"]["codecs"])
        else:
            return [*out, codec]
    return out


def _verdict(document: dict[str, Any]) -> Exception | None:
    """The error zarr-python refuses `document` with, or None if it accepts it."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ArrayV3Metadata.from_dict(document)
    except Exception as error:  # noqa: BLE001
        return error
    return None


def _raised_encoding_fill_value(error: Exception) -> bool:
    """Whether `error` was raised by a codec encoding the fill value."""
    frames = traceback.extract_tb(error.__traceback__)
    return isinstance(error, ValueError) and any(f.name == "resolve_metadata" for f in frames)


@pytest.mark.parametrize("slice_", list(_DOCUMENTS))
@given(data=st.data())
def test_refuses_only_fill_values_the_codecs_cannot_encode(
    slice_: str, data: st.DataObject
) -> None:
    """zarr-python accepts each document, or refuses its fill value as the codecs
    cannot encode it, and then refuses it too with the shards' codecs moved out of
    the shards."""
    # As a reader sees it: zarr-metadata draws JSON arrays as tuples.
    document = json.loads(json.dumps(data.draw(_DOCUMENTS[slice_])))
    if _shard_behind_transpose(document["codecs"]):
        reject()  # "sharding behind transpose" in _DIVERGENCES
    with zarr.config.set({"array.rectilinear_chunks": True}):
        error = _verdict(document)
        if error is None:
            event("accepted")
            return
        event("refused")
        assert _raised_encoding_fill_value(error), (
            f"zarr-python refuses a document the spec allows: {error!r}"
        )
        spliced = copy.deepcopy(document)
        spliced["codecs"] = _spliced(document["codecs"])
        assert _verdict(spliced) is not None, (
            f"zarr-python refuses the fill value inside a shard ({error!r}) but accepts it "
            "with the shard's codecs outside the shard"
        )


def _document(
    data_type: Any, fill_value: Any, codecs: list[Any], chunk_shape: tuple[int, ...] = (4,)
) -> dict[str, Any]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": chunk_shape,
        "data_type": data_type,
        "fill_value": fill_value,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": codecs,
    }


@pytest.mark.xfail(strict=True, raises=ValueError, reason=_DIVERGENCES["data type spellings"])
@pytest.mark.parametrize(
    "data_type",
    [
        {"name": "int8"},
        {"name": "int8", "configuration": {}},
        {"name": "int8", "must_understand": True},
    ],
)
def test_data_type_object_spelling(data_type: dict[str, Any]) -> None:
    ArrayV3Metadata.from_dict(_document(data_type, 0, [{"name": "bytes"}]))


@pytest.mark.xfail(strict=True, raises=ValueError, reason=_DIVERGENCES["raw bits"])
def test_raw_bits_data_type() -> None:
    ArrayV3Metadata.from_dict(_document("r16", [0, 0], [{"name": "bytes"}]))


@pytest.mark.xfail(strict=True, raises=TypeError, reason=_DIVERGENCES["bytes fill value"])
def test_bytes_fill_value_as_integers() -> None:
    ArrayV3Metadata.from_dict(_document("bytes", [1, 2], [{"name": "vlen-bytes"}]))


@pytest.mark.xfail(strict=True, raises=ValueError, reason=_DIVERGENCES["sharding behind transpose"])
def test_sharding_behind_transpose() -> None:
    # The transposed chunk is (4, 2), which inner chunks of (4, 1) divide.
    sharding = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (4, 1),
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        },
    }
    transpose = {"name": "transpose", "configuration": {"order": (1, 0)}}
    ArrayV3Metadata.from_dict(_document("uint8", 0, [transpose, sharding], chunk_shape=(2, 4)))


@pytest.mark.skipif(not _HAS_CAST_VALUE_RS, reason="cast-value-rs not installed")
@pytest.mark.xfail(strict=True, raises=ValueError, reason=_DIVERGENCES["cast_value hex scalar map"])
def test_cast_value_scalar_map_hex_float() -> None:
    cast = {
        "name": "cast_value",
        "configuration": {
            "data_type": "float32",
            "scalar_map": {"encode": [[1, "0x7fc00000"]]},
        },
    }
    ArrayV3Metadata.from_dict(
        _document("uint8", 0, [cast, {"name": "bytes", "configuration": {"endian": "little"}}])
    )


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_DIVERGENCES["empty rectilinear edges"]
)
def test_zarr_metadata_refuses_empty_rectilinear_edges() -> None:
    document = _document("bool", False, [{"name": "bytes"}], chunk_shape=(0,))
    document["chunk_grid"] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[]]},
    }
    assert len(read_array_metadata_v3(document).problems) != 0


@pytest.mark.xfail(
    strict=True, raises=MemoryError, reason=_DIVERGENCES["expanded rectilinear runs"]
)
def test_rectilinear_run_is_not_expanded() -> None:
    """A run of 2**62 chunks of length 2 is read without listing them."""
    document = _document("uint8", 0, [{"name": "bytes"}], chunk_shape=(2**63 - 1,))
    document["chunk_grid"] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[[2, 2**62]]]},
    }
    with zarr.config.set({"array.rectilinear_chunks": True}):
        ArrayV3Metadata.from_dict(document)
