"""Hypothesis strategies over codec chains, derived from the codec types.

Two kinds of strategy, because the two halves of "is this validator right?"
need opposite inputs.

`codec_chains` draws each codec with `st.from_type` over its own
`TypedDict`, so the values are structurally plausible and almost always
semantically wrong. That is what the rejection side wants. The chain is
assembled by *kind* — `array->array`* `array->bytes` `bytes->bytes`* —
which guarantees an `array->bytes` codec is present: measured over 3000
documents, an entity rule is dispatched for 100% of ordered chains and one
reports a problem for 86%, against 78% for a flat list of the same codecs.
Nothing short-circuits on a misordered chain, so the gain is coverage of
the array->bytes codecs rather than avoided early exit.

What this cannot reach, because `st.from_type` honours the TypedDicts
exactly, is a configuration member of the wrong *type* —
`corrupted_chains` exists for that.

`valid_documents` builds documents that must be accepted, by construction:
extents are drawn, then inner chunk shapes are drawn from their divisors.
Nothing else here can test the accept side, because fewer than 5% of
`from_type` documents are valid.

`st.from_type` handles `ReadOnly`, `closed=True`, `NotRequired` and
`Literal` unaided. It cannot resolve the recursive `JSONValue` alias, so
this module registers one for it; without that, every codec whose
configuration admits arbitrary JSON (`sharding_indexed`, `cast_value`,
`scale_offset`) fails to resolve.
"""

from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING, Any, cast

from hypothesis import strategies as st

from zarr_metadata._common import JSONValue
from zarr_metadata.v3.codec.blosc import BloscCodecObject
from zarr_metadata.v3.codec.bytes import BytesCodecObject
from zarr_metadata.v3.codec.cast_value import CastValueCodecObject
from zarr_metadata.v3.codec.crc32c import Crc32cCodecObject
from zarr_metadata.v3.codec.gzip import GzipCodecObject
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodecObject
from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecObject
from zarr_metadata.v3.codec.transpose import TransposeCodecObject
from zarr_metadata.v3.codec.zstd import ZstdCodecObject

if TYPE_CHECKING:
    from collections.abc import Sequence

JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=5,
)
# `JSONValue` is a `TypeAliasType`, which `register_type_strategy` does not
# accept in its signature but does resolve at runtime — it is exactly the
# forward reference `from_type` fails on.
st.register_type_strategy(JSONValue, JSON_VALUES)  # pyright: ignore[reportArgumentType]

# The codec TypedDicts, by pipeline kind. Hand-written because there is no
# name-to-type table to derive it from; `test_chain_properties.py` asserts it
# covers every codec the package models.
ARRAY_ARRAY = (TransposeCodecObject, CastValueCodecObject, ScaleOffsetCodecObject)
ARRAY_BYTES = (BytesCodecObject, ShardingIndexedCodecObject)
BYTES_BYTES = (BloscCodecObject, Crc32cCodecObject, GzipCodecObject, ZstdCodecObject)


_LITTLE: dict[str, object] = {"name": "bytes", "configuration": {"endian": "little"}}


def _any_of(types: Sequence[type]) -> st.SearchStrategy[Any]:
    return st.one_of(*[st.from_type(entry) for entry in types])


def codec_chains() -> st.SearchStrategy[tuple[object, ...]]:
    """Chains in the shape the spec requires, with arbitrary configurations."""
    return st.tuples(
        st.lists(_any_of(ARRAY_ARRAY), max_size=2),
        _any_of(ARRAY_BYTES),
        st.lists(_any_of(BYTES_BYTES), max_size=2),
    ).map(lambda parts: (*parts[0], parts[1], *parts[2]))


@st.composite
def rank_matched_shards(draw: st.DrawFn) -> tuple[object, ...]:
    """Chains whose shard has the document's rank, with arbitrary extents.

    `codec_chains` draws a shard's `chunk_shape` freely, so it almost never
    has the right rank and the rank check short-circuits before geometry is
    reached. Matching the rank is what exposes divisibility to the fuzzer.
    """
    extents = tuple(draw(st.lists(st.integers(min_value=1, max_value=48), min_size=2, max_size=2)))
    return (
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": extents,
                "codecs": (_LITTLE,),
                "index_codecs": (_LITTLE,),
            },
        },
    )


def document(codecs: object, **overrides: object) -> dict[str, object]:
    """A v3 array document around `codecs`, valid apart from what is passed."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (64, 64),
        "data_type": "uint16",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (32, 32)}},
        "chunk_key_encoding": "default",
        "codecs": codecs,
        **overrides,
    }


def _divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


@st.composite
def valid_documents(draw: st.DrawFn) -> dict[str, object]:
    """Documents that must validate clean, correlated by construction.

    A transpose permutes the grid and the shard's inner shape follows it,
    so the inner extents are checked against the axis they actually meet —
    including a rectilinear axis whose chunks differ, where only a common
    divisor of every length will do.
    """
    rank = draw(st.integers(min_value=1, max_value=3))
    rectilinear = draw(st.booleans())
    # Per axis: the lengths its chunks take, and an extent dividing all of them.
    lengths: list[tuple[int, ...]] = []
    for _ in range(rank):
        if rectilinear:
            axis = tuple(draw(st.lists(st.sampled_from([8, 12, 16, 24]), min_size=1, max_size=3)))
        else:
            axis = (draw(st.sampled_from([8, 12, 16, 24])),)
        lengths.append(axis)
    inner = [draw(st.sampled_from(_divisors(gcd(*axis, axis[0])))) for axis in lengths]
    order = tuple(draw(st.permutations(range(rank))))

    if rectilinear:
        grid: dict[str, object] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": tuple(lengths)},
        }
        shape = tuple(sum(axis) for axis in lengths)
    else:
        grid = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(axis[0] for axis in lengths)},
        }
        shape = tuple(axis[0] * 2 for axis in lengths)

    shard = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": tuple(inner[axis] for axis in order),
            "codecs": (_LITTLE,),
            "index_codecs": (_LITTLE,),
        },
    }
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "uint16",
        "fill_value": 0,
        "chunk_grid": grid,
        "chunk_key_encoding": "default",
        "codecs": ({"name": "transpose", "configuration": {"order": order}}, shard),
    }


@st.composite
def corrupted_chains(draw: st.DrawFn) -> tuple[object, ...]:
    """A well-typed chain with one configuration member replaced by any JSON.

    `st.from_type` honours the TypedDicts, so it never produces an ill-typed
    member — and an ill-typed member is exactly what the `reads` gate exists
    to stand rules down for. Without this, the gate that makes
    `configuration["level"]` safe inside a rule has no generated coverage,
    and a validator that raises `TypeError` on ordinary malformed metadata
    looks identical to one that does not.
    """
    chain = list(draw(codec_chains()))
    index = draw(st.integers(min_value=0, max_value=len(chain) - 1))
    codec = chain[index]
    if not isinstance(codec, dict):
        return tuple(chain)
    entry = cast("dict[str, object]", dict(codec))
    configuration = entry.get("configuration")
    if not isinstance(configuration, dict) or len(cast("dict[str, object]", configuration)) == 0:
        return tuple(chain)
    members = cast("dict[str, object]", dict(configuration))
    member = draw(st.sampled_from(sorted(members)))
    members[member] = draw(JSON_VALUES)
    entry["configuration"] = members
    chain[index] = entry
    return tuple(chain)


__all__ = [
    "ARRAY_ARRAY",
    "ARRAY_BYTES",
    "BYTES_BYTES",
    "JSON_VALUES",
    "codec_chains",
    "corrupted_chains",
    "document",
    "rank_matched_shards",
    "valid_documents",
]
