"""Tests for array-spec propagation through a codec chain.

The property under test: every codec is judged against the array it
*receives*, which is the document's chunk only for the first codec in
the chain. Anything that transforms the array — a transpose, a cast, a
shard — changes what the next codec sees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.rules._spec import (
    NOTHING_KNOWN,
    ArraySpec,
    propagate,
    transitions_registered,
)
from zarr_metadata.v3.codec.kind import ARRAY_ARRAY_CODEC_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping


def _doc(codecs: tuple[object, ...], chunk: tuple[int, ...] = (6, 4)) -> Mapping[str, object]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (12, 8),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk}},
        "chunk_key_encoding": "default",
        "codecs": codecs,
    }


def _transpose(*order: int) -> Mapping[str, object]:
    return {"name": "transpose", "configuration": {"order": order}}


def _shard(
    inner: tuple[int, ...],
    codecs: tuple[object, ...] = ("bytes",),
    index_codecs: tuple[object, ...] = ({"name": "bytes", "configuration": {"endian": "little"}},),
) -> Mapping[str, object]:
    return {
        "name": "sharding_indexed",
        "configuration": {"chunk_shape": inner, "codecs": codecs, "index_codecs": index_codecs},
    }


# (document, expected verdict). The chunk is (6, 4); a transpose (1, 0) in
# front of a shard means the shard receives (4, 6), and the verdict must
# follow the transposed shape, not the grid.
CASES: dict[str, tuple[Mapping[str, object], bool]] = {
    "shard-alone-divides": (_doc((_shard((3, 2)),)), True),
    "shard-alone-does-not-divide": (_doc((_shard((4, 3)),)), False),
    # Regression: before propagation these two verdicts were reversed.
    "transpose-then-shard-divides-transposed": (_doc((_transpose(1, 0), _shard((2, 3)))), True),
    "transpose-then-shard-does-not-divide-transposed": (
        _doc((_transpose(1, 0), _shard((3, 2)))),
        False,
    ),
    "two-transposes-cancel": (_doc((_transpose(1, 0), _transpose(1, 0), _shard((3, 2)))), True),
    # Inside a shard the incoming array is the inner chunk, so a nested
    # transpose is judged against the inner chunk's rank, and a nested
    # shard against the transposed inner chunk.
    "nested-transpose-matches-inner-rank": (
        _doc((_shard((3, 2), codecs=(_transpose(1, 0), "bytes")),)),
        True,
    ),
    "nested-shard-follows-nested-transpose": (
        # inner chunk (3, 2) transposed -> (2, 3); nested shard (2, 1) divides it.
        _doc((_shard((3, 2), codecs=(_transpose(1, 0), _shard((2, 1)))),)),
        True,
    ),
    "nested-shard-violates-transposed-inner": (
        # inner chunk (3, 2) transposed -> (2, 3); nested shard (3, 1): 3 does not divide 2.
        _doc((_shard((3, 2), codecs=(_transpose(1, 0), _shard((3, 1)))),)),
        False,
    ),
    # An unknown codec might change the shape, so downstream geometry
    # declines rather than guessing — an otherwise-invalid shard passes.
    "unknown-codec-stops-propagation": (_doc(({"name": "zfpy"}, _shard((4, 3)))), True),
    # The shard index is a uint64 array, so a bytes codec inside
    # `index_codecs` needs an endianness like any multi-byte encoding.
    "index-codecs-bare-bytes-needs-endian": (
        _doc((_shard((3, 2), index_codecs=("bytes", "crc32c")),)),
        False,
    ),
}


@pytest.mark.parametrize(("doc", "valid"), CASES.values(), ids=list(CASES))
def test_verdict_follows_the_incoming_array(doc: Mapping[str, object], valid: bool) -> None:
    problems = validate_array_metadata_v3(doc)
    assert (len(problems) == 0) is valid, [str(p) for p in problems]


def test_error_locates_the_offending_shard() -> None:
    problems = validate_array_metadata_v3(_doc((_transpose(1, 0), _shard((3, 2)))))
    assert [(p.loc, p.kind) for p in problems] == [
        (("codecs", 1, "configuration", "chunk_shape", 0), "invalid_value")
    ]


def test_propagate_yields_incoming_spec_per_codec() -> None:
    from zarr_metadata.rules._registry import entity_configuration
    from zarr_metadata.v3._extension_points import CODECS

    chain = (_transpose(1, 0), "bytes", "crc32c")
    start = ArraySpec((6, 4), "uint8")
    seen = list(propagate(chain, start, lambda c: entity_configuration(CODECS, c)))
    incoming = [spec for _, _, spec in seen]
    assert incoming[0] == ArraySpec((6, 4), "uint8")  # transpose receives the chunk
    assert incoming[1] == ArraySpec((4, 6), "uint8")  # bytes receives the transposed chunk
    # past array->bytes: no array, so no shape; the type carries through
    assert incoming[2] == ArraySpec(None, "uint8")


def test_cast_value_changes_the_downstream_data_type() -> None:
    from zarr_metadata.rules._registry import entity_configuration
    from zarr_metadata.v3._extension_points import CODECS

    chain = ({"name": "cast_value", "configuration": {"data_type": "float32"}}, "bytes")
    start = ArraySpec((6, 4), "uint8")
    seen = list(propagate(chain, start, lambda c: entity_configuration(CODECS, c)))
    assert seen[1][2] == ArraySpec((6, 4), "float32")


def test_unknown_codec_yields_nothing_known() -> None:
    from zarr_metadata.rules._registry import entity_configuration
    from zarr_metadata.v3._extension_points import CODECS

    start = ArraySpec((6, 4), "uint8")
    seen = list(
        propagate(({"name": "zfpy"}, "bytes"), start, lambda c: entity_configuration(CODECS, c))
    )
    assert seen[1][2] is NOTHING_KNOWN


def test_every_array_array_codec_registers_a_transition() -> None:
    # A modelled array->array codec with no transition is treated as
    # unknown and stops propagation — safe, but silently weaker than
    # intended. Make it a decision, not an omission.
    assert set(ARRAY_ARRAY_CODEC_NAMES) <= transitions_registered() | {"scale_offset"}


def test_error_transition_for_a_non_array_array_codec() -> None:
    from zarr_metadata.rules._spec import spec_transition

    with pytest.raises(ValueError, match="only array->array codecs"):

        @spec_transition("gzip")
        def _nope(configuration: object, incoming: ArraySpec) -> ArraySpec:  # pragma: no cover
            return incoming
