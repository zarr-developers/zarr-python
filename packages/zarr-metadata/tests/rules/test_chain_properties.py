"""Property tests over codec chains drawn from the codec types themselves.

The existing totality test feeds arbitrary JSON to the document
validators, which is the right guard for the structural layer and no guard
at all for this one: a random object never names a codec, so it never
dispatches an entity rule. `test_the_chain_strategy_reaches_the_rules`
exists so that stays visible — a refactor that silently stops dispatching
fails here instead of staying green.
"""

from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

from hypothesis import HealthCheck, given, settings

import zarr_metadata.v3.codec
from tests.rules.strategies import (
    ARRAY_ARRAY,
    ARRAY_BYTES,
    BYTES_BYTES,
    codec_chains,
    corrupted_chains,
    document,
    rank_matched_shards,
    valid_documents,
)
from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    ArrayArrayCodec,
    ArrayBytesCodec,
    CodecEntity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# Drawing a codec chain and validating a document is slow per example and
# the outer functions below run a whole @given loop of their own, so the two
# timing checks are suppressed; nothing else is.
_SLOW = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


def test_the_strategies_cover_every_codec_the_package_models() -> None:
    # The kind tuples are hand-written; a new codec module must join one.
    drawn = {
        entry.__annotations__["name"].__args__[0]
        for entry in (*ARRAY_ARRAY, *ARRAY_BYTES, *BYTES_BYTES)
    }
    modelled = {
        value
        for info in pkgutil.iter_modules(zarr_metadata.v3.codec.__path__)
        if not info.name.startswith("_")
        for attribute, value in vars(
            __import__(f"zarr_metadata.v3.codec.{info.name}", fromlist=["_"])
        ).items()
        if attribute.endswith("_CODEC_NAME") and isinstance(value, str)
    }
    assert modelled == drawn
    for kinds, expected in ((ARRAY_ARRAY, ArrayArrayCodec), (ARRAY_BYTES, ArrayBytesCodec)):
        for entry in kinds:
            name = entry.__annotations__["name"].__args__[0]
            entity = CORE_AND_EXTENSIONS.resolve(CodecEntity, name)
            assert entity is not None, name
            assert issubclass(entity, expected)


@given(codec_chains())
@_SLOW
def test_a_well_ordered_chain_always_produces_a_verdict(codecs: tuple[object, ...]) -> None:
    """Untrusted input gets judged, never raises, however odd the values."""
    assert isinstance(validate_array_metadata_v3(document(codecs)), tuple)


# The rules a chain of arbitrary codec configurations reliably exercises,
# with the observed rate over 600 examples. The three chain rules absent
# here are out of this strategy's reach by construction, not by accident:
# pipeline ordering cannot fire because the chain is assembled in order,
# and variable-length data types and variable-size index codecs need a
# correlated data type and index chain that only a hand-written case
# supplies. Those have their own tests in `test_v3_array_rules.py`.
_WITNESSES: Mapping[str, str] = {
    "rank against the incoming array": "incoming array has",  # 63%
    "positive chunk extents": "expected an integer >= 1",  # 34%
    "transpose order is a permutation": "expected a permutation",  # 21%
    "endianness for multi-byte types": "endian is required",  # 20%
}


def test_the_chain_strategy_reaches_the_rules() -> None:
    reached: set[str] = set()

    @given(codec_chains())
    @_SLOW
    def sample(codecs: tuple[object, ...]) -> None:
        messages = [problem.message for problem in validate_array_metadata_v3(document(codecs))]
        reached.update(
            label
            for label, witness in _WITNESSES.items()
            if any(witness in message for message in messages)
        )

    sample()
    assert reached == set(_WITNESSES)


def test_the_shard_strategy_reaches_chunk_geometry() -> None:
    # A shard whose rank matches is what exposes divisibility: drawn freely
    # it almost never has the right rank, and the rank check returns first.
    reached = [False]

    @given(rank_matched_shards())
    @_SLOW
    def sample(codecs: tuple[object, ...]) -> None:
        if any(
            "does not evenly divide" in problem.message
            for problem in validate_array_metadata_v3(document(codecs))
        ):
            reached[0] = True

    sample()
    assert reached[0]


@given(valid_documents())
@_SLOW
def test_documents_valid_by_construction_are_accepted(doc: Mapping[str, object]) -> None:
    """The accept side: extents drawn, then inner shapes from their divisors.

    Every fix to this layer has made it stricter, and almost nothing drawn
    from the codec types is valid, so this is the only property guarding
    the other direction over generated input.
    """
    problems = validate_array_metadata_v3(doc)
    assert problems == (), [problem.message for problem in problems]


@given(corrupted_chains())
@_SLOW
def test_an_ill_typed_configuration_member_is_reported_not_raised(
    codecs: tuple[object, ...],
) -> None:
    """A member of the wrong type stands its rules down; it never crashes one.

    This is the property the `reads` gate exists for, and the only generated
    input that exercises it: `st.from_type` honours the TypedDicts, so every
    other strategy here produces well-typed members. Without this, four
    separate one-token changes to the gate leave the suite green while
    `validate_*` raises `TypeError` on ordinary malformed metadata.
    """
    problems = validate_array_metadata_v3(document(codecs))
    assert isinstance(problems, tuple)


def test_the_corrupting_strategy_reaches_the_gate() -> None:
    # It has to produce ill-typed members, or the property above is vacuous.
    reached = [False]

    @given(corrupted_chains())
    @_SLOW
    def sample(codecs: tuple[object, ...]) -> None:
        if any(
            problem.kind == "invalid_type"
            for problem in validate_array_metadata_v3(document(codecs))
        ):
            reached[0] = True

    sample()
    assert reached[0]
