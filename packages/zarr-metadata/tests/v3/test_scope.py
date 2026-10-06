"""The algebra of scopes: value semantics for `Context`, claims, refinement, disagreements and joins."""

from __future__ import annotations

import pickle

import pytest

from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, Context


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (Context.of(GZIP_CODEC, BYTES_CODEC), Context.of(BYTES_CODEC, GZIP_CODEC), True),
        (CORE, pickle.loads(pickle.dumps(CORE)), True),
        (CORE, CORE_AND_EXTENSIONS, False),
        (Context.of(GZIP_CODEC), Context.of(GZIP_CODEC, ZSTD_CODEC), False),
        (Context.of(), Context.of(), True),
    ],
    ids=["order", "pickle", "core-vs-extensions", "subset", "empty"],
)
def test_a_scope_is_equal_to_another_by_the_definitions_it_files(
    left: Context, right: Context, equal: bool
) -> None:
    """Two scopes are one when they file the same definitions under the same names, however they were built; equal scopes hash alike."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


def test_a_scope_is_a_set_member() -> None:
    """A scope hashes, so it can key a dict or sit in a set, which it could not when its tables were mapping proxies."""
    assert len({CORE, CORE_AND_EXTENSIONS, pickle.loads(pickle.dumps(CORE))}) == 2
