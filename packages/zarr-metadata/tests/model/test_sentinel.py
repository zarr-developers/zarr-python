"""Tests for the pickling and copying behavior of the `UNSET` sentinel.

The sentinel's contract is identity, so it must never be reconstructed from
state. typing_extensions >= 4.16 pickles sentinels by reference (a lookup of
the sentinel's name on its defining module), which preserves the singleton
across process boundaries; these tests pin that behavior, since models hold
`UNSET` as field values and must survive pickling and deep-copying.

The model round-trip tests compare whole structures: dataclass equality
compares every field, and `UNSET` compares by identity, so an impostor
sentinel produced by state-based pickling would fail the equality check.
"""

from __future__ import annotations

import copy
import pickle

import pytest
from typing_extensions import Sentinel

from zarr_metadata.model import (
    UNSET,
    ZarrV2ArrayMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
)

# Whole-model cases covering the states we know are problematic for
# serialization: every optional-key field in the UNSET (absent) state, the
# same fields in the present state (including present-but-empty, which must
# stay distinct from absent), and UNSET nested inside consolidated metadata.
MODEL_CASES = {
    "array-v3-dimension-names-unset": ZarrV3ArrayMetadata.create_default(shape=(4,)),
    "array-v3-dimension-names-set": ZarrV3ArrayMetadata.create_default(shape=(2, 2)).update(
        dimension_names=("x", None)
    ),
    "array-v2-attributes-unset": ZarrV2ArrayMetadata.create_default(shape=(4,)),
    "array-v2-attributes-empty": ZarrV2ArrayMetadata.create_default(shape=(4,), attributes={}),
    "group-v2-attributes-unset": ZarrV2GroupMetadata.create_default(),
    "group-v2-attributes-set": ZarrV2GroupMetadata.create_default(attributes={"a": 1}),
    "group-v3-consolidated-unset": ZarrV3GroupMetadata.create_default(),
    "group-v3-consolidated-with-unset-inside": ZarrV3GroupMetadata.create_default(
        consolidated_metadata=ZarrV3ConsolidatedMetadata(
            metadata={
                "child": ZarrV3ArrayMetadata.create_default(shape=(4,)),
                "subgroup": ZarrV3GroupMetadata.create_default(),
            }
        )
    ),
}


def test_unset_pickle_round_trip_preserves_identity() -> None:
    restored = pickle.loads(pickle.dumps(UNSET))
    assert restored is UNSET


def test_unset_copy_preserves_identity() -> None:
    assert copy.copy(UNSET) is UNSET
    assert copy.deepcopy(UNSET) is UNSET


@pytest.mark.parametrize("model", MODEL_CASES.values(), ids=MODEL_CASES.keys())
def test_model_pickle_round_trip(
    model: ZarrV2ArrayMetadata | ZarrV3ArrayMetadata | ZarrV2GroupMetadata | ZarrV3GroupMetadata,
) -> None:
    restored = pickle.loads(pickle.dumps(model))
    assert restored == model


@pytest.mark.parametrize("model", MODEL_CASES.values(), ids=MODEL_CASES.keys())
def test_model_deepcopy(
    model: ZarrV2ArrayMetadata | ZarrV3ArrayMetadata | ZarrV2GroupMetadata | ZarrV3GroupMetadata,
) -> None:
    assert copy.deepcopy(model) == model


def test_non_importable_sentinel_fails_to_pickle() -> None:
    """Sentinels pickle by reference, never by state. A sentinel that is not
    an importable attribute of its module has no reference to pickle, so
    dumping it must fail loudly — a successful dump here would mean the
    implementation regressed to state-based pickling, which would produce
    identity-breaking impostor objects on the receiving side."""
    local_sentinel = Sentinel("local_sentinel")
    with pytest.raises((pickle.PicklingError, TypeError)):
        pickle.dumps(local_sentinel)
