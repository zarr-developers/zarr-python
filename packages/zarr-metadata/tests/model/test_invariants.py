"""What holds for every v3 document the strategies generate: how a model compares with the same document read in another scope, or written in another order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given

from zarr_metadata.model import ZarrV3ArrayMetadata, ZarrV3GroupMetadata
from zarr_metadata.testing.strategies import (
    array_documents,
    consolidated_group_documents,
    group_documents,
)
from zarr_metadata.v3.definition import CORE_CONTEXT, ZARR_PYTHON_CONTEXT, ZarrV3Context


@pytest.mark.parametrize(
    "poorer",
    [
        ZarrV3Context(),
        CORE_CONTEXT,
        ZARR_PYTHON_CONTEXT.copy(update={"data_types": {}}),
        ZARR_PYTHON_CONTEXT.copy(update={"codecs": {}}),
    ],
    ids=["empty", "core", "no-data-types", "no-codecs"],
)
@given(document=array_documents())
def test_every_array_read_in_a_richer_scope_refines_the_one_a_poorer_scope_read(
    poorer: ZarrV3Context, document: Mapping[str, Any]
) -> None:
    """`refined_in` returns the model that the richer scope reads, and that model refines the one it came from, whatever the richer scope learns of the chunk each codec receives."""
    low = ZarrV3ArrayMetadata.from_json(document, context=poorer)
    high = low.refined_in(ZARR_PYTHON_CONTEXT)
    assert high == ZarrV3ArrayMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT)
    assert high.refines(low)
    assert low.refines(low)


def _reordered(value: object) -> Any:  # noqa: ANN401
    """`value` with the members of every object it holds, at any depth, in the reverse order. The items of an array keep their order."""
    if isinstance(value, Mapping):
        return {key: _reordered(value[key]) for key in reversed(value)}
    if isinstance(value, (list, tuple)):
        return [_reordered(item) for item in value]
    return value


def _assert_one_model(
    model: ZarrV3ArrayMetadata | ZarrV3GroupMetadata,
    reordered: ZarrV3ArrayMetadata | ZarrV3GroupMetadata,
) -> None:
    assert reordered == model
    assert hash(reordered) == hash(model)
    assert reordered.refines(model)  # pyright: ignore[reportArgumentType]
    assert model.refines(reordered)  # pyright: ignore[reportArgumentType]


@given(document=array_documents())
def test_every_array_reads_as_one_model_whatever_order_its_members_are_in(
    document: Mapping[str, Any],
) -> None:
    """The order of the members of an object means nothing, at any depth: in the document, in a configuration, and in a configuration that holds declarations."""
    _assert_one_model(
        ZarrV3ArrayMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT),
        ZarrV3ArrayMetadata.from_json(_reordered(document), context=ZARR_PYTHON_CONTEXT),
    )


@given(document=group_documents())
def test_every_group_reads_as_one_model_whatever_order_its_members_are_in(
    document: Mapping[str, Any],
) -> None:
    _assert_one_model(
        ZarrV3GroupMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT),
        ZarrV3GroupMetadata.from_json(_reordered(document), context=ZARR_PYTHON_CONTEXT),
    )


@given(document=consolidated_group_documents())
def test_every_listing_reads_as_one_model_whatever_order_its_members_are_in(
    document: Mapping[str, Any],
) -> None:
    """A group lists the documents of its nodes. Each is read, so the listing is one model whatever order the members of those documents are in."""
    listings = [
        ZarrV3GroupMetadata.from_json(each, context=ZARR_PYTHON_CONTEXT).consolidated()
        for each in (document, _reordered(document))
    ]
    assert listings[0] is not None
    assert listings[1] is not None
    assert listings[1] == listings[0]
    assert hash(listings[1]) == hash(listings[0])
    assert listings[1].refines(listings[0])
    assert listings[0].refines(listings[1])
