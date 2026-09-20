"""The correspondence between an entity's dataclass and its TypedDicts.

A configuration TypedDict unpacked is exactly the dataclass constructor's
signature, and the object TypedDict is exactly what `to_json` returns.
Asserting the first is what lets the dataclass carry its own checks
instead of a hand-written per-member table elsewhere: the two cannot
drift, because one drifting makes this fail.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, get_type_hints

import pytest

from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS
from zarr_metadata.v3.codec.blosc import BloscCodec, BloscCodecConfiguration

if TYPE_CHECKING:
    from zarr_metadata.v3._entity import MetadataEntity

# Each registered entity, paired with the TypedDict its constructor mirrors.
CONFIGURATIONS: dict[str, tuple[type[MetadataEntity], type]] = {
    "blosc": (BloscCodec, BloscCodecConfiguration),
}


@pytest.mark.parametrize(
    ("entity", "configuration"), CONFIGURATIONS.values(), ids=list(CONFIGURATIONS)
)
def test_the_constructor_mirrors_the_configuration(
    entity: type[MetadataEntity], configuration: type
) -> None:
    # `must_understand` belongs to the object, not the configuration, so it
    # is the one field the two deliberately do not share.
    fields = {field.name for field in dataclasses.fields(entity)} - {"must_understand"}
    assert fields == set(get_type_hints(configuration))


def test_every_registered_entity_is_checked_here() -> None:
    registered = {
        identifier for entities in CORE_AND_EXTENSIONS.entities.values() for identifier in entities
    }
    assert registered == set(CONFIGURATIONS)


def test_core_is_a_subset_of_core_and_extensions() -> None:
    for field, entities in CORE.entities.items():
        assert entities.items() <= CORE_AND_EXTENSIONS.entities[field].items()


def test_a_name_out_of_scope_resolves_to_nothing() -> None:
    # Not an error: an unmodelled extension is left unjudged, not rejected.
    assert CORE.resolve("codecs", "mycorp.secret") is None
    assert CORE.resolve("codecs", "blosc") is BloscCodec


def test_an_entity_round_trips_through_its_json_form() -> None:
    original = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
            "blocksize": 0,
            "typesize": 4,
        },
    }
    codec, problems = BloscCodec.coerce(original, CORE)
    assert problems == ()
    assert codec is not None
    assert codec.to_json() == original
