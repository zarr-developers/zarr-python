"""Ways for a test to look into JSON it has just been handed, without lying about its type."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def entry_at(document: object, *steps: str | int) -> object:
    """The value `steps` lead to in `document`: a key into an object, an index into an array."""
    node = document
    for step in steps:
        if isinstance(step, str):
            assert isinstance(node, Mapping), (steps, node)
            node = cast("Mapping[str, object]", node)[step]
        else:
            assert isinstance(node, Sequence), (steps, node)
            assert not isinstance(node, str), (steps, node)
            node = cast("Sequence[object]", node)[step]
    return node


def configuration_of(field: object) -> Mapping[str, object]:
    """The configuration of a metadata field spelled as an object; empty if it has none."""
    assert isinstance(field, Mapping), field
    configuration = cast("Mapping[str, object]", field).get("configuration", {})
    assert isinstance(configuration, Mapping), configuration
    return cast("Mapping[str, object]", configuration)
