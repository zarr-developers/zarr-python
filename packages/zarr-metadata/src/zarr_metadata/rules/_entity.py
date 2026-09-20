"""Reading a named entity's configuration.

An entity's `configuration` is only worth reading once its shape has been
vouched for, and there are two useful strictnesses. `entity_configuration`
is all-or-nothing, for callers that derive something from the whole
configuration (a spec transition). `run_entity_rules` wants the finer
per-member judgment and reaches for `configuration_mapping` plus the shape
verdict directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.rules._engine import as_string_mapping
from zarr_metadata.v3._shape import blocking_problems, validate_known_entity_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._extension_points import ExtensionPointField


def entity_configuration(field: ExtensionPointField, value: object) -> Mapping[str, object] | None:
    """`value`'s configuration if every modelled field is usable, else None.

    The all-or-nothing gate `propagate` needs: a spec transition reads the
    configuration to compute what the next codec receives, so one unusable
    member makes the whole outgoing spec a guess. `run_entity_rules` uses
    the finer per-member gate instead. `unknown_key` problems do not make
    an entity unusable; anything else does.
    """
    verdict = validate_known_entity_metadata(field, value)
    if verdict is None or len(blocking_problems(verdict)) != 0:
        return None
    return configuration_mapping(value)


def configuration_mapping(value: object) -> Mapping[str, object] | None:
    """`value`'s configuration mapping, with no judgment of its contents."""
    mapping = as_string_mapping(value)
    if mapping is None:
        # Bare-string metadata is the canonical spelling for entities whose
        # configuration is optional. Rules still need a real mapping to run
        # against, especially when they judge a missing optional member.
        return {} if isinstance(value, str) else None
    if "configuration" not in mapping:
        return {}
    return as_string_mapping(mapping["configuration"])


__all__ = [
    "configuration_mapping",
    "entity_configuration",
]
