"""Metadata protocol (v1).

Defines the structural interface for objects that can be serialized
to and deserialized from JSON dictionaries.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

type JSON = str | int | float | bool | dict[str, JSON] | list[JSON] | None


@runtime_checkable
class Metadata(Protocol):
    """Protocol for objects that round-trip through JSON dictionaries."""

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Create an instance from a JSON dictionary."""
        ...

    def to_dict(self) -> dict[str, JSON]:
        """Serialize to a JSON dictionary."""
        ...
