from typing import Any

__all__ = [
    "BaseZarrError",
    "ContainsArrayAndGroupError",
    "ContainsArrayError",
    "ContainsGroupError",
    "MetadataValidationError",
    "NodeTypeValidationError",
]


class BaseZarrError(ValueError):
    """
    Base error which all zarr errors are sub-classed from.
    """

    _msg = ""

    def __init__(self, *args: Any) -> None:
        super().__init__(self._msg.format(*args))


class ContainsGroupError(BaseZarrError):
    """Raised when a group already exists at a certain path."""

    _msg = "A group exists in store {!r} at path {!r}."


class ContainsArrayError(BaseZarrError):
    """Raised when an array already exists at a certain path."""

    _msg = "An array exists in store {!r} at path {!r}."


class ContainsArrayAndGroupError(BaseZarrError):
    """Raised when both array and group metadata are found at the same path."""

    _msg = (
        "Array and group metadata documents (.zarray and .zgroup) were both found in store "
        "{!r} at path {!r}. "
        "Only one of these files may be present in a given directory / prefix. "
        "Remove the .zarray file, or the .zgroup file, or both."
    )


class MetadataValidationError(BaseZarrError):
    """Raised when the Zarr metadata is invalid in some way"""

    _msg = "Invalid value for '{}'. Expected '{}'. Got '{}'."


class NodeTypeValidationError(MetadataValidationError):
    """
    Specialized exception when the node_type of the metadata document is incorrect..

    This can be raised when the value is invalid or unexpected given the context,
    for example an 'array' node when we expected a 'group'.
    """
