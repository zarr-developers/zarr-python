from typing_extensions import deprecated

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
    Base class for Zarr errors.
    """


class ContainsGroupError(BaseZarrError):
    """Raised when a group already exists at a certain path."""


class ContainsArrayError(BaseZarrError):
    """Raised when an array already exists at a certain path."""


class ArrayNotFoundError(BaseZarrError):
    """Raised when an array does not exist at a certain path."""


class GroupNotFoundError(BaseZarrError):
    """Raised when a group does not exist at a certain path."""


@deprecated("Use NodeNotFoundError instead.", category=None)
class PathNotFoundError(BaseZarrError):
    # Backwards compatibility with v2. Superseded by NodeNotFoundError.
    ...


class NodeNotFoundError(PathNotFoundError):
    """Raised when an array or group does not exist at a certain path."""


class ContainsArrayAndGroupError(BaseZarrError):
    """Raised when both array and group metadata are found at the same path. Zarr V2 only."""


class MetadataValidationError(BaseZarrError):
    """Raised when a Zarr metadata document is invalid"""


class NodeTypeValidationError(MetadataValidationError):
    """
    Specialized exception when the node_type of the metadata document is incorrect..

    This can be raised when the value is invalid or unexpected given the context,
    for example an 'array' node when we expected a 'group'.
    """


class ReadOnlyError(PermissionError, BaseZarrError):
    """
    Exception for when a mutation is attempted on an immutable resource.
    """
