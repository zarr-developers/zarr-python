"""Test errors"""

from zarr.errors import (
    ArrayNotFoundError,
    ContainsArrayAndGroupError,
    ContainsArrayError,
    ContainsGroupError,
    GroupNotFoundError,
    MetadataValidationError,
    NodeTypeValidationError,
)


def test_group_not_found_error() -> None:
    """
    Test that calling GroupNotFoundError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = GroupNotFoundError("store", "path")
    assert str(err) == "No group found in store 'store' at path 'path'"


def test_array_not_found_error() -> None:
    """
    Test that calling ArrayNotFoundError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = ArrayNotFoundError("store", "path")
    assert str(err) == "No array found in store 'store' at path 'path'"


def test_metadata_validation_error() -> None:
    """
    Test that calling MetadataValidationError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = MetadataValidationError("a", "b", "c")
    assert str(err) == "Invalid value for 'a'. Expected 'b'. Got 'c'."


def test_contains_group_error() -> None:
    """
    Test that calling ContainsGroupError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = ContainsGroupError("store", "path")
    assert str(err) == "A group exists in store 'store' at path 'path'."


def test_contains_array_error() -> None:
    """
    Test that calling ContainsArrayError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = ContainsArrayError("store", "path")
    assert str(err) == "An array exists in store 'store' at path 'path'."


def test_contains_array_and_group_error() -> None:
    """
    Test that calling ContainsArrayAndGroupError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = ContainsArrayAndGroupError("store", "path")
    assert str(err) == (
        "Array and group metadata documents (.zarray and .zgroup) were both found in store 'store' "
        "at path 'path'. Only one of these files may be present in a given directory / prefix. "
        "Remove the .zarray file, or the .zgroup file, or both."
    )


def test_node_type_validation_error() -> None:
    """
    Test that calling NodeTypeValidationError with multiple arguments returns a formatted string.
    This is deprecated behavior.
    """
    err = NodeTypeValidationError("a", "b", "c")
    assert str(err) == "Invalid value for 'a'. Expected 'b'. Got 'c'."
