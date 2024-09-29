from typing import Any


class _BaseZarrError(ValueError):
    _msg = ""

    def __init__(self, *args: Any) -> None:
        super().__init__(self._msg.format(*args))


class ContainsGroupError(_BaseZarrError):
    _msg = "A group exists in store {0!r} at path {1!r}."


class ContainsArrayError(_BaseZarrError):
    _msg = "An array exists in store {0!r} at path {1!r}."


class ContainsArrayAndGroupError(_BaseZarrError):
    _msg = (
        "Array and group metadata documents (.zarray and .zgroup) were both found in store "
        "{0!r} at path {1!r}."
        "Only one of these files may be present in a given directory / prefix. "
        "Remove the .zarray file, or the .zgroup file, or both."
    )


__all__ = [
    "ContainsArrayAndGroupError",
    "ContainsArrayError",
    "ContainsGroupError",
]
