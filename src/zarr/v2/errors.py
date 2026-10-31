class MetadataError(Exception):
    pass


class CopyError(RuntimeError):
    pass


class _BaseZarrError(ValueError):
    _msg = ""

    def __init__(self, *args):
        super().__init__(self._msg.format(*args))


class ArrayIndexError(IndexError):
    pass


class _BaseZarrIndexError(IndexError):
    _msg = ""

    def __init__(self, *args):
        super().__init__(self._msg.format(*args))


class ContainsGroupError(_BaseZarrError):
    _msg = "path {0!r} contains a group"


class ContainsArrayError(_BaseZarrError):
    _msg = "path {0!r} contains an array"


class ArrayNotFoundError(_BaseZarrError):
    _msg = "array not found at path %r' {0!r}"


class GroupNotFoundError(_BaseZarrError):
    _msg = "group not found at path {0!r}"


class PathNotFoundError(_BaseZarrError):
    _msg = "nothing found at path {0!r}"


class BadCompressorError(_BaseZarrError):
    _msg = "bad compressor; expected Codec object, found {0!r}"


class FSPathExistNotDir(GroupNotFoundError):
    _msg = "path exists but is not a directory: %r"


class ReadOnlyError(PermissionError):
    def __init__(self):
        super().__init__("object is read-only")


class BoundsCheckError(_BaseZarrIndexError):
    _msg = "index out of bounds for dimension with length {0}"


class NegativeStepError(IndexError):
    def __init__(self):
        super().__init__("only slices with step >= 1 are supported")


def err_too_many_indices(selection, shape):
    raise IndexError(f"too many indices for array; expected {len(shape)}, got {len(selection)}")


class VindexInvalidSelectionError(_BaseZarrIndexError):
    _msg = (
        "unsupported selection type for vectorized indexing; only "
        "coordinate selection (tuple of integer arrays) and mask selection "
        "(single Boolean array) are supported; got {0!r}"
    )
