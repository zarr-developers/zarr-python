from typing import TypedDict

from numcodecs.compat import NDArrayLike


class Context(TypedDict, total=False):
    """A context for component specific information

    All keys are optional. Any component reading the context must provide
    a default implementation in the case a key cannot be found.

    Items
    -----
    meta_array : array-like, optional
        An array-like instance to use for determining the preferred output
        array type.
    """

    meta_array: NDArrayLike
