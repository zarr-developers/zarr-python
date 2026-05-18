"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3.array import ExtensionFieldV3


class GroupMetadataV3(TypedDict, extra_items=ExtensionFieldV3):  # type: ignore[call-arg]
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys are permitted if they conform to `ExtensionFieldV3`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, object]]


class GroupMetadataV3Partial(TypedDict, total=False, extra_items=ExtensionFieldV3):  # type: ignore[call-arg]
    """
    Partial form of `GroupMetadataV3`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `GroupMetadataV3` exactly.
    The only difference is `total=False`. See `ArrayMetadataV3Partial`
    for the rationale.

    The `NotRequired[...]` wrapper on `attributes` is kept intentionally even
    though `total=False` already makes every field optional: removing it would
    change `__annotations__`, breaking the byte-identical comparison with
    `GroupMetadataV3` in `tests/test_partial_equivalence.py`. PEP 655
    explicitly permits `NotRequired` inside a `total=False` `TypedDict`.

    Drift between this type and `GroupMetadataV3` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, object]]


__all__ = [
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
]
