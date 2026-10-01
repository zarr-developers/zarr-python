"""The deprecations that warn from inside the code that handles them.

A deprecated function, class, or property carries a `@deprecated` decorator, whose
message is rendered in the API reference and on the Deprecations page. Everything
else, such as a deprecated import location or a deprecated way of passing a value,
is declared here once: the warning text, the migration path, and the planned removal.
The Deprecations page lists this table, and `tests/test_deprecations.py` fails if a
deprecation warning is raised any other way.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from zarr.errors import ZarrDeprecationWarning


@dataclass(frozen=True, kw_only=True)
class Deprecation:
    """One deprecation, as the Deprecations page lists it and the warning states it."""

    deprecated: str
    """What is deprecated, as Markdown, for the Deprecations page."""
    replacement: str
    """What to use instead, as Markdown, for the Deprecations page."""
    message: str
    """The warning message; a `str.format` template over the fields passed to `warn`."""
    removal: str = "A future release"
    """When the deprecated behavior is planned to go away."""
    category: type[Warning] = ZarrDeprecationWarning


DEPRECATIONS: dict[str, Deprecation] = {
    "data-type-validation-error-import": Deprecation(
        deprecated=(
            "Importing `DataTypeValidationError` from `zarr.dtype`, `zarr.core.dtype`, "
            "or `zarr.core.dtype.common`"
        ),
        replacement="`zarr.errors.DataTypeValidationError`",
        message=(
            "Importing DataTypeValidationError from {module} is deprecated. "
            "Use zarr.errors.DataTypeValidationError instead."
        ),
    ),
    "storage-default-compressor": Deprecation(
        deprecated="Assigning `zarr.storage.default_compressor`",
        replacement=(
            "`zarr.config.set({'array.v2_default_compressor.numeric': ...})` "
            "(see [Configuration](user-guide/config.md))"
        ),
        message=(
            "setting zarr.storage.default_compressor is deprecated, use "
            "zarr.config to configure array.v2_default_compressor "
            "e.g. config.set({{'codecs.zstd':'numcodecs.Zstd', "
            "'array.v2_default_compressor.numeric': 'zstd'}})"
        ),
    ),
    "v2-metadata-chunk-grid": Deprecation(
        deprecated="`ArrayV2Metadata.chunk_grid`",
        replacement="`ChunkGrid.from_metadata(metadata)`",
        message=(
            "ArrayV2Metadata.chunk_grid is deprecated. "
            "Use ChunkGrid.from_metadata(metadata) instead."
        ),
    ),
    "codec-enum-member": Deprecation(
        deprecated=(
            "Member access on `BloscShuffle`, `BloscCname`, `Endian`, and "
            "`ShardingCodecIndexLocation` (for example `BloscShuffle.shuffle`)"
        ),
        replacement='The equivalent literal string (`"shuffle"`)',
        message="{cls}.{name} is deprecated; pass the string {value!r} instead.",
    ),
    "codec-enum-parameter": Deprecation(
        deprecated=(
            "Passing an enum instance as a codec parameter "
            "(for example `BloscCodec(shuffle=SomeEnum.shuffle)`)"
        ),
        replacement="The equivalent literal string",
        message=(
            "Passing an enum to {codec}(..., {param}=...) is deprecated; "
            "pass the equivalent literal string instead."
        ),
    ),
}


def warn(key: str, *, stacklevel: int = 2, **fields: Any) -> None:
    """Emit the warning declared for `key`, formatting its message with `fields`.

    `stacklevel` is counted from the caller of this function, as it is for
    `warnings.warn`; the frame this function adds is accounted for here.
    """
    deprecation = DEPRECATIONS[key]
    warnings.warn(
        deprecation.message.format(**fields), deprecation.category, stacklevel=stacklevel + 1
    )
