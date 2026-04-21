"""
CRC32C codec.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the envelope's
``configuration`` key is absent.
"""

from collections.abc import Mapping
from typing import Literal

from zarr_metadata.common import NamedConfig

Crc32cCodecName = Literal["crc32c"]
"""The ``name`` field value of a ``crc32c`` codec envelope."""


Crc32cCodec = NamedConfig[Crc32cCodecName, Mapping[str, object]]
"""``crc32c`` codec named-config envelope.

Per spec, the CRC32C codec has no configuration fields, so the
``configuration`` key is optional and, if present, should be an empty
mapping.
"""


__all__ = [
    "Crc32cCodec",
    "Crc32cCodecName",
]
