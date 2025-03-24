from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.common import JSON

from dataclasses import dataclass, fields

import numpy as np

__all__ = ["Metadata"]


@dataclass(frozen=True)
class Metadata:
    def to_dict(self) -> dict[str, JSON]:
        """
        Recursively serialize this model to a dictionary.
        This method inspects the fields of self and calls `x.to_dict()` for any fields that
        are instances of `Metadata`. Sequences of `Metadata` are similarly recursed into, and
        the output of that recursion is collected in a list.
        """
        out_dict = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if isinstance(value, Metadata):
                out_dict[field.name] = getattr(self, field.name).to_dict()
            elif isinstance(value, str):
                out_dict[key] = value
            elif isinstance(value, Sequence):
                out_dict[key] = tuple(v.to_dict() if isinstance(v, Metadata) else v for v in value)
            else:
                out_dict[key] = value

        return out_dict

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """
        Create an instance of the model from a dictionary
        """

        return cls(**data)
    
    def __eq__(self, other: Any) -> bool:
        """Checks metadata are identical, including special treatment for NaN fill_values."""
        if not isinstance(other, type(self)):
            return False
        
        metadata_dict1 = self.to_dict()
        metadata_dict2 = other.to_dict()

        # fill_value is a special case because numpy NaNs cannot be compared using __eq__, see https://stackoverflow.com/a/10059796
        fill_value1 = metadata_dict1.pop("fill_value")
        fill_value2 = metadata_dict2.pop("fill_value")
        if np.isnan(fill_value1) and np.isnan(fill_value2):
            fill_values_equal = fill_value1.dtype == fill_value2.dtype
        else:
            fill_values_equal = fill_value1 == fill_value2

        # everything else in ArrayV3Metadata is a string, Enum, or Dataclass
        return fill_values_equal and metadata_dict1 == metadata_dict2

