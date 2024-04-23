from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from typing import Dict
    from typing_extensions import Self

from dataclasses import fields, dataclass

from zarr.common import JSON


@dataclass(frozen=True)
class Metadata:
    def to_dict(self) -> JSON:
        """
        Recursively serialize this model to a dictionary.
        This method inspects the fields of self and calls `x.to_dict()` for any fields that
        are instances of `Metadata`. Sequences of `Metadata` are similarly recursed into, and
        the output of that recursion is collected in a list.
        """
        ...
        out_dict = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if isinstance(value, Metadata):
                out_dict[field.name] = getattr(self, field.name).to_dict()
            elif isinstance(value, str):
                out_dict[key] = value
            elif isinstance(value, Sequence):
                out_dict[key] = [v.to_dict() if isinstance(v, Metadata) else v for v in value]
            else:
                out_dict[key] = value

        return out_dict

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        """
        Create an instance of the model from a dictionary
        """
        ...

        return cls(**data)
