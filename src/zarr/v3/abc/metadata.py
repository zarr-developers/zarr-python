from abc import ABC
from typing import Dict
from typing_extensions import Self
from dataclasses import asdict

from zarr.v3.common import JSON


class Metadata(ABC):
    def to_dict(self) -> Dict[str, JSON]:
        """
        Serialize this model to a dictionary
        """
        ...
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        """
        Create an instance of the model from a dictionary
        """
        ...

        return cls(**data)
