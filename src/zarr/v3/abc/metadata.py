from abc import ABC
from typing import TYPE_CHECKING, Dict, Any
from typing_extensions import Self
from dataclasses import asdict


class Metadata(ABC):
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this model to a dictionary
        """
        ...
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the model from a dictionary
        """
        ...

        return cls(**data)
