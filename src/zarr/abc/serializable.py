from typing import Protocol


class JSONSerializable[T_co](Protocol):
    def to_json(self) -> T_co:
        """
        Serialize to a JSON-compatible Python object.
        """
        ...
