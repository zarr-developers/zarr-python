"""What every model compares and hashes by: a key its constructor computes once of everything the model shows."""


class Keyed:
    """A model whose `==` and `hash` compare a key computed once, when it is built, of what its document means.

    Each model computes its own key in `_key_of` and stores it; two models
    are equal when they are of one class and their keys are, so a model of
    another class, or anything else, is never equal to one.
    """

    __slots__ = ("_key",)

    _key: tuple[object, ...]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)
