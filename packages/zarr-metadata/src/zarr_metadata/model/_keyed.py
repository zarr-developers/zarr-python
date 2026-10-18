"""What every model compares and hashes by: a key computed once, the first time the model is compared, of everything the model shows."""


class Keyed:
    """A model whose `==` and `hash` compare a key computed once of what its document means.

    Each model computes its own key in `_key_of`; `_model_key` computes it
    the first time it is asked for -- by `==`, `hash`, or a model holding
    this one -- and keeps it, so a model that is only read pays nothing
    for it. Two models are equal when they are of one class and their keys
    are, so a model of another class, or anything else, is never equal to
    one.
    """

    __slots__ = ("_key",)

    _key: tuple[object, ...]

    def _key_of(self) -> tuple[object, ...]:
        """What `==` and `hash` compare of this model: each model's own."""
        raise NotImplementedError

    @property
    def _model_key(self) -> tuple[object, ...]:
        """The key, computed the first time it is asked for and kept."""
        try:
            return self._key
        except AttributeError:
            key = self._key_of()
            self._key = key
            return key

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._model_key == other._model_key

    def __hash__(self) -> int:
        return hash(self._model_key)
