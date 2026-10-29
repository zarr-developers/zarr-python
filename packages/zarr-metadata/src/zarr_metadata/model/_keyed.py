"""The key every model uses for comparison and hashing. The key covers everything the model shows, and is computed once, the first time the model is compared."""


class Keyed:
    """A model whose `==` and `hash` compare a cached key that describes what its document means.

    Each model computes its own key in `_key_of`. `_model_key` computes
    the key the first time something asks for it (`==`, `hash`, or a
    model that holds this one) and caches it. A model that is only read
    never pays for the key. Two models are equal when they have the same
    class and equal keys. A model of another class, or any other object,
    is never equal to a model.
    """

    __slots__ = ("_key",)

    _key: tuple[object, ...]

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for this model. Each model class defines its own."""
        raise NotImplementedError

    @property
    def _model_key(self) -> tuple[object, ...]:
        """The key. It is computed on first access and cached."""
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
