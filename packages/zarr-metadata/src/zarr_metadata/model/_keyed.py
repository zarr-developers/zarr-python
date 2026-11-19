"""What every model derives from one normal form: its key, `==`, `hash`, `canonicalize` and `refines`.

A model's normal form is its document in its simplest equivalent
spelling, as its scope reads it. Each model writes that document in one
function, `_canonical_document`. Nothing else in a model decides what
its document means: the key is the text of the normal form with the
claims the scope made, `canonicalize` reads the normal form, and
`refines` reads the other model's document in this model's scope and
compares the two normal forms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata._json import MetadataValidationError, json_text

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.definition._scope import Claims


def normal_key(claims: Claims, document: Mapping[str, JSONValue]) -> tuple[object, ...]:
    """The key of a document in its simplest equivalent spelling, read with `claims`: the claims, and the document as JSON text.

    Two documents mean the same thing when a scope reads the names they
    use with the same definitions, and they are one document in their
    simplest spelling. The text sorts the members of each object, so the
    order a document writes them in is not compared.
    """
    return (frozenset(claims.items()), json_text(dict(document)))


class Keyed:
    """A model whose `==`, `hash`, `canonicalize` and `refines` are derived from one normal form.

    Each model writes its normal form in `_canonical_document`, and names
    the claims its scope made in `_claims`. The normal form and the key
    are computed the first time something asks for them, and cached. A
    model that is only read never pays for either. Two models are equal
    when they have the same class and equal keys. A model of another
    class, or any other object, is never equal to a model.
    """

    __slots__ = ("_key", "_normal")

    _key: tuple[object, ...]
    _normal: Mapping[str, JSONValue]

    def _canonical_document(self) -> Mapping[str, JSONValue]:
        """This model's document in its simplest equivalent spelling. Each model class writes its own."""
        raise NotImplementedError

    def _claims(self) -> Claims:
        """The claims the scope made for each name the document uses. Each model class returns its own."""
        raise NotImplementedError

    @property
    def _normal_form(self) -> Mapping[str, JSONValue]:
        """The normal form. It is computed on first access and cached."""
        try:
            return self._normal
        except AttributeError:
            normal = self._canonical_document()
            self._normal = normal
            return normal

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for this model, as `normal_key` computes it."""
        return normal_key(self._claims(), self._normal_form)

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

    def _refines(self, other: object, read_here: Callable[[], object]) -> bool:
        """Whether this model holds everything `other` holds: every claim `other` holds, and what `other`'s document reads as in this model's scope. `read_here` returns that reading, and raises `MetadataValidationError` when the document has a problem in this scope.

        A model refines another when its scope read every name the other
        scope read, with the same definition, and the other model's
        document, read in this model's scope, is this model. So one
        reader decides what both documents mean, with everything this
        scope knows: the chunk each codec receives, the data type that
        spells the fill value, and the definitions of the names the
        other scope left unclaimed. A document with a problem in this
        scope is not one this model refines.
        """
        if type(other) is not type(self):
            return False
        mine = self._claims()
        for key, definition in other._claims().items():
            if definition is not None and mine.get(key) != definition:
                return False
        try:
            again = read_here()
        except MetadataValidationError:
            return False
        return again == self
