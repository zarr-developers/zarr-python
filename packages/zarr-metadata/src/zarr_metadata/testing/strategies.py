"""Hypothesis strategies for valid Zarr v3 metadata fields.

Every strategy here draws metadata its definition accepts, and that the
spec allows: a reader, writer or converter can test itself against the
whole space of a codec's, data type's, chunk grid's or chunk key
encoding's configurations instead of the handful its authors thought to
write down.

```python
from hypothesis import given
from zarr_metadata.testing.strategies import fields
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC

@given(fields(BLOSC_CODEC))
def test_my_reader_reads_every_blosc_codec(field):
    my_reader.read_codec(field)
```

What a configuration depends on, each kind's function takes:
`fill_values` a data type's configuration, `chunk_grid_configurations`
the shape of the array, `codec_configurations` the `Chunk` the codec is
handed. Left out, nothing is known of it, and what is drawn is what the
definition's rules accept. `fields` wraps configurations in the
envelopes a document writes them in.

How each definition's configurations are drawn is a value,
`Strategies`, keyed by the definition: `DEFAULT_STRATEGIES` holds one
for each definition this package defines that holds no other field. An
extension author adds their own with
`DEFAULT_STRATEGIES | {MY_CODEC: CodecStrategy(...)}`.

Configurations are drawn at the data level the TypedDicts declare, JSON
arrays as tuples; `json.dumps` takes them as they are.

Requires the `testing` extra (`pip install zarr-metadata[testing]`).
"""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

from hypothesis import strategies as st

from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._definition import Definition, as_kind, canonicalize, kind_of
from zarr_metadata.testing._chunk_grid import CHUNK_GRID_STRATEGIES
from zarr_metadata.testing._chunk_key_encoding import CHUNK_KEY_ENCODING_STRATEGIES
from zarr_metadata.testing._codec import CODEC_STRATEGIES
from zarr_metadata.testing._data_type import DATA_TYPE_STRATEGIES
from zarr_metadata.testing._strategy import (
    NOTHING_KNOWN,
    ChunkGridStrategy,
    ChunkKeyEncodingStrategy,
    CodecStrategy,
    Configurations,
    DataTypeStrategy,
    DefinitionStrategy,
    Strategies,
)
from zarr_metadata.v3._context import ZarrV3Context
from zarr_metadata.v3._core import CORE_CONTEXT
from zarr_metadata.v3._definition import Chunk, DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3 import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._context import ZarrV3ContextSlots
    from zarr_metadata.v3._definition import (
        ChunkGridDefinition,
        ChunkKeyEncodingDefinition,
        CodecDefinition,
    )

C = TypeVar("C")

Spellings = Literal["all", "canonical"]
"""How fields are written: every spelling the spec allows, or only the simplest.

`"all"` draws each envelope the spec gives a field -- a bare name, an
object without a configuration, an empty configuration, a
`must_understand` of `true` -- and configurations that say what a
simpler one says, as a `blosc` `typesize` under `noshuffle` does.
`"canonical"` draws each in the spelling `canonicalize` gives it, which
is what a writer that writes the fewest words writes.
"""

DEFAULT_STRATEGIES: Final[MappingProxyType[Definition[Any], DefinitionStrategy]] = MappingProxyType(
    {
        **DATA_TYPE_STRATEGIES,
        **CHUNK_GRID_STRATEGIES,
        **CHUNK_KEY_ENCODING_STRATEGIES,
        **CODEC_STRATEGIES,
    }
)
"""A strategy for each definition this package defines that holds no other field."""


def _strategy_for(definition: Definition[Any], strategies: Strategies) -> DefinitionStrategy:
    """The strategy `strategies` holds for `definition`, which draws configurations of its kind."""
    found = strategies.get(definition)
    if found is None:
        msg = f"no strategy for {definition.name!r}; pass one in `strategies`"
        raise LookupError(msg)
    if not isinstance(definition, found.kind):
        msg = (
            f"the strategy for {definition.name!r}, a {type(definition).__name__}, draws "
            f"configurations of a {found.kind.__name__}: {found!r}"
        )
        raise TypeError(msg)
    return found


def data_type_configurations(
    definition: DataTypeDefinition[C], *, strategies: Strategies = DEFAULT_STRATEGIES
) -> SearchStrategy[C]:
    """Configurations of the data type `definition` defines."""
    found = cast("DataTypeStrategy", _strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations)


def fill_values(
    definition: DataTypeDefinition[C],
    configuration: C,
    *,
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[JSONValue]:
    """Fill values of the data type `definition` and `configuration` describe, in every spelling the spec gives them."""
    found = cast("DataTypeStrategy", _strategy_for(definition, strategies))
    return found.fill_values(cast("Mapping[str, JSONValue]", configuration))


def chunk_grid_configurations(
    definition: ChunkGridDefinition[C],
    shape: tuple[int, ...] | None = None,
    *,
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[C]:
    """Configurations of the chunk grid `definition` defines, that fit an array of `shape`.

    With no shape, what is drawn is what the grid's rules accept.
    """
    found = cast("ChunkGridStrategy", _strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations(shape))


def chunk_key_encoding_configurations(
    definition: ChunkKeyEncodingDefinition[C], *, strategies: Strategies = DEFAULT_STRATEGIES
) -> SearchStrategy[C]:
    """Configurations of the chunk key encoding `definition` defines."""
    found = cast("ChunkKeyEncodingStrategy", _strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations)


def codec_configurations(
    definition: CodecDefinition[C],
    chunk: Chunk = NOTHING_KNOWN,
    *,
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[C] | None:
    """Configurations of the codec `definition` defines, handed `chunk`; None when none takes it.

    With no chunk, nothing is known of it, and what is drawn is what the
    codec's rules accept.
    """
    found = cast("CodecStrategy", _strategy_for(definition, strategies))
    return cast("SearchStrategy[C] | None", found.configurations(chunk))


def _written_bare(definition: Definition[Any]) -> bool:
    """Whether the spec has the field written as its bare name: a data type it defines itself.

    "If the data type is defined in this specification, then the value
    must be the data type identifier provided as a string"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L509-L512).
    """
    if not isinstance(definition, DataTypeDefinition):
        return False
    return CORE_CONTEXT.claimant(DataTypeDefinition, definition.name) is definition


@st.composite
def _any_spelling(
    draw: st.DrawFn, definition: Definition[Any], configurations: SearchStrategy[Any]
) -> ZarrV3MetadataFieldJSON:
    """A configuration of `definition` in one of the envelopes the spec allows it.

    A configuration with nothing in it is written as the bare name, an
    object without one, or an object with an empty one; any object may
    say `must_understand` is `true`, which is what absence means. A name
    that carries the configuration, as `r16` does, and a data type the
    spec defines itself, are written as the bare name.
    """
    configuration = cast("Mapping[str, JSONValue]", draw(configurations))
    carried = definition._carrying_name(configuration)  # pyright: ignore[reportPrivateUsage]
    if carried is not None:
        return carried
    if _written_bare(definition):
        return definition.name
    empty = len(configuration) == 0
    if empty and draw(st.booleans()):
        return definition.name
    field: dict[str, JSONValue] = {"name": definition.name}
    if not empty or draw(st.booleans()):
        field["configuration"] = configuration
    if draw(st.booleans()):
        field["must_understand"] = True
    return cast("ZarrV3MetadataFieldJSON", field)


def _scope_of(definition: Definition[Any]) -> ZarrV3Context:
    """The scope that claims `definition` and nothing else; a `TypeError` for a definition no slot of a Zarr v3 scope holds."""
    for slot, kind in ZarrV3Context.kinds.items():
        if isinstance(definition, kind):
            return ZarrV3Context(**cast("ZarrV3ContextSlots", {slot: (definition,)}))
    msg = (
        f"{definition.name!r} is a {type(definition).__name__}, "
        "which no slot of a Zarr v3 scope holds"
    )
    raise TypeError(msg)


def _checked(
    field: ZarrV3MetadataFieldJSON,
    *,
    definition: Definition[Any],
    kind: type[Definition[Any]],
    scope: ZarrV3Context,
    spellings: Spellings,
) -> ZarrV3MetadataFieldJSON:
    """`field`, or its canonical spelling, once it reads without a problem in `scope`; a `ValueError` when it does not."""
    canonical, problems = canonicalize(field, kind, scope)
    if len(problems) != 0:
        msg = f"{field!r}, drawn for {definition.name!r}, which is refused: {problems!r}"
        raise ValueError(msg)
    if spellings == "canonical":
        return cast("ZarrV3MetadataFieldJSON", canonical)
    return field


def fields(
    definition: Definition[C],
    configurations: SearchStrategy[C] | UNSET = UNSET,
    *,
    spellings: Spellings = "all",
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[ZarrV3MetadataFieldJSON]:
    """Metadata fields of `definition`: its name and a configuration, in an envelope a document writes.

    `configurations` is what to draw the configuration from, by default
    the definition's own with nothing known of what they depend on: pass
    `codec_configurations(definition, chunk)` for a codec handed a chunk,
    once it is known not to be None. `spellings` says whether every
    spelling the spec allows is drawn, or only the simplest.

    Each field is read in a scope of its definition alone, with no chunk
    known, and one its definition's rules refuse, as they refuse one
    drawn from `configurations` they do not accept, raises a
    `ValueError`. A codec's chunk rules are not checked: what fits the
    chunk is what `codec_configurations` draws.
    """
    drawn: Configurations | SearchStrategy[C] = (
        _strategy_for(definition, strategies).unconstrained()
        if configurations is UNSET
        else configurations
    )
    checked = partial(
        _checked,
        definition=definition,
        kind=as_kind(kind_of(definition)),
        scope=_scope_of(definition),
        spellings=spellings,
    )
    return _any_spelling(definition, drawn).map(checked)


__all__ = [
    "DEFAULT_STRATEGIES",
    "NOTHING_KNOWN",
    "ChunkGridStrategy",
    "ChunkKeyEncodingStrategy",
    "CodecStrategy",
    "Configurations",
    "DataTypeStrategy",
    "DefinitionStrategy",
    "Spellings",
    "Strategies",
    "chunk_grid_configurations",
    "chunk_key_encoding_configurations",
    "codec_configurations",
    "data_type_configurations",
    "fields",
    "fill_values",
]
