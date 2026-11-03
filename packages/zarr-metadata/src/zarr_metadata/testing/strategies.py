"""Hypothesis strategies for valid Zarr v3 metadata declarations.

Every strategy here draws metadata its definition accepts, and that the
spec allows: a reader, writer or converter can test itself against the
whole space of a codec's, data type's, chunk grid's or chunk key
encoding's configurations instead of the handful its authors thought to
write down.

```python
from hypothesis import given
from zarr_metadata.testing.strategies import declarations
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC

@given(declarations(BLOSC_CODEC))
def test_my_reader_reads_every_blosc_codec(declaration):
    my_reader.read_codec(declaration)
```

What a configuration depends on, each kind's function takes:
`fill_values` a data type's configuration, `chunk_grid_configurations`
the shape of the array, `codec_configurations` the `Chunk` the codec is
handed. Left out, nothing is known of it, and what is drawn is what the
definition's rules accept. `declarations` wraps configurations in the
envelopes a document writes them in. `pipelines` draws a whole codec
pipeline for a chunk, each codec drawn for what the one before it hands
on.

How each definition's configurations are drawn is a value,
`Strategies`, keyed by the definition: `DEFAULT_STRATEGIES` holds one
for each definition this package defines. A configuration that holds
other declarations -- a struct's field types, the data type a
`cast_value` casts to, a shard's codecs -- draws them from the same
value, so they are drawn from every definition it holds, and read in a
scope of those definitions. An extension author adds their own with
`DEFAULT_STRATEGIES | {MY_CODEC: CodecStrategy(...)}`; one under a name
the defaults hold replaces theirs, as the later of two with one name.

Configurations are drawn at the data level the TypedDicts declare, JSON
arrays as tuples; `json.dumps` takes them as they are.

Requires the `testing` extra (`pip install zarr-metadata[testing]`).
"""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._definition import Definition, as_kind, canonicalize, kind_of
from zarr_metadata.testing._chunk_grid import CHUNK_GRID_STRATEGIES
from zarr_metadata.testing._chunk_key_encoding import CHUNK_KEY_ENCODING_STRATEGIES
from zarr_metadata.testing._codec import CODEC_STRATEGIES
from zarr_metadata.testing._data_type import DATA_TYPE_STRATEGIES
from zarr_metadata.testing._draw import (
    any_spelling,
    briefly,
    pipelines_for,
    scope_with,
    strategy_for,
)
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3 import ZarrV3DeclarationJSON
    from zarr_metadata.v3._context import ZarrV3Context
    from zarr_metadata.v3._definition import (
        Chunk,
        ChunkGridDefinition,
        ChunkKeyEncodingDefinition,
        CodecDefinition,
        DataTypeDefinition,
    )

C = TypeVar("C")

Spellings = Literal["all", "canonical"]
"""How declarations are written: every spelling the spec allows, or only the simplest.

`"all"` draws each envelope the spec gives a declaration -- a bare name, an
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
"""A strategy for each definition this package defines."""


def data_type_configurations(
    definition: DataTypeDefinition[C], *, strategies: Strategies = DEFAULT_STRATEGIES
) -> SearchStrategy[C]:
    """Configurations of the data type `definition` defines."""
    strategies = briefly(strategies)
    found = cast("DataTypeStrategy", strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations(strategies))


def fill_values(
    definition: DataTypeDefinition[C],
    configuration: C,
    *,
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[JSONValue]:
    """Fill values of the data type `definition` and `configuration` describe, in every spelling the spec gives them."""
    strategies = briefly(strategies)
    found = cast("DataTypeStrategy", strategy_for(definition, strategies))
    return found.fill_values(cast("Mapping[str, JSONValue]", configuration), strategies)


def chunk_grid_configurations(
    definition: ChunkGridDefinition[C],
    shape: tuple[int, ...] | None = None,
    *,
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[C]:
    """Configurations of the chunk grid `definition` defines, that fit an array of `shape`.

    With no shape, what is drawn is what the grid's rules accept.
    """
    strategies = briefly(strategies)
    found = cast("ChunkGridStrategy", strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations(shape, strategies))


def chunk_key_encoding_configurations(
    definition: ChunkKeyEncodingDefinition[C], *, strategies: Strategies = DEFAULT_STRATEGIES
) -> SearchStrategy[C]:
    """Configurations of the chunk key encoding `definition` defines."""
    strategies = briefly(strategies)
    found = cast("ChunkKeyEncodingStrategy", strategy_for(definition, strategies))
    return cast("SearchStrategy[C]", found.configurations(strategies))


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
    strategies = briefly(strategies)
    found = cast("CodecStrategy", strategy_for(definition, strategies))
    return cast("SearchStrategy[C] | None", found.configurations(chunk, strategies))


def pipelines(
    chunk: Chunk = NOTHING_KNOWN, *, strategies: Strategies = DEFAULT_STRATEGIES
) -> SearchStrategy[tuple[ZarrV3DeclarationJSON, ...]] | None:
    """Codec pipelines handed `chunk`, as an array's `codecs` holds them; None when none is drawn for it.

    Array -> array codecs, then one array -> bytes codec, then bytes ->
    bytes codecs, each drawn for the chunk the codec before it hands on,
    as its definition's `transition` says: a `transpose` hands on the
    chunk's axes permuted, a `cast_value` the data type it casts to. A
    codec that holds pipelines of its own, as a shard does, has each drawn
    the same way, handed the chunk its definition's `pipelines` says.
    None when no array -> bytes codec takes the chunk, as none takes a
    `string` one when the strategies hold no `vlen-utf8`.
    """
    strategies = briefly(strategies)
    return pipelines_for(chunk, strategies)


def _checked(
    declaration: ZarrV3DeclarationJSON,
    *,
    definition: Definition[Any],
    kind: type[Definition[Any]],
    scope: ZarrV3Context,
    spellings: Spellings,
) -> ZarrV3DeclarationJSON:
    """`declaration`, or its canonical spelling, once it reads without a problem in `scope`; a `ValueError` when it does not."""
    canonical, problems = canonicalize(declaration, kind, scope)
    if len(problems) != 0:
        msg = f"{declaration!r}, drawn for {definition.name!r}, which is refused: {problems!r}"
        raise ValueError(msg)
    if spellings == "canonical":
        return cast("ZarrV3DeclarationJSON", canonical)
    return declaration


def declarations(
    definition: Definition[C],
    configurations: SearchStrategy[C] | UNSET = UNSET,
    *,
    spellings: Spellings = "all",
    strategies: Strategies = DEFAULT_STRATEGIES,
) -> SearchStrategy[ZarrV3DeclarationJSON]:
    """Metadata declarations of `definition`: its name and a configuration, in an envelope a document writes.

    `configurations` is what to draw the configuration from, by default
    the definition's own with nothing known of what they depend on: pass
    `codec_configurations(definition, chunk)` for a codec handed a chunk,
    once it is known not to be None. `spellings` says whether every
    spelling the spec allows is drawn, or only the simplest.

    Each declaration is read in the strategies' scope, with `definition`
    in it whether or not they hold it, and with no chunk known; one its
    definition's rules refuse, as they refuse one drawn from
    `configurations` they do not accept, raises a `ValueError`. A
    codec's chunk rules are not checked: what fits the chunk is what
    `codec_configurations` draws.
    """
    strategies = briefly(strategies)
    drawn: Configurations | SearchStrategy[C] = (
        strategy_for(definition, strategies).unconstrained(strategies)
        if configurations is UNSET
        else configurations
    )
    checked = partial(
        _checked,
        definition=definition,
        kind=as_kind(kind_of(definition)),
        scope=scope_with(definition, strategies),
        spellings=spellings,
    )
    return any_spelling(definition, drawn).map(checked)


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
    "declarations",
    "fill_values",
    "pipelines",
]
