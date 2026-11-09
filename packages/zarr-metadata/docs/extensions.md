# Writing an extension

A v3 document may name a codec, data type, chunk grid, chunk key encoding
or storage transformer that this package does not define. Each of these is
an
[extension point](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#extension-points)
of the v3 spec.

The readers leave such a declaration unclaimed. To check it, a program
adds a **definition** to the context it reads in.

The [`zarr_metadata.definition`](api/definition.md) module docstring is the
full reference. This page is the short version.

A definition is a value, not a class to subclass. It has:

- a `name`,
- a `TypedDict` for the configuration's JSON,
- a `rules` function, when the type cannot express every rule. The function
  yields the spec violations in a configuration.

A v3 codec definition also says:

- what kind of codec it is,
- whether its output size is fixed by its input size.

```python test="true" session="extensions"
from collections.abc import Iterator
from typing import Annotated, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata import ValidationProblem
from zarr_metadata.definition import Nested
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition, ZarrV3Context


class AcmeLz4Configuration(TypedDict, closed=True):
    acceleration: Annotated[int, Ge(1)]
    dictionary: NotRequired[str]
    dictionary_size: NotRequired[Annotated[int, Ge(1)]]


def acme_lz4_rules(
    configuration: AcmeLz4Configuration, nested: Nested
) -> Iterator[ValidationProblem]:
    if "dictionary" in configuration and "dictionary_size" not in configuration:
        yield ValidationProblem(
            ("dictionary_size",), "a dictionary needs its size", "missing_key"
        )


ACME_LZ4 = CodecDefinition(
    name="acme.lz4",
    configuration=AcmeLz4Configuration,
    kind="bytes_bytes",
    size="dynamic",
    rules=acme_lz4_rules,
)
CONTEXT = CORE_AND_EXTENSIONS_CONTEXT.union(ZarrV3Context(codecs=[ACME_LZ4]))
```

Pass `context=CONTEXT` to any reader or model constructor. A document that
names `acme.lz4` is then checked against this definition.

## Rules for the pieces

**The `TypedDict`**

- It must be a `typing_extensions.TypedDict`.
- It must say how undeclared keys are handled: `closed=True`,
  `extra_items=`, or `closed=False`. An open-by-default `TypedDict` is
  refused, because it would accept a misspelled key silently.

**Its members** must be JSON shapes:

- `int`, `float`, `bool`, `str`, `None` or `JSONValue`,
- a `Literal`,
- `tuple[T, ...]` or `tuple[T1, T2]`,
- a union,
- a `TypedDict` or `Mapping[str, V]`,
- a `NewType` or a type alias.

A number may carry bounds from annotated-types: `Gt`, `Ge`, `Lt`, `Le` or
`Interval`. Any other `Annotated` metadata is refused when the definition
is built.

**The rules function**

- It runs only on a configuration within its bounds, so it can rely on
  them.
- It reports the location of each problem. The value there becomes the
  problem's `input`.
- Show values in messages with `zarr_metadata.problems.shown`. Then they
  read like the package's own messages.

**Pickling.** Define each function at module top level. A model holds the
definitions that read it, so the model can be pickled only if they can.

## Other kinds of definition

Other kinds declare more:

- A data type declares its fill value (`fill_value`, `fill_value_rules`,
  `fill_value_canonical`). It also declares how its values are stored
  (`storage`).
- A chunk grid declares which shapes it fits (`shape_rules`) and its chunk
  lengths (`chunk_lengths`).
- A codec declares:
    - what it checks about the chunk it receives (`chunk_rules`),
    - what it hands on (`transition`),
    - any pipelines it contains (`pipelines`).

The [`zarr_metadata.v3.definition`](api/v3/definition.md) docstring
describes each of these.
