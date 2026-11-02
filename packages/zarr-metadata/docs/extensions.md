# Writing an extension

A document may name a codec, data type, chunk grid, chunk key encoding or
storage transformer that this package does not define: each is an
[extension point](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#extension-points) of the v3 spec. The readers leave
such a declaration unclaimed, and a program that needs to check it adds a
**definition** to the context it reads in. The
[`zarr_metadata.definition`](api/definition.md) module docstring is the
full reference; this page is the short version.

A definition is a value, not a class to subclass. It has a `name`, a
`TypedDict` for the configuration's JSON, and, when the type cannot
express every rule, a `rules` function that yields the spec violations in a
configuration. A v3 codec also says what kind of codec it is and whether
its output size is fixed by its input size.

```python test="true" session="extensions"
from collections.abc import Iterator
from typing import Annotated, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata import ValidationProblem
from zarr_metadata.definition import Nested
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition


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
CONTEXT = CORE_AND_EXTENSIONS_CONTEXT.update(codecs=[*CORE_AND_EXTENSIONS_CONTEXT.codecs.values(), ACME_LZ4])
```

Pass `context=CONTEXT` to any reader or model constructor, and a document
that names `acme.lz4` is checked against this definition.

Some rules for the pieces:

- The `TypedDict` must be a `typing_extensions.TypedDict`, and must say
  how undeclared keys are handled: `closed=True`, `extra_items=`, or
  `closed=False`. An open-by-default `TypedDict` is refused, because it
  would accept a misspelled key silently.
- Its members must be JSON shapes: `int`, `float`, `bool`, `str`, `None`,
  `JSONValue`, a `Literal`, `tuple[T, ...]`, `tuple[T1, T2]`, a union, a
  `TypedDict`, `Mapping[str, V]`, a `NewType` or a type alias. A number
  may carry bounds from annotated-types (`Gt`, `Ge`, `Lt`, `Le`,
  `Interval`). Any other `Annotated` metadata is refused when the
  definition is built.
- The rules run only on a configuration within its bounds, so a rule can
  rely on them. A rule reports the location of a problem, and the value
  there becomes the problem's `input`. Show values in messages with
  `zarr_metadata.problems.shown`, so they read like the package's own.
- Define each function at module top level. A model holds the definitions
  that read it, so it can be pickled only if they can.

A data type also declares its fill value (`fill_value`,
`fill_value_rules`, `fill_value_canonical`) and how its values are stored
(`storage`). A chunk grid declares which shapes it fits (`shape_rules`) and
its chunk lengths (`chunk_lengths`). A codec declares what it checks about
the chunk it receives (`chunk_rules`), what it hands on (`transition`), and
any pipelines it contains (`pipelines`). The
[`zarr_metadata.v3.definition`](api/v3/definition.md) docstring describes
each of these.
