# Visual Indexing Guide Design

**Date:** 2026-08-02

**Status:** Approved for implementation planning

**Package:** `packages/zarr-indexing`

## Purpose

The package exposes coordinate transforms, lazy view composition, chunk plans,
and paired chunk projections. These abstractions are useful to zarr-python and
to consumers such as napari that need lazy indexing without adopting a general
computation graph. The current documentation presents the formal model before
giving a NumPy user enough concrete experience to interpret it.

The new guide will teach the model visually and incrementally. It begins with
familiar NumPy indexing, reuses one small image throughout the narrative, and
introduces each formal term only after the reader has seen the problem that the
term names. Readers interested only in lazy array views can stop after the third
chapter. Library integrators can continue into chunk planning and projections.

## Goals

- Give NumPy users a correct mental model of an index as a coordinate mapping.
- Distinguish positions from coordinates: coordinate domains may have arbitrary
  origins, and a negative integer may name a real coordinate rather than wrap
  from the end.
- Explain why lazy views compose without reading array data.
- Show how a selection is partitioned over a chunk grid.
- Make the shared-domain relationship between `chunk_transform` and
  `cell_transform` visually unambiguous.
- Demonstrate the payoff for zarr chunk I/O and napari-style viewport reads.
- Make diagrams accessible and maintainable in light and dark themes.
- Keep displayed code synchronized with the public API through tests.

## Non-goals

- This is not a complete tutorial on NumPy indexing.
- This does not add a scheduler, cache, codec, or asynchronous I/O layer.
- The main narrative will not enumerate every indexing edge case. Those belong
  in an illustrated reference.
- The guide will not reproduce TensorStore prose or artwork. TensorStore remains
  conceptual prior art; all material will be independently authored.
- The guide will not require animation or JavaScript. Static numbered panels
  must preserve the explanation when printed or viewed with reduced motion.

## Audience and entry point

The primary reader understands expressions such as `array[1:5, 2]` but is
unfamiliar with index domains or chunk projection. This NumPy-first opening is
the least lossy common ground for array users and storage implementers.

The landing page exposes two routes:

1. **Use lazy indexing** begins the tour and permits stopping after composition.
2. **Integrate a chunked source** links to the chunk chapters and integration
   examples, while recommending the coordinate-mapping chapter when needed.

The API reference remains the durable lookup surface. The guide links to it but
does not duplicate complete signatures or parameter descriptions.

## Guided tour

The dense landing-page explanation will become five short chapters. Each page
answers one question, introduces at most one conceptual cluster, uses one
dominant figure, and ends with a runnable example plus previous/next links.

### 1. An index selects coordinates

Start with a 6 by 8 image and `image[1:5, 2]`. Show the selected source cells,
the one-dimensional result, its shape, and ordering. Introduce *request
coordinates* informally, without transform classes.

Reader outcome: an index says which source coordinate contributes to each
result coordinate.

### 2. Coordinates are addresses; a transform remembers the mapping

Begin with a number line whose domain is `[-2, 3)`. Its five valid coordinates
are `-2, -1, 0, 1, 2`: shape five does not imply origin zero, and `-1` names the
literal coordinate `-1`. It does not mean “the last element.” The visual must
show zero as an ordinary interior tick and negative coordinates with the same
styling as positive coordinates.

Explain why the algebra needs this model before showing its classes. Coordinates
are durable addresses, while positions are offsets within a particular view.
Arbitrary origins let translated views, cropped regions, chunk domains, and
external coordinate systems retain their addresses instead of repeatedly
renumbering every cell from zero.

Then contrast the two public dialects explicitly:

```python
domain = IndexDomain(inclusive_min=(-2,), exclusive_max=(3,))
domain.contains((-1,))  # True: -1 is a real coordinate
domain.narrow(-1)       # selects the literal interval [-1, 0)

LazyArray(values).lazy[-1].result()  # NumPy position: the final element
```

`LazyArray` is intentionally NumPy-like: each view is re-zeroed and negative
positions count from its end. The boundary converts that positional request into
literal coordinates before it enters the transform algebra. This distinction is
not an edge case; it is the reason the boundary layer exists.

Finally, redraw `image[1:5, 2]` as a mapping from result coordinate `i` to source
coordinate `(i + 1, 2)`. Introduce `IndexDomain`, `DimensionMap`, and
`ConstantMap` only where they name parts already visible in the figures.

Reader outcome: coordinates are literal addresses with arbitrary origins;
NumPy-style positions are a boundary dialect; and a transform is a reusable
mapping rather than materialized data.

### 3. Lazy views compose without reading

Apply a reverse and trim to the first view. Show the intermediate mapping and
collapse it into the equivalent direct mapping. Connect this to
`LazyArray.lazy[...]`, `shape`, `transform`, and `result()`.

This is the stopping point for ordinary lazy-array consumers. It states the
boundary explicitly: indexing is lazy; NumPy computation materializes.

Reader outcome: view chains are cheap and data is read only at materialization.

### 4. A selection crosses chunks

Reveal the image's existing 3 by 4 chunk grid. The array and selection do not
change; only the storage partition becomes visible. Identify the two touched
chunks, translate their selected coordinates to chunk-local coordinates, and
introduce `ChunkPlan`.

The figure labels the same touched chunk twice: first by its global
`chunk_domain`, then by the zero-origin coordinates used by `chunk_transform`.
This makes re-zeroing visible as an explicit translation between coordinate
systems, not an assumption that all coordinate indexing begins at zero.

Reader outcome: a chunk plan partitions a global transform without coupling it
to a source, codec pipeline, or scheduler.

### 5. A projection reconnects cells to the result

Complete the basic slice, then add exactly one contrast:

```python
lazy.lazy.oindex[[4, 1, 1], 2:6]
```

The reordered and duplicated rows show why a bounding box or chunk-local
selector cannot reconstruct the result. Introduce the synthetic cell domain
only now. Draw two arrows from that shared input domain:

- `cell_transform`: cell domain to request coordinates;
- `chunk_transform`: cell domain to chunk-local storage coordinates.

Arrows follow the actual transform direction. The prose states that both
transforms share a domain, matching TensorStore's effective paired-transform
contract while retaining this package's single-value ownership model.

Reader outcome: a projection says what to read inside one chunk and where those
cells belong in the request.

## Supporting pages

### Indexing patterns

An illustrated lookup page covers basic slices, integer axis removal, negative
strides, empty selections, boolean masks, orthogonal indexing, vectorized
indexing, broadcasting, and repeated or out-of-order coordinates. Each pattern
shows its expression, result shape, and mapping category. It begins with a
compact positional-versus-coordinate table, including `-1` in a zero-origin
NumPy view and `-1` in a domain that actually contains that coordinate.

### Integrations

Two small end-to-end examples show:

- a zarr-oriented consumer iterating a `ChunkPlan`, dispatching reads by
  `chunk_coords`, evaluating `chunk_transform`, and assembling through
  `cell_transform`;
- a napari-like viewport consumer composing a lazy spatial selection and
  materializing only that view, without requiring dask as its indexing layer.

These examples describe integration boundaries, not production schedulers.
Concurrency, caching, and async dispatch are explicit extension points.

The existing `ndsel.md` and `design-notes.md` remain advanced reference. Material
duplicated by the tour will be shortened and replaced with links so each concept
has one canonical explanation.

## Canonical example ladder

The guide uses one 6 by 8 image with 3 by 4 chunks. Coordinate labels dominate
the figures; values appear only when result ordering matters.

1. `image[1:5, 2]` establishes familiar selection behavior.
2. A short `[-2, 3)` number line distinguishes coordinate addresses from
   zero-based positions without changing the canonical image.
3. A second lazy index demonstrates composition.
4. Chunk boundaries are overlaid on the unchanged selection.
5. `oindex[[4, 1, 1], 2:6]` demonstrates reordering and duplication.
6. The same projections feed zarr chunk reads or a viewport consumer.

All other forms move to the patterns reference. This preserves continuity while
proving that paired projections do more than rectangular slicing.

## Visual grammar

Every figure uses the same semantic roles:

| Role | Meaning | Required redundant cue |
| --- | --- | --- |
| Request | Coordinates and shape of the result | “request” label and blue border |
| Source | Global coordinates of the wrapped array | “source” label and neutral grid |
| Selected | Cells participating in the request | heavy outline and coordinate label |
| Chunk-local | Coordinates inside one storage chunk | label and green border |
| Cell domain | Shared input domain of paired transforms | label and violet border |

Color is never the only carrier of meaning. Figures use explicit labels,
different borders, coordinate text, and captions. Arrows always point from a
transform's input domain to its output coordinates. Solid heavy lines mark
chunk boundaries. Numbered static panels express sequences.

Domain axes always display their actual inclusive and exclusive bounds. Zero is
never implied as an origin, and negative coordinate labels are not styled as
errors or as special wrapping syntax. When a figure uses positional NumPy
semantics instead, it carries an explicit “position” label.

Each figure has a sentence-length caption, SVG `<title>` and `<desc>`,
`role="img"`, an `aria-labelledby` relationship, adjacent prose, and a meaningful
reading order at narrow widths.

## Diagram system

Diagrams are deterministic generated SVG fragments, not raster files or Mermaid
diagrams. Mermaid cannot express selected coordinate cells and shared domains
with enough control. Raster files scale poorly and duplicate accessibility work.

The system consists of:

- small declarative Python structures for grids, cells, labels, arrows, panels,
  and semantic roles;
- pure functions that render a specification to SVG;
- checked-in generated SVG fragments under `docs/_static/diagrams/`;
- one stylesheet defining semantic tokens for Material's `default` and `slate`
  schemes;
- `pymdownx.snippets` inclusion so SVGs are inlined and respond to the active
  Material theme.

Inline SVG is required because Material stores its toggle state on the page's
`data-md-color-scheme` attribute. An SVG loaded through `<img>` cannot inherit
that state. Generated elements use semantic classes instead of hard-coded host
text colors. A fixed-color cell always specifies its matching foreground rather
than inheriting a potentially incompatible theme color.

The generator uses only the standard library. Documentation builds consume
checked-in assets without mutation. A generation command updates assets; check
mode renders in memory and fails when committed output is stale.

## Executable examples

Runnable files beside the documentation contain named snippet regions.
`pymdownx.snippets` inserts those exact regions into fenced code blocks. Tests
execute the same files, preventing the displayed calls from drifting away from
the checked behavior.

Examples avoid nondeterministic representations, filesystem I/O, network access,
and large allocations. Integration examples use tiny recording sources so tests
can prove that untouched chunks are not read.

One executable example directly compares the dialects. It proves that `-1` is
contained by `IndexDomain((-2,), (3,))` and selects the literal coordinate in
the transform algebra, while `LazyArray(...).lazy[-1]` selects the final
position after NumPy-style normalization.

## File layout

```text
packages/zarr-indexing/
├── docs/
│   ├── index.md
│   ├── guide/
│   │   ├── index.md
│   │   ├── 01-indexing.md
│   │   ├── 02-transforms.md
│   │   ├── 03-composition.md
│   │   ├── 04-chunks.md
│   │   ├── 05-projections.md
│   │   ├── patterns.md
│   │   └── integrations.md
│   ├── examples/
│   │   ├── coordinate_origins.py
│   │   ├── canonical_slice.py
│   │   ├── lazy_composition.py
│   │   ├── chunk_projection.py
│   │   └── integrations.py
│   ├── diagrams/
│   │   ├── specifications.py
│   │   └── render.py
│   └── _static/
│       ├── indexing-guide.css
│       └── diagrams/*.svg
└── tests/
    ├── test_doc_diagrams.py
    └── test_doc_examples.py
```

`mkdocs.yml` adds the guided-tour navigation, snippets extension, and guide
stylesheet. It excludes the Python specification and example source directories
from the published site after the snippets extension has read them; generated
SVGs remain publishable. Existing API-reference navigation remains intact.

## Validation and failure behavior

Diagram rendering raises an error naming the figure and element when a
specification has an invalid grid size, an out-of-domain coordinate, a duplicate
identifier, missing accessible metadata, an unknown semantic role, or an arrow
whose source or target cannot be resolved.

Tests use one representative matrix of valid grid, mapping, and panel
combinations, plus one test per validation error. They also assert:

- checked-in SVGs are byte-for-byte current;
- generated IDs are unique and deterministic;
- every figure has `title`, `desc`, `role`, and `aria-labelledby`;
- every text element has an explicit semantic text role;
- both Material schemes define every semantic palette token;
- foreground/background pairs meet WCAG AA contrast for normal text;
- every semantic role declares a non-color cue.

Example tests execute every documentation example against the public API. The
integration fixture verifies both results and the exact touched chunks. The
advanced example verifies order and duplicate preservation, the behavioral
reason for documenting `cell_transform` separately.

A dedicated coordinate-origin test checks the paired examples above and fails
if the literal-coordinate and positional dialects are accidentally conflated.
Diagram tests also require the negative-origin number line to expose all five
tick labels and identify `[-2, 3)` as its domain.

Acceptance requires the package test suite, Ruff, Pyright, deterministic diagram
check, and `mkdocs build --strict` to pass. Manual visual QA covers Material
light and dark schemes at desktop and narrow widths, plus keyboard navigation
and screen-reader labels.

The package's lint and type-check commands explicitly include the diagram
generator, executable examples, and their tests. The existing Pyright `include`
setting covers only `src`, so implementation must either extend that setting or
pass the new paths directly; a passing source-only check is not sufficient.

The implementation will not add a large browser dependency solely for screenshot
snapshots. Deterministic SVG, palette, accessibility, and strict-build tests form
the automated contract; responsive composition receives focused manual QA.

## Acceptance criteria

1. A reader can follow the five chapters without first reading the API reference.
2. The first three chapters form a self-contained lazy-indexing introduction.
3. A reader can explain why an `IndexDomain` need not start at zero, why `-1`
   can be a valid literal coordinate, and why `LazyArray(...).lazy[-1]` still
   means the final position.
4. The last two chapters correctly explain chunk planning and transform
   directionality.
5. Zarr and viewport examples read only chunks touched by their selections.
6. Every displayed code fragment is backed by an executed example.
7. Every diagram is deterministic, accessible, and legible in both themes and
   at narrow widths.
8. The patterns page covers important variants without interrupting the tour.
9. Existing notes and API pages link into the guide without competing accounts.

## Approved decisions

- NumPy-first entry point.
- Five-page guided tour.
- Accessible generated SVG system.
- One canonical basic example plus one advanced contrast.
- Explicit user/integrator stopping point after lazy composition.
- Explicit distinction between literal coordinates and NumPy-style positions,
  including negative coordinate domains.
- No animation, JavaScript dependency, or color-only semantics.
- Light and dark contrast are tested requirements, not deferred visual polish.
