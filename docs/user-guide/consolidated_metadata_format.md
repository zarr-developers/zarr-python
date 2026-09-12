# Consolidated metadata format

This page is a specification-style description of the consolidated metadata
that Zarr-Python reads and writes. It is intended for people implementing
consolidated metadata in another library or language who want to interoperate
with Zarr-Python, and for people who need to know exactly what Zarr-Python
puts on disk. For an introduction to *using* consolidated metadata from
Python, see [Consolidated metadata](consolidated_metadata.md).

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this
document are to be interpreted as described in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Motivation

Text in this section is quoted from
[zarr-specs#309](https://github.com/zarr-developers/zarr-specs/pull/309)
(Tom Augspurger, 2024), licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/):

> Consolidated metadata can help reduce the time needed to load the metadata
> for an entire hierarchy, especially when the metadata is being served over a
> network. Without consolidated metadata, opening an entire hierarchy over the
> network requires an HTTP request per node. Consolidated metadata enables
> loading the metadata for every node in a hierarchy with a single HTTP
> request.

and, from the proposal's description:

> This PR adds a new optional field to the core metadata for consolidating all
> the metadata of all child nodes under a single object. The motivation is
> similar to consolidated metadata in Zarr V2: without consolidated metadata,
> the time to load metadata for an entire hierarchy scales linearly with the
> number of nodes. This can be costly, especially for large hierarchies served
> HTTP from a remote storage system (like Blob Storage).

## Concepts

A hierarchy is **consolidated at** a group (the *consolidating group*). The
consolidating group's own metadata document gains a copy of the metadata of
every node below it, at every depth. Nothing about the child nodes' own
metadata documents changes.

Two representations of the same information appear in this document:

Flat
:   A single mapping from *path* to *metadata document*, where path is the
    full path of a node relative to the consolidating group. This is the
    persisted (on-disk) form for both Zarr formats.

Nested
:   Each group holds only its *immediate* children, and child groups hold
    their own children recursively. This is Zarr-Python's in-memory form
    (`GroupMetadata.consolidated_metadata.metadata`). It is never written to a
    store and is mentioned here only because it leaks into the on-disk form in
    one place (the [empty child marker](#child-groups-carry-an-empty-marker)).

Converting between the two is lossless. `ConsolidatedMetadata.flattened_metadata`
goes nested → flat; `ConsolidatedMetadata._flat_to_nested` goes flat → nested.

## Paths

A **path** is the name of a node relative to the consolidating group:

* Segments are joined with `/`.
* There is no leading or trailing `/`.
* The consolidating group itself is **not** included (it would have the empty
  path).

Given the hierarchy below, where capital letters are groups and lowercase
letters are arrays:

```text
A/
  x
  B/
    y
    C/
```

consolidating at `A` produces the paths `B`, `B/C`, `B/y`, `x`; consolidating
at `B` produces `C`, `y`; consolidating at `C` produces no paths (an empty
mapping, which is still written).

!!! note "Difference from the zarr-specs#309 text"
    The worked example in zarr-specs#309 reads:

    > If we consolidate the metadata at the Group ``A``, the consolidated
    > metadata would have the keys ``"A", "A/B", "A/B/C", "A/B/C/x", ...``.
    >
    > If we consolidate the metadata at the Group ``B``, the consolidated
    > metadata would have the keys ``"C", "C/x", "C/y"``.

    The second sentence matches Zarr-Python; the first does not, since it
    includes the consolidating group's own name as a prefix. Zarr-Python
    always uses paths relative to the consolidating group and never includes
    the consolidating group itself. This is the behaviour the rest of the
    proposal describes ("the path of the node relative to the node at which
    the metadata is being consolidated"), so we read the first sentence as a
    typo.

## Zarr format 3

### Location

Consolidated metadata is stored **inline** in the consolidating group's
`zarr.json`, under a top-level key named `consolidated_metadata`. No other
file is written.

### Schema

From zarr-specs#309 (Tom Augspurger, CC-BY-4.0):

> `consolidated_metadata`
>
> An object consolidating all the Array and Group metadata of members below
> the root node in a hierarchy.
>
> | Field             | Type                      | Description |
> |-------------------|---------------------------|-------------|
> | `metadata`        | `Map<string, Metadata>`   | A mapping from node path to Group or Array `Metadata` object. |
> | `kind`            | const `'inline'`          | The string literal `'inline'`. Reserved for future use. |
> | `must_understand` | const `False`             | The boolean literal `False`. Indicates that the field is not required to load the Zarr hierarchy. |
>
> Note that *all* children Arrays and Groups should be included in
> consolidated metadata, not just the nodes immediately below the root Group.
> Children nested inside other groups should be included too as a flat list
> of nodes. The keys of `metadata` should be the path of the node relative to
> the node at which the metadata is being consolidated (i.e. the `Group`
> where this `consolidated_metadata` object is stored).
>
> Consolidated Metadata is optional. If present, then readers should use the
> consolidated metadata. When not present, readers should use the
> non-consolidated metadata located in the Store to load the data.
>
> The `kind` field indicates that consolidated metadata is stored inline in
> the root `zarr.json` object. At this time, `'inline'` is the only supported
> value for `kind`. Future versions of the specification may allow for
> consolidated metadata in other locations.

Zarr-Python's implementation of this schema, stated as requirements:

| Field             | Writer                                  | Reader |
|-------------------|-----------------------------------------|--------|
| `kind`            | MUST write `"inline"`.                  | MUST reject any value other than `"inline"` (`ValueError`). |
| `must_understand` | MUST write `false`.                     | Not checked. |
| `metadata`        | MUST write a JSON object (possibly empty). | MUST reject a non-object (`TypeError`). Each value MUST be a JSON object (`TypeError` otherwise). |

### Values of `metadata`

Each value is a complete node metadata document, i.e. exactly what would be
found in that node's own `zarr.json`, with the following rules:

* The document MUST contain `zarr_format`. Readers discriminate on this first;
  Zarr-Python accepts `2` or `3` **per entry**, so a consolidated Zarr format 3
  group MAY in principle contain Zarr format 2 children. Writers SHOULD NOT
  rely on this.
* For `zarr_format: 3`, the document MUST contain `node_type`, one of `"group"`
  or `"array"`, and is parsed as `GroupMetadata` or `ArrayV3Metadata`
  respectively.
* For `zarr_format: 2`, the presence of a `shape` key marks an array;
  otherwise the entry is a group.
* Array documents are written by `ArrayV3Metadata.to_dict()` and therefore
  contain every key Zarr-Python normally writes (including
  `storage_transformers: []`, and `attributes: {}` when empty).

### Child groups carry an empty marker

Every group entry in `metadata` MUST itself carry a `consolidated_metadata`
key whose value is the empty marker:

```json
{"kind": "inline", "must_understand": false, "metadata": {}}
```

This is the one place the nested in-memory form shows through. The marker
does **not** mean the child group has no children: the child's descendants are
still listed in the flat mapping at the consolidating group. It exists so
that, after a reader nests the flat mapping, a group with no children is
distinguishable from a group whose children are unknown (`consolidated_metadata`
absent / `null`). Zarr-Python's writer inserts the marker for every child
group ([`consolidate_metadata`][zarr.api.asynchronous.consolidate_metadata]
and `ConsolidatedMetadata.flattened_metadata`).

Readers SHOULD tolerate a child group entry that omits the marker; Zarr-Python
does, and normalises such entries to the empty marker when nesting.

### Key order

`metadata` is a JSON object and therefore unordered in principle. Since
Zarr-Python 3.1.1 the writer sorts keys as follows:

1. Primarily by **depth**, ascending, where depth is the number of `/`
   characters in the path.
2. Secondarily by the path string after Unicode NFKC normalisation and
   case-folding (`unicodedata.normalize("NFKC", key).casefold()`), ascending.

Paths whose normalised, case-folded forms are equal retain their input order.
This ordering alone therefore does not guarantee byte-identical output for
every hierarchy.

Readers MUST NOT depend on key order. In particular, keys sharing a parent are
not guaranteed to be adjacent, and Zarr-Python's reader groups by parent
rather than assuming adjacency.

### Root-level fields

Zarr-Python writes `consolidated_metadata` at the top level of the group
document alongside `zarr_format`, `node_type`, and `attributes`, not inside
`attributes`. When a group has no consolidated metadata, the key is omitted
entirely (never written as `null`).

### Complete example

Consolidating the hierarchy from [Paths](#paths) (with `x` an `int32` array of
shape `(10,)` chunked by `(5,)`, `y` a `float64` array of shape `(2, 2)`, and
default codecs) yields the following `zarr.json` at `A`. This output was
produced by Zarr-Python and is reformatted only for indentation.

```json
{
  "attributes": {"title": "example"},
  "zarr_format": 3,
  "consolidated_metadata": {
    "kind": "inline",
    "must_understand": false,
    "metadata": {
      "B": {
        "attributes": {"kind": "child"},
        "zarr_format": 3,
        "consolidated_metadata": {
          "kind": "inline",
          "must_understand": false,
          "metadata": {}
        },
        "node_type": "group"
      },
      "x": {
        "shape": [10],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
          {"name": "bytes", "configuration": {"endian": "little"}},
          {"name": "zstd", "configuration": {"level": 0, "checksum": false}}
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": []
      },
      "B/C": {
        "attributes": {},
        "zarr_format": 3,
        "consolidated_metadata": {
          "kind": "inline",
          "must_understand": false,
          "metadata": {}
        },
        "node_type": "group"
      },
      "B/y": {
        "shape": [2, 2],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0.0,
        "codecs": [
          {"name": "bytes", "configuration": {"endian": "little"}},
          {"name": "zstd", "configuration": {"level": 0, "checksum": false}}
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": []
      }
    }
  },
  "node_type": "group"
}
```

Observe the key order: depth 0 (`B`, `x`), then depth 1 (`B/C`, `B/y`), with
case-folded lexical order within each depth.

## Zarr format 2

### Location

Consolidated metadata is stored in a separate document at the key
`.zmetadata` in the consolidating group's directory, alongside `.zgroup` and
`.zattrs`. A reader MAY be configured to look for a different key
([`zarr.open_group`][] accepts a string for `use_consolidated`); Zarr-Python
only ever writes `.zmetadata`.

### Schema

```json
{
  "zarr_consolidated_format": 1,
  "metadata": { "<path>/.zarray": {...}, "<path>/.zattrs": {...}, "<path>/.zgroup": {...}, ... }
}
```

| Field                      | Writer                   | Reader |
|----------------------------|--------------------------|--------|
| `zarr_consolidated_format` | MUST write `1`.          | Not checked. |
| `metadata`                 | MUST write an object.    | MUST be an object; each key MUST end in `/.zarray`, `/.zattrs`, or `/.zgroup` (`ValueError` otherwise), except for the bare root keys below. |

### Keys of `metadata`

Unlike Zarr format 3, each *node* contributes up to **two** entries, one per
underlying metadata document, keyed by `<path>/<document name>`:

* Arrays: `<path>/.zarray` and `<path>/.zattrs`.
* Groups: `<path>/.zgroup` and `<path>/.zattrs`.

Zarr-Python always writes the `.zattrs` entry, as `{}` when there are no
attributes.

The consolidating group itself **is** included, under the bare keys `.zgroup`
and `.zattrs` (no path prefix). This is a difference from Zarr format 3 and
is inherited from Zarr-Python 2.x. On read, Zarr-Python **ignores** these two
root entries and uses the real `.zgroup` and `.zattrs` documents, which it
fetches in the same request batch.

### Values of `metadata`

`.zarray` and `.zgroup` values are the documents that would be found at the
corresponding store key, with one deviation:

!!! warning "Deviation from Zarr-Python 2.x"
    Zarr-Python 3 writes every child `.zgroup` entry as

    ```json
    {
      "zarr_format": 2,
      "consolidated_metadata": {"metadata": {}, "must_understand": false, "kind": "inline"}
    }
    ```

    rather than the `{"zarr_format": 2}` that Zarr-Python 2.x wrote. This is
    the Zarr format 3 [empty child marker](#child-groups-carry-an-empty-marker)
    leaking into the Zarr format 2 encoding. Zarr-Python's reader ignores
    unknown keys in `.zgroup` entries, and Zarr-Python 2.x likewise ignored
    them, but other readers that validate `.zgroup` strictly may reject this.
    Readers SHOULD ignore the key; writers targeting maximum compatibility
    with older Zarr format 2 tooling may wish to omit it.

### Key order

Entries are emitted in the same order as the Zarr format 3 case (depth, then
case-folded path), with each node's `.zattrs` entry immediately preceding its
`.zarray`/`.zgroup` entry. Readers MUST NOT depend on this order.

### Serialisation

`.zmetadata` is written **compactly** (no indentation), regardless of the
`json_indent` configuration setting that applies to other metadata documents.

### Complete example

The same hierarchy as above, written as Zarr format 2 with default
compressor. Output produced by Zarr-Python, reformatted for indentation.

```json
{
  "metadata": {
    ".zgroup": {"zarr_format": 2},
    ".zattrs": {"title": "example"},
    "B/.zattrs": {"kind": "child"},
    "B/.zgroup": {
      "zarr_format": 2,
      "consolidated_metadata": {"metadata": {}, "must_understand": false, "kind": "inline"}
    },
    "x/.zattrs": {},
    "x/.zarray": {
      "shape": [10],
      "chunks": [5],
      "dtype": "<i4",
      "fill_value": 0,
      "order": "C",
      "filters": null,
      "dimension_separator": ".",
      "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
      "zarr_format": 2
    },
    "B/C/.zattrs": {},
    "B/C/.zgroup": {
      "zarr_format": 2,
      "consolidated_metadata": {"metadata": {}, "must_understand": false, "kind": "inline"}
    },
    "B/y/.zattrs": {},
    "B/y/.zarray": {
      "shape": [2, 2],
      "chunks": [2, 2],
      "dtype": "<f8",
      "fill_value": 0.0,
      "order": "C",
      "filters": null,
      "dimension_separator": ".",
      "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
      "zarr_format": 2
    }
  },
  "zarr_consolidated_format": 1
}
```

## Writer procedure

This is what [`zarr.consolidate_metadata`][] does, stated so that another
implementation can reproduce it:

1. Open the consolidating group **without** using any existing consolidated
   metadata (`use_consolidated=False`), so that stale data is never copied
   forward.
2. If the store reports `supports_consolidated_metadata == False`, raise and
   do nothing.
3. List every descendant node at every depth by reading the store directly,
   and collect `(path, metadata)` pairs.
4. For every group in the collection, set its `consolidated_metadata` to the
   empty marker.
5. Attach the flat mapping to the consolidating group's metadata and write the
   group's metadata document(s): `zarr.json` for Zarr format 3;
   `.zgroup`, `.zattrs`, **and** `.zmetadata` for Zarr format 2.
6. For Zarr format 3, emit a `ZarrUserWarning` noting that the feature is not
   part of the specification.

Writers re-write the whole document; there is no incremental update.

## Reader procedure

This is what `AsyncGroup.open` does with the `use_consolidated` argument.

| `use_consolidated` | Consolidated metadata present | Absent |
|--------------------|-------------------------------|--------|
| `None` (default)   | Use it.                       | Read children from the store. |
| `True`             | Use it.                       | Raise `ValueError`. |
| `False`            | Ignore it.                    | Read children from the store. |
| Non-empty `str` (format 2 only) | Use the document at that key instead of `.zmetadata`. | Raise `ValueError`. |

If the store reports `supports_consolidated_metadata == False`, `None` is
treated as `False`, and `True` raises.

For Zarr format 3, presence is determined by the `consolidated_metadata` key
in `zarr.json`. For Zarr format 2, presence is determined by whether the
`.zmetadata` document exists; Zarr-Python requests it concurrently with
`.zgroup` and `.zattrs` so that the consolidated and unconsolidated open paths
cost the same number of round trips.

Once loaded, every child lookup (`group["x"]`, `group.members()`,
`group.tree()`, …) is served from the consolidated copy without touching the
store. A child group obtained this way carries the nested slice of the
consolidated metadata that applies to *its* descendants, so lookups remain
store-free at every depth. Readers that need a fresh view of a changing
hierarchy MUST reopen with `use_consolidated=False`.

### Consistency

From zarr-specs#309 (Tom Augspurger):

> Like Zarr v2, consolidated metadata introduces the possibility of
> "inconsistent" metadata between the consolidated and non-consolidated
> forms. Should the spec take any stance on how to handle this? I've currently
> worded things to say that readers should always use the consolidated
> metadata if it's present.

Zarr-Python follows that wording: when consolidated metadata is present and
not explicitly disabled, it is authoritative for reads, and no attempt is made
to detect drift from the per-node documents. Writers are responsible for
re-consolidating after modifying a hierarchy. Zarr-Python does **not**
re-consolidate automatically when nodes are created or have their attributes
updated. The one exception is deletion: deleting a member of a group that was
opened with consolidated metadata (`del group[name]`) removes that member's
entry from the consolidated copy and re-writes the group's metadata document.
See
[Synchronization and concurrency](consolidated_metadata.md#synchronization-and-concurrency).

## Known limitations and open questions

These are noted in zarr-specs#309 and remain open:

* **Root document size.** Quoting the proposal: "For *very* large
  hierarchies, this will bloat the size of the root `zarr.json`, slowing down
  operations that just want to open the metadata for the root." The `kind`
  field is reserved so that a future value could point at an external
  document instead.
* **Overlap with child listing.** The proposal notes overlap with
  [zarr-specs#284](https://github.com/zarr-developers/zarr-specs/issues/284),
  which would store child *paths* only, allowing a reader to list the
  hierarchy in one request and then fetch node metadata concurrently.
* **`must_understand`.** The field is always `false` and Zarr-Python's reader
  does not inspect it. It exists so that a conforming Zarr format 3 reader
  that does not implement consolidated metadata can safely ignore the key
  per the core specification's rules for unknown metadata fields.

## Attribution

Quoted passages in this document are from
[zarr-developers/zarr-specs#309, "Added consolidated metadata to spec"](https://github.com/zarr-developers/zarr-specs/pull/309)
by Tom Augspurger, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The Zarr format 3
implementation in Zarr-Python was contributed in
[zarr-python#2113](https://github.com/zarr-developers/zarr-python/pull/2113),
also by Tom Augspurger. The Zarr format 2 `.zmetadata` layout originates in
Zarr-Python 2.x.
