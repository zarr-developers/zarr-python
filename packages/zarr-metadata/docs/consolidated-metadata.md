# Consolidated metadata

Consolidated metadata is a copy of every metadata document below a group,
kept in one place. A reader can then open a whole hierarchy with one read.

Consolidated metadata is a convention of zarr-python, the reference
implementation. Neither Zarr specification defines it in full. This page
first says what is settled, then describes the API.

## Status in the specifications

**Zarr v2.** The v2 spec has no consolidated metadata.

- The `.zmetadata` file was designed in zarr-python
  [PR 268](https://github.com/zarr-developers/zarr-python/pull/268). It
  merged in November 2018 and shipped in zarr-python 2.3.0.
- The
  [zarr-python 2 tutorial](https://zarr.readthedocs.io/en/v2.18.3/tutorial.html#consolidating-metadata)
  describes it.
- zarr-specs [issue 113](https://github.com/zarr-developers/zarr-specs/issues/113),
  on standardizing the format, is open.

**Zarr v3.** The core spec names the member but does not define its
entries.

- zarr-specs [PR 309](https://github.com/zarr-developers/zarr-specs/pull/309)
  proposes the `consolidated_metadata` member, with the `kind: "inline"`
  envelope that zarr-python 3 writes. It follows
  [issue 136](https://github.com/zarr-developers/zarr-specs/issues/136). It
  has been open since August 2024 and is not merged.
- [PR 373](https://github.com/zarr-developers/zarr-specs/pull/373) merged
  in July 2026. Since then, the core spec names the member "for historical
  reasons" and fixes its envelope
  ([spec text](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816)).
  The envelope is an object with:
    - `must_understand: false`,
    - `kind: "inline"`,
    - a `metadata` object.
- PR 373 closed
  [issue 371](https://github.com/zarr-developers/zarr-specs/issues/371).
  That issue found a problem with the v3.1.0 extension rules. Read
  strictly, they put every consolidated group zarr-python 3 had written out
  of spec, because the envelope has no `name`.

This package reads the envelope the spec fixes. It reads the entries as
zarr-python writes them.

## Zarr v3

Consolidated metadata holds the hierarchy below its group, with the group
as the root.

- The document of the node at `/a/b` is stored under the key `a/b`. The
  [stored representation](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#stored-representation)
  strips the leading `/` of a path.
- The documents and the group must form a tree. Only groups have child
  nodes, and the parent of every node is present.
  `ZarrV3ConsolidatedMetadata` checks this when it reads the documents.

`NodeName` and `NodePath`, in `zarr_metadata.v3`, are string types. They
satisfy the spec's rules for
[node names](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#node-names)
and paths. They are modeled on the zarrs types with the same names. These
functions check a string against those rules:

- `validate_node_name_v3`, `is_node_name_v3` and `parse_node_name_v3`,
- the matching `node_path` functions.

### Reading

A group does not read the documents its `consolidated_metadata` lists.
Consolidated metadata is a convention of the reference implementation, not
core metadata. So the group checks the
[envelope that the core spec fixes](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816),
then holds the member as written. `CORE_CONTEXT` holds
`CONSOLIDATED_METADATA_EXTRA_FIELD`, the definition that reads this
envelope.

A reader can use the member as a source of child documents, in place of
the store. There are two ways:

- Read one document with `read_node_metadata_v3` when that node is opened.
- Read all of them at once, with one of:
    - the group's `consolidated()`,
    - `ZarrV3ConsolidatedMetadata`,
    - `read_consolidated_metadata_v3`, which returns problems instead of
      raising them.

Either way, a document with a problem fails only when it is read.

Which context the documents are read in:

- `consolidated()` uses the group's own context.
- The other three take a `context`. Pass `context=group.context` to use the
  group's own context. With no `context`, they use
  `CORE_AND_EXTENSIONS_CONTEXT`.

### Writing

To write consolidated metadata:

1. Build a `ZarrV3ConsolidatedMetadata`.
2. Pass it to the group's `replace`.

The group reads the listed models in its own context. It refuses a model
that its context reads differently. Then it writes the document.

`ZarrV3ConsolidatedMetadata` reads every document strictly. It also
accepts node models as entries. It accepts such an entry when the entry's
claims refine into the model's context. Otherwise it refuses the entry at
its path. To consolidate children read in several contexts, consolidate
them in the `union` of those contexts.

## Zarr v2

A v2 hierarchy's consolidated metadata is the `.zmetadata` document at the
root of the store. zarr-python 2.3 and later write it. It is an object
with:

- `zarr_consolidated_format: 1`,
- a `metadata` object. This maps each metadata file's path to that file's
  contents, under keys such as `a/.zarray`, `a/.zattrs` and `a/b/.zgroup`.

The format comes from zarr-python
[PR 268](https://github.com/zarr-developers/zarr-python/pull/268). No
specification defines it.

`ZarrV2ConsolidatedMetadata` reads such a document.
`read_consolidated_metadata_v2` does the same, but returns problems instead
of raising them. Both take a `context`. The default is `CORE_V2_CONTEXT`.

How the entries are read:

- Each `.zarray` or `.zgroup` entry is read as a node, merged with its
  sibling `.zattrs`.
- A `.zattrs` with no sibling is kept, but forms no node.
- Any other entry must be JSON, and is kept.

`to_json` returns the entries keyed by file, as written.

The `.zmetadata` that zarr-python 3 writes for a v2 hierarchy has a known
bug. `read_repaired_consolidated_metadata_v2` undoes it. See
[Repairing known writer bugs](repair.md).
