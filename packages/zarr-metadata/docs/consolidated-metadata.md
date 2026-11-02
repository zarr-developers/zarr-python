# Consolidated metadata

Consolidated metadata is a copy of every metadata document below a group,
kept in one place so that a reader can open a hierarchy with one read. It
is a convention of zarr-python, the reference implementation, and neither
Zarr specification defines it in full. This page says what is settled
before it describes the API.

## Status in the specifications

- **Zarr v2.** The v2 spec has no consolidated metadata. The `.zmetadata`
  file was designed and shipped in zarr-python
  [PR 268](https://github.com/zarr-developers/zarr-python/pull/268), merged in November 2018 and released in
  zarr-python 2.3.0, and is described in the
  [zarr-python 2 tutorial](https://zarr.readthedocs.io/en/v2.18.3/tutorial.html#consolidating-metadata).
  zarr-specs [issue 113](https://github.com/zarr-developers/zarr-specs/issues/113), on standardizing the
  format, is open.
- **Zarr v3.** zarr-specs [PR 309](https://github.com/zarr-developers/zarr-specs/pull/309), open since
  August 2024 and following [issue 136](https://github.com/zarr-developers/zarr-specs/issues/136), proposes the
  `consolidated_metadata` member with the `kind: "inline"` envelope that
  zarr-python 3 writes. It is not merged. The core spec does not define the
  member's entries. Since [PR 373](https://github.com/zarr-developers/zarr-specs/pull/373), merged in July 2026,
  it names the member "for historical reasons" and fixes its envelope: an
  object with `must_understand: false`, `kind: "inline"` and a `metadata`
  object ([spec text](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816)). That change closed
  [issue 371](https://github.com/zarr-developers/zarr-specs/issues/371), which found that the v3.1.0 extension
  rules, read strictly, put every consolidated group zarr-python 3 had
  written out of spec, because the envelope has no `name`.

This package reads the envelope the spec fixes, and the entries as
zarr-python writes them.

## Zarr v3

Consolidated metadata holds the hierarchy below its group, with the group
as the root. The document of the node at `/a/b` is stored under the key
`a/b`, as the [stored representation](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#stored-representation) strips
the leading `/` of a path. The documents and the group must form a tree:
only groups have child nodes, and the parent of every node is present.
`ZarrV3ConsolidatedMetadata` checks this when it reads the documents.

`NodeName` and `NodePath`, in `zarr_metadata.v3`, are the types of strings
that satisfy the spec's rules for [node names](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#node-names) and
paths. They are modeled on the zarrs types with the same names.
`validate_node_name_v3`, `is_node_name_v3` and `parse_node_name_v3`, and
the matching `node_path` functions, check a string against those rules.

### Reading

A group does not read the documents its `consolidated_metadata` lists.
Consolidated metadata is a convention of the reference implementation, not
core metadata, so a group holds the member as written, after it checks the
[envelope that the core spec fixes](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816). `CORE_CONTEXT` holds
`CONSOLIDATED_METADATA_EXTRA_FIELD`, the definition that reads a group's
`consolidated_metadata` as that envelope.

A reader can use the member as a source of child documents in place of the
store: read one document with `read_node_metadata_v3` when that node is
opened, or read all of them with the group's `consolidated()`, with
`ZarrV3ConsolidatedMetadata`, or with `read_consolidated_metadata_v3`,
which returns problems instead of raising them. A document with a problem
then fails only when it is read. `consolidated()` reads the documents in
the group's own context. The other two take a `context`; pass
`context=group.context` to read the documents in the group's own context,
and with no `context` they are read in `CORE_AND_EXTENSIONS_CONTEXT`.

### Writing

To write consolidated metadata, build a `ZarrV3ConsolidatedMetadata` and
pass it to the group's `update`. The group reads the listed models in its
own context, refuses one that its context reads differently, and writes the
document. The model reads every document strictly, and also accepts node
models as entries. Such an entry is accepted when its claims refine into
the model's context, and refused at its path otherwise. To consolidate
children that were read in several contexts, consolidate them in the
`union` of those contexts.

## Zarr v2

A v2 hierarchy's consolidated metadata is the `.zmetadata` document at the
root of the store, as zarr-python 2.3 and later write it: an object with
`zarr_consolidated_format: 1` and a `metadata` object that maps each
metadata file's path to that file's contents, under keys such as
`a/.zarray`, `a/.zattrs` and `a/b/.zgroup`. The format comes from
zarr-python [PR 268](https://github.com/zarr-developers/zarr-python/pull/268); no specification defines it.

`ZarrV2ConsolidatedMetadata` reads such a document, and
`read_consolidated_metadata_v2` returns problems instead of raising them.
Both take a `context`, `CORE_V2_CONTEXT` when none is given. Each `.zarray` or
`.zgroup` entry is read as a node, merged with its sibling `.zattrs`. A
`.zattrs` with no sibling is kept but forms no node, and any other entry
must be JSON and is kept. `to_json` returns the entries keyed by file, as
written. The `.zmetadata` that zarr-python 3 writes for a v2 hierarchy has
a known bug, which `read_repaired_consolidated_metadata_v2` undoes; see
[Repairing known writer bugs](repair.md).
