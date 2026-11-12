# What is Zarr? The six layers

"Zarr" is not one thing. It is a stack of six layers, and you can opt in to
each layer independently. This page explains the layers, how they relate, and
where each part of zarr-python — and each neighbouring project — fits in.

Most people meet only the top of the stack: they open an array with
[`zarr.open_array`][] and never think about what is underneath. That is fine
until something goes wrong, or until you want to build something new on top of
Zarr. Then it helps to know which layer you are standing on, because each
layer has different owners, different guarantees, and different rules for
changing it.

## The six layers at a glance

| Layer | What it is | Where it lives |
| --- | --- | --- |
| 1. Native layout | The familiar on-disk arrangement of metadata documents and chunk files | A directory, an object store, a Zip file |
| 2. Specification | The rules for serializing array data and metadata to a key-value store | [zarr-specs.readthedocs.io](https://zarr-specs.readthedocs.io) |
| 3. Store ABC | The Python interface any key-value store must implement to work with Zarr | [`zarr.abc.store.Store`][] |
| 4. Standard stores | The stores zarr-python ships | [`zarr.storage`][] |
| 5. Python API | The functions and classes for reading and writing arrays and groups | [`zarr.api`][] and the top-level `zarr` namespace |
| 6. Extensions | Extra metadata conventions and extension points built on top | Registries, attributes, community conventions |

The layers are ordered from the most concrete (bytes on a disk) to the most
abstract (conventions shared by a community). A higher layer depends on the
layers below it, but it does not have to be used with them: a store can
implement layer 3 without ever writing layer 1, and an application can use
layer 5 without knowing which store sits underneath it.

## Layer 1: The native on-disk layout

The layout most people picture when they hear "Zarr" is a directory
containing one metadata document per array or group plus one file per chunk,
with chunk file names derived from chunk coordinates. In Zarr format 3 the
metadata document is `zarr.json`; in Zarr format 2 it is the `.zarray` and
`.zgroup` files (plus `.zattrs` for user attributes). The same arrangement
works on a local file system, in an object store such as S3, or inside a Zip
file — the only difference is how the keys are mapped onto the underlying
storage.

This layout is a convention that follows from layer 2, not a separate formal
specification of its own. That is worth knowing, because it explains why
readers in any language can be written for it without much ceremony — and
also why a store is free to use a completely different layout and still be
part of the Zarr ecosystem, as long as it honours the layers above it.

## Layer 2: The specification

The [Zarr specifications](https://zarr-specs.readthedocs.io) define how array
and group metadata and chunk data are serialized to byte streams addressed by
keys in an arbitrary key-value store. The specification is language
independent: it says what bytes a key must hold, not how any particular
programming language should read them. Zarr-python supports both format 2 and
format 3 of the specification.

Crucially, the specification is layer 2 and only layer 2. It does not define
a Python API, a store interface, or any on-disk format beyond the mapping
from keys to bytes. When someone says "the Zarr spec", this is what they mean.

## Layer 3: The Store ABC

[`zarr.abc.store.Store`][] is the abstract base class that any key-value
store must implement to be usable by zarr-python. It defines an asynchronous
interface for reading and writing byte ranges by key — `get`, `set`, `delete`,
`exists`, and listing (`list`, `list_prefix`, `list_dir`) — together with
capability flags such as `supports_writes`, `supports_deletes`,
`supports_listing`, and `supports_consolidated_metadata`, so that the layers
above can ask a store what it can do instead of assuming.

A store implementation can do whatever it likes behind that interface,
including not writing the native layout of layer 1 at all. That is the whole
point of the abstraction: the array code in layer 5 only ever talks to a
`Store`, so any storage system that implements the interface becomes
interchangeable with the built-in stores. See
[Developing custom stores](storage.md#developing-custom-stores) for how to
write one.

## Layer 4: The standard store implementations

[`zarr.storage`][] ships the canonical store implementations, which generally
do write using the native layout of layer 1:

- [`zarr.storage.LocalStore`][] — a directory on the local file system.
- [`zarr.storage.ZipStore`][] — a single Zip file.
- [`zarr.storage.FsspecStore`][] — any remote or local storage reachable
  through [fsspec](https://filesystem-spec.readthedocs.io/en/latest/)
  (S3, HTTP, and more).
- [`zarr.storage.ObjectStore`][] — object storage via
  [obstore](https://developmentseed.org/obstore/latest/).
- [`zarr.storage.MemoryStore`][] (and [`zarr.storage.ManagedMemoryStore`][])
  — in-memory storage, useful for testing and temporary data.
- [`zarr.storage.GpuMemoryStore`][] — in-memory storage on a GPU device.
- [`zarr.storage.WrapperStore`][] and [`zarr.storage.LoggingStore`][] —
  wrappers that add behaviour (such as logging) around another store.

See the [storage guide](storage.md) for details and examples of each.

## Layer 5: The Python API

[`zarr.api`][] — exposed as the top-level `zarr` namespace, with synchronous
wrappers in [`zarr.api.synchronous`][] — is the API most users interact with:
[`zarr.create_array`][], [`zarr.open_group`][], [`zarr.open_array`][],
[`zarr.Array`][], [`zarr.Group`][], and the `zarr.zeros` / `zarr.ones` /
`zarr.empty` family. It provides NumPy-like semantics for creating, reading,
and writing arrays and groups.

The API only ever interacts with implementations of layer 3: you hand it a
`Store` (or a [`zarr.storage.StoreLike`][] value such as a path string, from
which a store is created implicitly), and it works the same way regardless of
which store that is. That is why the same code can read from a local
directory, an S3 bucket, or a Zip file, and why third-party stores can be
dropped in without changes to user code.

## Layer 6: Extensions and metadata standards

Layer 6 is everything built on top of the other layers: informal metadata
conventions, formal extension proposals, and the extension points
zarr-python provides for them.

- **Custom codecs and data types.** New codecs can be registered through the
  [`zarr.registry`][] (see [Extending Zarr](extending.md)), and new data
  types can be defined as described in the
  [data types guide](data_types.md). Note that the extension mechanism for
  Zarr format 3 is still under development, so the details of custom codecs
  may change.
- **Consolidated metadata.** A single document holding the metadata of a
  whole hierarchy, so opening a group does not need one read per node. It is
  implemented for both format 2 and format 3; for format 3 it is still
  experimental, with a formal extension proposed in
  [zarr-specs#309](https://github.com/zarr-developers/zarr-specs/pull/309).
  See [Consolidated Metadata](consolidated_metadata.md). Support for it is
  a property of the store (layer 3) as well as of the API (layer 5), so not
  every store provides it.
- **Attributes.** Every array and group can carry free-form user attributes
  (see [Attributes](attributes.md)). A community can agree on what particular
  attributes mean — that agreement is a layer 6 convention, not part of the
  core specification.
- **Community metadata standards.** Efforts such as OME-Zarr (for microscopy
  data) and GeoZarr (for geospatial data) live entirely at this layer: they
  define what metadata to write and how to interpret it, on top of ordinary
  Zarr arrays and groups.

## How the projects map to the layers

Because projects opt in to some layers and not others, "is X Zarr?" is usually
the wrong question. The useful question is "which layers does X implement?"

| Project | Layers |
| --- | --- |
| The Zarr specification | 2 only |
| zarr-python | 3 (`zarr.abc`), 4 (`zarr.storage`), 5 (`zarr.api`) — reading and writing the layouts of layer 1 according to layer 2 |
| Zarr implementations in other languages | 1 and 2, with varying inspiration from 4 and 5 |
| VirtualiZarr's `ManifestStore` | 3, deliberately without 1 — it exposes non-Zarr data formats on disk through layer 5 |
| Icechunk | 3 and 5 in its Python API; its own storage format is an alternative to layer 1 that still follows layer 2 |
| Kerchunk | Its own replacement for layer 1, read through the [`zarr.storage.FsspecStore`][] of layer 4 |
| Xarray | Primarily a user of layer 5, plus a few layer 6 conventions of its own (for example the `coordinates` attribute) |
| OME-Zarr, GeoZarr | 6 only |

## Which layer do you need?

- **You just want to read or write arrays.** Stay at layer 5. Start with the
  [quick start](../quick-start.md) and the [user guide](index.md).
- **You want to choose or configure storage.** Read the
  [storage guide](storage.md) (layers 3 and 4).
- **You want to add a storage backend.** Implement layer 3 — subclass
  [`zarr.abc.store.Store`][] — and your store will work with all of layer 5.
- **You want to add a codec, data type, or metadata convention.** That is
  layer 6; see [Extending Zarr](extending.md) and the
  [consolidated metadata guide](consolidated_metadata.md).
- **You are writing a Zarr implementation in another language.** Layer 2 is
  your contract, and layer 1 is the layout your readers and writers must
  agree on.
