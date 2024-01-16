# Zarr Python Roadmap

- Status: draft
- Author: Joe Hamman
- Created On: October 31, 2023
- Input from:
    - Davis Bennett / @d-v-b
    - Norman Rzepka / @normanrz
    - Deepak Cherian @dcherian
    - Brian Davis / @monodeldiablo
    - Oliver McCormack / @olimcc
    - Ryan Abernathey / @rabernat
    - Jack Kelly / @JackKelly
    - Martin Durrant / @martindurant

## Introduction

This document lays out a design proposal for version 3.0 of the [Zarr-Python](https://zarr.readthedocs.io/en/stable/) package. A specific focus of the design is to bring Zarr-Python's API up to date with the [Zarr V3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html), with the hope of enabling the development of the many features and extensions that motivated the V3 Spec. The ideas presented here are expected to result in a major release of Zarr-Python (version 3.0) including significant a number of breaking API changes.
For clarity, “V3” will be used to describe the version of the Zarr specification and “3.0” will be used to describe the release tag of the Zarr-Python project.

### Current status of V3 in Zarr-Python

During the development of the V3 Specification, a [prototype implementation](https://github.com/zarr-developers/zarr-python/pull/898) was added to the Zarr-Python library. Since that implementation, the V3 spec evolved in significant ways and as a result, the Zarr-Python library is now out of sync with the approved spec. Downstream libraries (e.g. [Xarray](https://github.com/pydata/xarray)) have added support for this implementation and will need to migrate to the accepted spec when its available in Zarr-Python.

## Goals

- Provide a complete implementation of Zarr V3 through the Zarr-Python API
- Clear the way for exciting extensions / ZEPs (i.e. [sharding](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html), [variable chunking](https://zarr.dev/zeps/draft/ZEP0003.html), etc.)
- Provide a developer API that can be used to implement and register V3 extensions
- Improve the performance of Zarr-Python by streamlining the interface between the Store layer and higher level APIs (e.g. Groups and Arrays)
- Clean up the internal and user facing APIs 
- Improve code quality and robustness (e.g. achieve 100% type hint coverage)
- Align the Zarr-Python array API with the [array API Standard](https://data-apis.org/array-api/latest/)

## Examples of what 3.0 will enable?
1. Reading and writing V3 spec-compliant groups and arrays
2. V3 extensions including sharding and variable chunking.
3. Improved performance by leveraging concurrency when creating/reading/writing to stores (imagine a `create_hierarchy(zarr_objects)` function).
4. User-developed extensions (e.g. storage-transformers) can be registered with Zarr-Python at runtime

## Non-goals (of this document)

- Implementation of any unaccepted Zarr V3 extensions
- Major revisions to the Zarr V3 spec

## Requirements

1. Read and write spec compliant V2 and V3 data
2. Limit unnecessary traffic to/from the store
3. Cleanly define the Array/Group/Store abstractions
4. Cleanly define how V2 will be supported going forward
5. Provide a clear roadmap to help users upgrade to 3.0
6. Developer tools / hooks for registering extensions

## Design

### Async API

Zarr-Python is an IO library. As such, supporting concurrent action against the storage layer is critical to achieving acceptable performance. The Zarr-Python 2 was not designed with asynchronous computation in mind and as a result has struggled to effectively leverage the benefits of concurrency. At one point, `getitems` and `setitems` support was added to the Zarr store model but that is only used for operating on a set of chunks in a single variable. 

With Zarr-Python 3.0, we have the opportunity to revisit this design. The proposal here is as follows:

1. The `Store` interface will be entirely async. 
2. On top of the async `Store` interface, we will provide an  `AsyncArray` and `AsyncGroup` interface.
3. Finally, the primary user facing API will be synchronous `Array` and `Group` classes that wrap the async equivalents.

**Examples**

- **Store**
    
    ```python
    class Store:
        ...
        async def get(self, key: str) -> bytes:
            ...
        async def get_partial_values(self, key_ranges: List[Tuple[str, Tuple[int, Optional[int]]]]) -> bytes:
            ...
    # (no sync interface here)
    ``` 
- **Array**
    
    ```python
    class AsyncArray:
        ...
    
        async def getitem(self, selection: Selection) -> np.ndarray:
           # the core logic for getitem goes here
    
    class Array:
        _async_array: AsyncArray
    
        def __getitem__(self, selection: Selection) -> np.ndarray:
            return sync(self._async_array.getitem(selection))
    ```
- **Group**
    
    ```python
    class AsyncGroup:
        ...
    
        async def create_group(self, path: str, **kwargs) -> AsyncGroup:
           # the core logic for create_group goes here
    
    class Group:
        _async_group: AsyncGroup
    
        def create_group(self, path: str, **kwargs) -> Group:
            return sync(self._async_group.create_group(path, **kwargs))
    ```
**Internal Synchronization API**

With the `Store` and core `AsyncArray`/ `AsyncGroup` classes being predominantly async, Zarr-Python will need an internal API to provide a synchronous API. The proposal here is to use the approach in [fsspec](https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py) to provide a high-level `sync` function that takes an `awaitable` and runs it in its managed IO Loop / thread. 

**FAQ**
1. Why two levels of Arrays/groups?
  a. First, this is an intentional decision and departure from the current Zarrita implementation
  b. The idea is that users rarely want to mix interfaces. Either they are working within an async context (currently quite rare) or they are in a typical synchronous context. 
  c. Splitting the two will allow us to clearly define behavior on the `AsyncObj` and simply wrap it in the `SyncObj`.
2. What if a store is only has a synchronous backend?
  a. First off, this is expected to be a fairly rare occurrence. Most storage backends have async interfaces.
  b. But in the event a storage backend doesn’t have a async interface, there is nothing wrong with putting synchronous code in `async` methods. There are approaches to enabling concurrent action through wrappers like AsyncIO's `loop.run_in_executor` ([ref 1](https://stackoverflow.com/questions/38865050/is-await-in-python3-cooperative-multitasking ), [ref 2](https://stackoverflow.com/a/43263397/732596), [ref 3](https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-5.html), [ref 4](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor).  
3. Will Zarr help manage the async contexts encouraged by some libraries (e.g. [AioBotoCore](https://aiobotocore.readthedocs.io/en/latest/tutorial.html#using-botocore))?
  a. Many async IO libraries require entering an async context before interacting with the API. We expect some experimentation to be needed here but the initial design will follow something close to what fsspec does ([example in s3fs](https://github.com/fsspec/s3fs/blob/949442693ec940b35cda3420c17a864fbe426567/s3fs/core.py#L527)).
4. Why not provide a synchronous Store interface?
  a. We could but this design is simpler. It would mean supporting it in the `AsyncGroup` and `AsyncArray` classes which, may be more trouble than its worth. Storage backends that do not have an async API will be encouraged to wrap blocking calls in an async wrapper (e.g. `loop.run_in_executor`).

### Store API

The `Store` API is specified directly in the V3 specification. All V3 stores should implement this abstract API, omitting Write and List support as needed. As described above, all stores will be expected to expose the required methods as async methods.

**Example**

```python
class ReadWriteStore:
		...
    async def get(self, key: str) -> bytes:
        ...

    async def get_partial_values(self, key_ranges: List[Tuple[str, int, int]) -> bytes:
        ...
		
    async def set(self, key: str, value: Union[bytes, bytearray, memoryview]) -> None:
        ...  # required for writable stores

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, Union[bytes, bytearray, memoryview]]]) -> None:
        ...  # required for writable stores

    async def list(self) -> List[str]:
        ...  # required for listable stores

    async def list_prefix(self, prefix: str) -> List[str]:
        ...  # required for listable stores

    async def list_dir(self, prefix: str) -> List[str]:
        ...  # required for listable stores

    # additional (optional methods)
    async def getsize(self, prefix: str) -> int:
        ...
        
    async def rename(self, src: str, dest: str) -> None
        ...
        
```

Recognizing that there are many Zarr applications today that rely on the `MutableMapping` interface supported by Zarr-Python 2, a wrapper store will be developed to allow existing stores to plug directly into this API.

### Array API

The user facing array interface will implement a subset of the [Array API Standard](https://data-apis.org/array-api/latest/). Most of the computational parts of the Array API Standard don’t fit into Zarr right now. That’s okay. What matters most is that we ensure we can give downstream applications a compliant API. 

*Note, Zarr already does most of this so this is more about formalizing the relationship than a substantial change in API.*

|  | Included | Not Included | Unknown / Maybe possible? |
| --- | --- | --- | --- |
| Attributes | `dtype` | `mT` | `device` |
| | `ndim` | `T` | |
| | `shape` | | |
| | `size` | | |
| Methods | `__getitem__` | `__array_namespace__` | `to_device` |
| | `__setitem__` | `__abs__` | `__bool__` |
| | `__eq__` | `__add__` | `__complex__` |
| | `__bool__` | `__and__` | `__dlpack__` |
| | | `__floordiv__` | `__dlpack_device__` |
| | | `__ge__` | `__float__` |
| | | `__gt__` | `__index__` |
| | | `__invert__` | `__int__` |
| | | `__le__` | |
| | | `__lshift__` | |
| | | `__lt__` | |
| | | `__matmul__` | |
| | | `__mod__` | |
| | | `__mul__` | |
| | | `__ne__` | |
| | | `__neg__` | |
| | | `__or__` | |
| | | `__pos__` | |
| | | `__pow__` | |
| | | `__rshift__` | |
| | | `__sub__` | |
| | | `__truediv__` | |
| | | `__xor__` | |
| Creation functions (`zarr.creation`) | `zeros` | | `arange` |
| | `zeros_like` | | `asarray` |
| | `ones` | | `eye` |
| | `ones_like` | | `from_dlpack` |
| | `full` | | `linspace` |
| | `full_like` | | `meshgrid` |
| | `empty` | | `tril` |
| | `empty_like` | | `triu` |

In addition to the core array API defined above, the Array class should have the following Zarr specific properties:

- `.metadata` (see Metadata Interface below)
- `.attrs` - (pull from metadata object)
- `.info` - (pull from existing property †)

*† In Zarr-Python 2, the info property lists the store to identify initialized chunks. By default this will be turned off in 3.0 but will be configurable.*

**Indexing**

Zarr-Python currently supports `__getitem__` style indexing and the special `oindex` and `vindex` indexers. These are not part of the current Array API standard (see [data-apis/array-api\#669](https://github.com/data-apis/array-api/issues/669)) but they have been [proposed as a NEP](https://numpy.org/neps/nep-0021-advanced-indexing.html). Zarr-Python will maintain these in 3.0.

We are also exploring a new high-level indexing API that will enabled optimized batch/concurrent loading of many chunks. We expect this to be important to enable performant loading of data in the context of sharding. See [this discussion](https://github.com/zarr-developers/zarr-python/discussions/1569) for more detail.

Concurrent indexing across multiple arrays will be possible using the AsyncArray API.

**Async and Sync Array APIs**

Most the logic to support Zarr Arrays will live in the `AsyncArray` class. There are a few notable differences that should be called out.

| Sync Method | Async Method |
| --- | --- |
| `__getitem__` | `getitem` |
| `__setitem__` | `setitem` |
| `__eq__` | `equals` |

**Metadata interface**

Zarr-Python 2.* closely mirrors the V2 spec metadata schema in the Array and Group classes. In 3.0, we plan to move the underlying metadata representation to a separate interface (e.g. `Array.metadata`). This interface will return either a `V2ArrayMetadata` or `V3ArrayMetadata` object (both will inherit from a parent `ArrayMetadataABC` class. The  `V2ArrayMetadata` and `V3ArrayMetadata` classes will be responsible for producing valid JSON representations of their metadata, and yielding a consistent view to the `Array` or `Group` class.

### Group API

The main question is how closely we should follow the existing Zarr-Python implementation / `MutableMapping` interface. The table below shows the primary `Group` methods in Zarr-Python 2 and attempts to identify if and how they would be implemented in 3.0.

| V2 Group Methods | `AsyncGroup` | `Group` | `h5py_compat.Group`` |
| --- | --- | --- | --- |
| `__len__` | `length` | `__len__` | `__len__` |
| `__iter__` | `__aiter__` | `__iter__` | `__iter__` |
| `__contains__` | `contains` | `__contains__` | `__contains__` |
| `__getitem__` | `getitem` | `__getitem__` | `__getitem__` |
| `__enter__` | N/A | N/A | `__enter__` |
| `__exit__` | N/A | N/A | `__exit__` |
| `group_keys` | `group_keys` | `group_keys` | N/A |
| `groups` | `groups` | `groups` | N/A |
| `array_keys` | `array_key` | `array_keys` | N/A |
| `arrays` | `arrays`* | `arrays` | N/A |
| `visit` | ? | ? | `visit` |
| `visitkeys` | ? | ? | ? |
| `visitvalues` | ? | ? | ? |
| `visititems` | ? | ? | `visititems` |
| `tree` | `tree` | `tree` | `Both` |
| `create_group` | `create_group` | `create_group` | `create_group` |
| `require_group` | N/A | N/A | `require_group` |
| `create_groups` | ? | ? | N/A |
| `require_groups` | ? | ? | ? |
| `create_dataset` | N/A | N/A | `create_dataset` |
| `require_dataset` | N/A | N/A | `require_dataset` |
| `create` | `create_array` | `create_array` | N/A |
| `empty` | `empty` | `empty` | N/A |
| `zeros` | `zeros` | `zeros` | N/A |
| `ones` | `ones` | `ones` | N/A |
| `full` | `full` | `full` | N/A |
| `array` | `create_array` | `create_array` | N/A |
| `empty_like` | `empty_like` | `empty_like` | N/A |
| `zeros_like` | `zeros_like` | `zeros_like` | N/A |
| `ones_like` | `ones_like` | `ones_like` | N/A |
| `full_like` | `full_like` | `full_like` | N/A |
| `move` | `move` | `move` | `move` |

**`zarr.h5compat.Group`**

Zarr-Python 2.* made an attempt to align its API with that of [h5py](https://docs.h5py.org/en/stable/index.html). With 3.0, we will relax this alignment in favor of providing an explicit compatibility module (`zarr.h5py_compat`). This module will expose the `Group` and `Dataset` APIs that map to Zarr-Python’s `Group` and `Array` objects. 

### Creation API

Zarr-Python 2.* bundles together the creation and serialization of Zarr objects. Zarr-Python 3.* will make it possible to create objects in memory separate from serializing them. This will specifically enable writing hierarchies of Zarr objects in a single batch step. For example:

```python

arr1 = Array(shape=(10, 10), path="foo/bar", dtype="i4", store=store)
arr2 = Array(shape=(10, 10), path="foo/spam", dtype="f8", store=store)

arr1.save()
arr2.save()

# or equivalently

zarr.save_many([arr1 ,arr2])
```

*Note: this batch creation API likely needs additional design effort prior to implementation.*

### Plugin API

Zarr V3 was designed to be extensible at multiple layers. Zarr-Python will support these extensions through a combination of [Abstract Base Classes](https://docs.python.org/3/library/abc.html) (ABCs) and [Entrypoints](https://packaging.python.org/en/latest/specifications/entry-points/). 

**ABCs**

Zarr V3 will expose Abstract base classes for the following objects:

- `Store`, `ReadStore`, `ReadWriteStore`, `ReadListStore`, and `ReadWriteListStore`
- `BaseArray`, `SynchronousArray`, and `AsynchronousArray`
- `BaseGroup`, `SynchronousGroup`, and `AsynchronousGroup`
- `Codec`, `ArrayArrayCodec`, `ArrayBytesCodec`, `BytesBytesCodec`

**Entrypoints**

Lots more thinking here but the idea here is to provide entrypoints for `data type`, `chunk grid`, `chunk key encoding`, `codecs`,  `storage_transformers` and `stores`.  These might look something like:

```
entry_points="""
    [zarr.codecs]
    blosc_codec=codec_plugin:make_blosc_codec
    zlib_codec=codec_plugin:make_zlib_codec
"""
```

### Python type hints and static analysis

Target 100% Mypy coverage in 3.0 source. 

### Observability

A persistent problem in Zarr-Python is diagnosing problems that span many parts of the stack. To address this in 3.0, we will add a basic logging framework that can be used to debug behavior at various levels of the stack. We propose to add the separate loggers for the following namespaces:

- `array`
- `group`
- `store`
- `codec`

These should be documented such that users know how to activate them and developers know how to use them when developing extensions.

### Dependencies

Today, Zarr-Python has the following required dependencies:

```python
dependencies = [
    'asciitree',
    'numpy>=1.20,!=1.21.0',
    'fasteners',
    'numcodecs>=0.10.0',
]
```

What other dependencies should be considered?

1. Attrs - Zarrita makes extensive use of the Attrs library
2. Fsspec - Zarrita has a hard dependency on Fsspec. This could be easily relaxed though.

## Breaking changes relative to Zarr-Python 2.*

1. H5py compat moved to a stand alone module?
2. `Group.__getitem__` support moved to `Group.members.__getitem__`?
3. Others? 

## Open questions

1. How to treat V2
    a. Note: Zarrita currently implements a separate `V2Array` and `V3Array` classes. This feels less than ideal.
    b. We could easily convert metadata from v2 to the V3 Array, but what about writing? 
    c. Ideally, we don’t have completely separate code paths. But if its too complicated to support both within one interface, its probably better.
2. How and when to remove the current implementation of V3.
    a. It's hidden behind a hard-to-use feature flag so we probably don't need to do anything.
4. How to model runtime configuration?
5. Which extensions belong in Zarr-Python and which belong in separate packages?
    a. We don't need to take a strong position on this here. It's likely that someone will want to put Sharding in. That will be useful to develop in parallel because it will give us a good test case for the plugin interface.

## Testing

Zarr-python 3.0 adds a major new dimension to Zarr: Async support. This also comes with a compatibility risk, we will need to thoroughly test support in key execution environments. Testing plan:
- Reuse the existing test suite for testing the `v3` API.
    - `xfail` tests that expose breaking changes with `3.0 - breaking change` description. This will help identify additional and/or unintentional breaking changes
    - Rework tests that were only testing internal APIs.
- Add a set of functional / integration tests targeting real-world workflows in various contexts (e.g. w/ Dask)

## Development process

Zarr-Python 3.0 will introduce a number of new APIs and breaking changes to existing APIs. In order to facilitate ongoing support for Zarr-Python 2.*, we will take on the following development process:

- Create a `v3` branch that can be use for developing the core functionality apart from the `main` branch. This will allow us to support ongoing work and bug fixes on the `main` branch.
- Put the `3.0` APIs inside a `zarr.v3` module. Imports from this namespace will all be new APIs that users can develop and test against once the `v3` branch is merged to `main`.
- Kickstart the process by pulling in the current state of `zarrita` - which has many of the features described in this design.
- Release a series of 2.* releases with the `v3` namespace
- When `v3` is complete, move contents of `v3` to the package root

**Milestones**

Below are a set of specific milestones leading toward the completion of this process. As work begins, we expect this list to grow in specificity.

1. Port current version of Zarrita to Zarr-Python
2. Formalize Async interface by splitting `Array` and `Group` objects into Sync and Async versions
4. Implement "fancy" indexing operations on the `AsyncArray`
6. Implement an abstract base class for the `Store` interface and a wrapper `Store` to make use of existing `MutableMapping` stores. 
7. Rework the existing unit test suite to use the `v3` namespace.
8. Develop a plugin interface for extensions
9. Develop a set of functional and integration tests
10. Work with downstream libraries (Xarray, Dask, etc.) to test new APIs 

## TODOs

The following subjects are not covered in detail above but perhaps should be. Including them here so they are not forgotten.

1. [Store] Should Zarr provide an API for caching objects after first read/list/etc. Read only stores?
2. [Array] buffer protocol support
3. [Array] `meta_array` support
4. [Extensions] Define how Zarr-Python will consume the various plugin types
5. [Misc] H5py compatibility requires a bit more work and a champion to drive it forward.
6. [Misc] Define `chunk_store` API in 3.0
7. [Misc] Define `synchronizer` API in 3.0

## References

1. [Zarr-Python repository](https://github.com/zarr-developers/zarr-python)
2. [Zarr core specification (version 3.0) — Zarr specs  documentation](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#)
3. [Zarrita repository](https://github.com/scalableminds/zarrita)
4. [Async-Zarr](https://github.com/martindurant/async-zarr)
5. [Zarr-Python Discussion Topic](https://github.com/zarr-developers/zarr-python/discussions/1569)
