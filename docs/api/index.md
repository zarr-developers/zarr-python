# API Reference

Complete reference documentation for the Zarr-Python API.

## Core API

### Essential Classes and Functions

- **[Array](array.md)** - The main Zarr array class for N-dimensional data
- **[Group](group.md)** - Hierarchical organization of arrays and subgroups
- **[Create](create.md)** - Functions for creating new arrays and groups
- **[Open](open.md)** - Opening existing Zarr stores and arrays

### Data Operations

- **[Load](load.md)** - Loading data from Zarr stores
- **[Save](save.md)** - Saving data to Zarr format
- **[Convenience](convenience.md)** - High-level convenience functions

### Data Types and Configuration

- **[Data Types](dtype.md)** - Supported NumPy data types and type handling
- **[Configuration](config.md)** - Runtime configuration and settings

## Storage and Compression

- **[Codecs](codecs.md)** - Compression and filtering codecs
- **[Storage](storage.md)** - Storage backend implementations and interfaces
- **[Registry](registry.md)** - Codec and storage backend registry

## API Variants

Zarr-Python provides both synchronous and asynchronous APIs:

- **[Async API](api_async.md)** - Asynchronous operations for concurrent access
- **[Sync API](api_sync.md)** - Synchronous operations for simple usage

## Abstract Base Classes

The ABC module defines interfaces for extending Zarr:

- **[Codec ABC](abc/codec.md)** - Interface for custom compression codecs
- **[Metadata ABC](abc/metadata.md)** - Interface for metadata handling
- **[Store ABC](abc/store.md)** - Interface for custom storage backends

## Utilities

- **[Errors](errors.md)** - Exception classes and error handling
- **[Testing](testing.md)** - Utilities for testing Zarr-based code


## Migration and Compatibility

- **[Deprecated Functions](deprecated/convenience.md)** - Legacy convenience functions
- **[Deprecated Creation](deprecated/creation.md)** - Legacy array creation functions

These deprecated modules are maintained for backward compatibility but should be avoided in new code.

## Getting Help

- Check the [User Guide](../user-guide/index.md) for tutorials and examples
- Browse function signatures and docstrings in the API reference
- Report issues on [GitHub](https://github.com/zarr-developers/zarr-python)
- Join discussions on the [Zarr community forum](https://github.com/zarr-developers/community)
