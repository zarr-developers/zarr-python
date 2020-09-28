from zarr.util import normalize_storage_path
from zarr.errors import err_contains_array, err_contains_group


async def init_group(store, overwrite=False, path=None, chunk_store=None):
    """Initialize a group store. Note that this is a low-level function and there should be no
    need to call this directly from user code.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.

    """

    # normalize path
    path = normalize_storage_path(path)

    # initialise metadata
    _init_group_metadata(
        store=store, overwrite=overwrite, path=path, chunk_store=chunk_store
    )


async def _init_group_metadata(store, overwrite=False, path=None, chunk_store=None):

    # guard conditions
    if overwrite:
        raise NotImplementedError
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
        if chunk_store is not None:
            rmdir(chunk_store, path)
    elif await contains_array(store, path):
        err_contains_array(path)
    elif contains_group(store, path):
        err_contains_group(path)

    # initialize metadata
    # N.B., currently no metadata properties are needed, however there may
    # be in future
    meta = dict()
    raise NotImplementedError
    key = _path_to_prefix(path) + group_meta_key
    store[key] = encode_group_metadata(meta)


async def contains_array(store, path=None):
    """Return True if the store contains an array at the given logical path."""
    path = normalize_storage_path(path)
    key = "meta/root" + path + ".array"
    return key in await store.list()


def contains_group(store, path=None):
    """Return True if the store contains a group at the given logical path."""
    raise NotImplementedError
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = prefix + group_meta_key
    return key in store
