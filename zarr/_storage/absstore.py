"""This module contains storage classes related to Azure Blob Storage (ABS)"""

import warnings
from numcodecs.compat import ensure_bytes
from zarr.util import normalize_storage_path
from zarr._storage.store import _get_metadata_suffix, data_root, meta_root, Store, StoreV3

__doctest_requires__ = {
    ('ABSStore', 'ABSStore.*'): ['azure.storage.blob'],
}


class ABSStore(Store):
    """Storage class using Azure Blob Storage (ABS).

    Parameters
    ----------
    container : string
        The name of the ABS container to use.

        .. deprecated::
           Use ``client`` instead.

    prefix : string
        Location of the "directory" to use as the root of the storage hierarchy
        within the container.

    account_name : string
        The Azure blob storage account name.

        .. deprecated:: 2.8.3
           Use ``client`` instead.

    account_key : string
        The Azure blob storage account access key.

        .. deprecated:: 2.8.3
           Use ``client`` instead.

    blob_service_kwargs : dictionary
        Extra arguments to be passed into the azure blob client, for e.g. when
        using the emulator, pass in blob_service_kwargs={'is_emulated': True}.

        .. deprecated:: 2.8.3
           Use ``client`` instead.

    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

    client : azure.storage.blob.ContainerClient, optional
        And ``azure.storage.blob.ContainerClient`` to connect with. See
        `here <https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.containerclient?view=azure-python>`_  # noqa
        for more.

        .. versionadded:: 2.8.3

    Notes
    -----
    In order to use this store, you must install the Microsoft Azure Storage SDK for Python,
    ``azure-storage-blob>=12.5.0``.
    """

    def __init__(self, container=None, prefix='', account_name=None, account_key=None,
                 blob_service_kwargs=None, dimension_separator=None,
                 client=None,
                 ):
        self._dimension_separator = dimension_separator
        self.prefix = normalize_storage_path(prefix)
        if client is None:
            # deprecated option, try to construct the client for them
            msg = (
                "Providing 'container', 'account_name', 'account_key', and 'blob_service_kwargs'"
                "is deprecated. Provide and instance of 'azure.storage.blob.ContainerClient' "
                "'client' instead."
            )
            warnings.warn(msg, FutureWarning, stacklevel=2)
            from azure.storage.blob import ContainerClient
            blob_service_kwargs = blob_service_kwargs or {}
            client = ContainerClient(
                "https://{}.blob.core.windows.net/".format(account_name), container,
                credential=account_key, **blob_service_kwargs
                )

        self.client = client
        self._container = container
        self._account_name = account_name
        self._account_key = account_key

    @staticmethod
    def _warn_deprecated(property_):
        msg = ("The {} property is deprecated and will be removed in a future "
               "version. Get the property from 'ABSStore.client' instead.")
        warnings.warn(msg.format(property_), FutureWarning, stacklevel=3)

    @property
    def container(self):
        self._warn_deprecated("container")
        return self._container

    @property
    def account_name(self):
        self._warn_deprecated("account_name")
        return self._account_name

    @property
    def account_key(self):
        self._warn_deprecated("account_key")
        return self._account_key

    def _append_path_to_prefix(self, path):
        if self.prefix == '':
            return normalize_storage_path(path)
        else:
            return '/'.join([self.prefix, normalize_storage_path(path)])

    @staticmethod
    def _strip_prefix_from_path(path, prefix):
        # normalized things will not have any leading or trailing slashes
        path_norm = normalize_storage_path(path)
        prefix_norm = normalize_storage_path(prefix)
        if prefix:
            return path_norm[(len(prefix_norm)+1):]
        else:
            return path_norm

    def __getitem__(self, key):
        from azure.core.exceptions import ResourceNotFoundError
        blob_name = self._append_path_to_prefix(key)
        try:
            return self.client.download_blob(blob_name).readall()
        except ResourceNotFoundError:
            raise KeyError('Blob %s not found' % blob_name)

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        blob_name = self._append_path_to_prefix(key)
        self.client.upload_blob(blob_name, value, overwrite=True)

    def __delitem__(self, key):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self.client.delete_blob(self._append_path_to_prefix(key))
        except ResourceNotFoundError:
            raise KeyError('Blob %s not found' % key)

    def __eq__(self, other):
        return (
            isinstance(other, ABSStore) and
            self.client == other.client and
            self.prefix == other.prefix
        )

    def keys(self):
        return list(self.__iter__())

    def __iter__(self):
        if self.prefix:
            list_blobs_prefix = self.prefix + '/'
        else:
            list_blobs_prefix = None
        for blob in self.client.list_blobs(list_blobs_prefix):
            yield self._strip_prefix_from_path(blob.name, self.prefix)

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        blob_name = self._append_path_to_prefix(key)
        return self.client.get_blob_client(blob_name).exists()

    def listdir(self, path=None):
        dir_path = normalize_storage_path(self._append_path_to_prefix(path))
        if dir_path:
            dir_path += '/'
        items = [
            self._strip_prefix_from_path(blob.name, dir_path)
            for blob in self.client.walk_blobs(name_starts_with=dir_path, delimiter='/')
        ]
        return items

    def rmdir(self, path=None):
        dir_path = normalize_storage_path(self._append_path_to_prefix(path))
        if dir_path:
            dir_path += '/'
        for blob in self.client.list_blobs(name_starts_with=dir_path):
            self.client.delete_blob(blob)

    def getsize(self, path=None):
        store_path = normalize_storage_path(path)
        fs_path = self._append_path_to_prefix(store_path)
        if fs_path:
            blob_client = self.client.get_blob_client(fs_path)
        else:
            blob_client = None

        if blob_client and blob_client.exists():
            return blob_client.get_blob_properties().size
        else:
            size = 0
            if fs_path == '':
                fs_path = None
            elif not fs_path.endswith('/'):
                fs_path += '/'
            for blob in self.client.walk_blobs(name_starts_with=fs_path, delimiter='/'):
                blob_client = self.client.get_blob_client(blob)
                if blob_client.exists():
                    size += blob_client.get_blob_properties().size
            return size

    def clear(self):
        self.rmdir()


class ABSStoreV3(ABSStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __eq__(self, other):
        return (
            isinstance(other, ABSStoreV3) and
            self.client == other.client and
            self.prefix == other.prefix
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def rmdir(self, path=None):

        if not path:
            # Currently allowing clear to delete everything as in v2

            # If we disallow an empty path then we will need to modify
            # TestABSStoreV3 to have the create_store method use a prefix.
            ABSStore.rmdir(self, '')
            return

        meta_dir = meta_root + path
        meta_dir = meta_dir.rstrip('/')
        ABSStore.rmdir(self, meta_dir)

        # remove data folder
        data_dir = data_root + path
        data_dir = data_dir.rstrip('/')
        ABSStore.rmdir(self, data_dir)

        # remove metadata files
        sfx = _get_metadata_suffix(self)
        array_meta_file = meta_dir + '.array' + sfx
        if array_meta_file in self:
            del self[array_meta_file]
        group_meta_file = meta_dir + '.group' + sfx
        if group_meta_file in self:
            del self[group_meta_file]

    # TODO: adapt the v2 getsize method to work for v3
    #       For now, calling the generic keys-based _getsize
    def getsize(self, path=None):
        from zarr.storage import _getsize  # avoid circular import
        return _getsize(self, path)


ABSStoreV3.__doc__ = ABSStore.__doc__
