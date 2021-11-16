import collections
import os

from zarr.storage import Store, StoreV3

import pytest


class CountingDict(Store):

    def __init__(self):
        self.wrapped = dict()
        self.counter = collections.Counter()

    def __len__(self):
        self.counter['__len__'] += 1
        return len(self.wrapped)

    def keys(self):
        self.counter['keys'] += 1
        return self.wrapped.keys()

    def __iter__(self):
        self.counter['__iter__'] += 1
        return iter(self.wrapped)

    def __contains__(self, item):
        self.counter['__contains__', item] += 1
        return item in self.wrapped

    def __getitem__(self, item):
        self.counter['__getitem__', item] += 1
        return self.wrapped[item]

    def __setitem__(self, key, value):
        self.counter['__setitem__', key] += 1
        self.wrapped[key] = value

    def __delitem__(self, key):
        self.counter['__delitem__', key] += 1
        del self.wrapped[key]


class CountingDictV3(CountingDict, StoreV3):
    pass


def skip_test_env_var(name):
    """ Checks for environment variables indicating whether tests requiring services should be run
    """
    value = os.environ.get(name, '0')
    return pytest.mark.skipif(value == '0', reason='Tests not enabled via environment variable')


try:
    import fsspec  # noqa: F401

    have_fsspec = True
except ImportError:  # pragma: no cover
    have_fsspec = False


def abs_container():
    from azure.core.exceptions import ResourceExistsError
    import azure.storage.blob as asb

    URL = "http://127.0.0.1:10000"
    ACCOUNT_NAME = "devstoreaccount1"
    KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
    CONN_STR = (
        f"DefaultEndpointsProtocol=http;AccountName={ACCOUNT_NAME};"
        f"AccountKey={KEY};BlobEndpoint={URL}/{ACCOUNT_NAME};"
    )

    blob_service_client = asb.BlobServiceClient.from_connection_string(CONN_STR)
    try:
        container_client = blob_service_client.create_container("test")
    except ResourceExistsError:
        container_client = blob_service_client.get_container_client("test")

    return container_client
