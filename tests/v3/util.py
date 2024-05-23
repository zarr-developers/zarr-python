import collections
import os
import tempfile

import pytest

from zarr.store.memory import MemoryStore


class CountingDict(MemoryStore):
    def __init__(self):
        super().__init__()
        self.counter = collections.Counter()

    async def get(self, key, byte_range=None):
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__getitem__", key_suffix] += 1
        return await super().get(key, byte_range)

    async def set(self, key, value, byte_range=None):
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__setitem__", key_suffix] += 1
        return await super().set(key, value, byte_range)


def skip_test_env_var(name):
    """Checks for environment variables indicating whether tests requiring services should be run"""
    value = os.environ.get(name, "0")
    return pytest.mark.skipif(value == "0", reason="Tests not enabled via environment variable")


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


def mktemp(**kwargs):
    f = tempfile.NamedTemporaryFile(**kwargs)
    f.close()
    return f.name
