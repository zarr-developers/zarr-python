import pathlib
import pytest
import os

from zarr.v3.store import LocalStore, StorePath, MemoryStore
from zarr.v3.store.remote import RemoteStore


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request):
    return request.param


@pytest.fixture()
def mock_s3(request):
    # writable local S3 system
    import shlex
    import subprocess
    import time

    if "BOTO_CONFIG" not in os.environ:  # pragma: no cover
        os.environ["BOTO_CONFIG"] = "/dev/null"
    if "AWS_ACCESS_KEY_ID" not in os.environ:  # pragma: no cover
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECRET_ACCESS_KEY"] = "bar"
    requests = pytest.importorskip("requests")
    s3fs = pytest.importorskip("s3fs")
    pytest.importorskip("moto")
    port = 5555
    endpoint_uri = "http://127.0.0.1:%d/" % port
    proc = subprocess.Popen(
        shlex.split("moto_server s3 -p %d" % port),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(endpoint_uri)
            if r.ok:
                break
        except Exception:  # pragma: no cover
            pass
        timeout -= 0.1  # pragma: no cover
        time.sleep(0.1)  # pragma: no cover
    s3so = dict(client_kwargs={"endpoint_url": endpoint_uri}, use_listings_cache=False)
    s3 = s3fs.S3FileSystem(anon=False, **s3so)
    s3.mkdir("test")
    request.cls.s3so = s3so
    yield
    proc.terminate()
    proc.wait()


# todo: harmonize this with local_store fixture
@pytest.fixture
def store_path(tmpdir):
    store = LocalStore(str(tmpdir))
    p = StorePath(store)
    return p


@pytest.fixture(scope="function")
def local_store(tmpdir):
    return LocalStore(str(tmpdir))


@pytest.fixture(scope="function")
def remote_store():
    return RemoteStore()


@pytest.fixture(scope="function")
def memory_store():
    return MemoryStore()
