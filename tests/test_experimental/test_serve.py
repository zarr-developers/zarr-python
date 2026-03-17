from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.core.buffer import cpu
from zarr.core.sync import sync

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import ZarrFormat

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient

from zarr.experimental.serve import CorsOptions, _parse_range_header, node_app, store_app


@pytest.fixture
def group_with_arrays(store: Store) -> zarr.Group:
    """Create a group containing a regular array and a sharded array."""
    root = zarr.open_group(store, mode="w")
    zarr.create_array(root.store_path / "regular", shape=(4, 4), chunks=(2, 2), dtype="f8")
    zarr.create_array(
        root.store_path / "sharded",
        shape=(8, 8),
        chunks=(2, 2),
        shards=(4, 4),
        dtype="i4",
    )
    return root


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestNodeAppDoesNotExposeNonZarrKeys:
    """node_app must never expose keys that are not part of the zarr hierarchy."""

    def test_non_zarr_key_returns_404(self, store: Store, group_with_arrays: zarr.Group) -> None:
        """A key that is not valid zarr metadata or a valid chunk key should return 404,
        even if the underlying store contains data at that path."""
        non_zarr_buf = cpu.buffer_prototype.buffer.from_bytes(b"secret data")
        sync(store.set("secret.txt", non_zarr_buf))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        # The non-zarr key must not be accessible.
        response = client.get("/secret.txt")
        assert response.status_code == 404

    def test_non_zarr_key_nested_returns_404(
        self, store: Store, group_with_arrays: zarr.Group
    ) -> None:
        """A non-zarr key nested under a real array's path should return 404,
        even though the path prefix matches a valid zarr node."""
        non_zarr_buf = cpu.buffer_prototype.buffer.from_bytes(b"not a chunk")
        sync(store.set("regular/notes.txt", non_zarr_buf))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        response = client.get("/regular/notes.txt")
        assert response.status_code == 404

    def test_valid_metadata_is_accessible(self, group_with_arrays: zarr.Group) -> None:
        """Zarr metadata keys (zarr.json) for both the root group and child arrays
        should be served with a 200 status."""
        app = node_app(group_with_arrays)
        client = TestClient(app)

        # Root group metadata
        response = client.get("/zarr.json")
        assert response.status_code == 200

        # Array metadata
        response = client.get("/regular/zarr.json")
        assert response.status_code == 200

    def test_valid_chunk_is_accessible(self, group_with_arrays: zarr.Group) -> None:
        """A valid, in-bounds chunk key for an array with written data should
        be served with a 200 status."""
        arr = group_with_arrays["regular"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.ones((4, 4))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        # c/0/0 is a valid chunk key for a (4,4) array with (2,2) chunks.
        response = client.get("/regular/c/0/0")
        assert response.status_code == 200

    def test_out_of_bounds_chunk_key_returns_404(self, group_with_arrays: zarr.Group) -> None:
        """A chunk key that is syntactically valid but references indices beyond
        the array's chunk grid should return 404."""
        arr = group_with_arrays["regular"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.ones((4, 4))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        # (4,4) array with (2,2) chunks has grid shape (2,2), so c/99/99 is
        # syntactically valid but out of bounds.
        response = client.get("/regular/c/99/99")
        assert response.status_code == 404

    def test_empty_path_returns_404(self, group_with_arrays: zarr.Group) -> None:
        """A request to the root path '/' should return 404 because an empty
        string is not a valid zarr key."""
        app = node_app(group_with_arrays)
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestShardedArrayByteRangeReads:
    """Byte-range reads against a sharded array served via node_app."""

    def test_range_read_returns_206(self, group_with_arrays: zarr.Group) -> None:
        """A Range header requesting a specific byte range (e.g. bytes=0-7) should
        return 206 Partial Content with exactly those bytes."""
        arr = group_with_arrays["sharded"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        # c/0/0 is the first shard key for an (8,8) array with (4,4) shards.
        full_response = client.get("/sharded/c/0/0")
        assert full_response.status_code == 200
        full_body = full_response.content

        # Request the first 8 bytes.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=0-7"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[:8]

    def test_suffix_range_read(self, group_with_arrays: zarr.Group) -> None:
        """A suffix byte range (e.g. bytes=-4) should return the last N bytes
        of the resource with a 206 status."""
        arr = group_with_arrays["sharded"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        full_response = client.get("/sharded/c/0/0")
        full_body = full_response.content

        # Request the last 4 bytes.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=-4"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[-4:]

    def test_offset_range_read(self, group_with_arrays: zarr.Group) -> None:
        """An offset byte range (e.g. bytes=4-) should return all bytes from
        the given offset to the end, with a 206 status."""
        arr = group_with_arrays["sharded"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        full_response = client.get("/sharded/c/0/0")
        full_body = full_response.content

        # Request everything from byte 4 onward.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=4-"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[4:]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestMalformedRangeHeaders:
    """Malformed Range headers must return 416, never crash the server."""

    @pytest.mark.parametrize(
        "header",
        [
            "bytes=abc-def",
            "bytes=-abc",
            "bytes=abc-",
            "bytes=0-7,10-20",
        ],
    )
    def test_malformed_range_returns_416(self, store: Store, header: str) -> None:
        """Range headers with non-numeric values, multiple ranges, or other
        malformed syntax should return 416 Range Not Satisfiable instead of
        crashing the server."""
        buf = cpu.buffer_prototype.buffer.from_bytes(b"some data here")
        sync(store.set("key", buf))

        app = store_app(store)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/key", headers={"Range": header})
        assert response.status_code == 416

    def test_non_bytes_unit_returns_416(self, store: Store) -> None:
        """A Range header using a unit other than 'bytes' (e.g. 'chars=0-7')
        should return 416 because only byte ranges are supported."""
        buf = cpu.buffer_prototype.buffer.from_bytes(b"some data")
        sync(store.set("key", buf))

        app = store_app(store)
        client = TestClient(app)

        response = client.get("/key", headers={"Range": "chars=0-7"})
        assert response.status_code == 416


class TestParseRangeHeader:
    """Unit tests for _parse_range_header."""

    def test_valid_range(self) -> None:
        """'bytes=0-99' should parse into a RangeByteRequest with start=0 and
        end=100 (end is exclusive, so the inclusive HTTP end is incremented)."""
        from zarr.abc.store import RangeByteRequest

        result = _parse_range_header("bytes=0-99")
        assert result == RangeByteRequest(start=0, end=100)

    def test_valid_suffix(self) -> None:
        """'bytes=-50' should parse into a SuffixByteRequest requesting the
        last 50 bytes of the resource."""
        from zarr.abc.store import SuffixByteRequest

        result = _parse_range_header("bytes=-50")
        assert result == SuffixByteRequest(suffix=50)

    def test_valid_offset(self) -> None:
        """'bytes=10-' should parse into an OffsetByteRequest starting at
        byte 10 and reading to the end of the resource."""
        from zarr.abc.store import OffsetByteRequest

        result = _parse_range_header("bytes=10-")
        assert result == OffsetByteRequest(offset=10)

    def test_non_bytes_unit(self) -> None:
        """A Range header with a non-'bytes' unit should return None."""
        assert _parse_range_header("chars=0-7") is None

    def test_garbage_values(self) -> None:
        """Non-numeric values in the byte range should return None instead
        of raising a ValueError."""
        assert _parse_range_header("bytes=abc-def") is None

    def test_multi_range(self) -> None:
        """Multi-range requests (e.g. bytes=0-7,10-20) are not supported and
        should return None."""
        assert _parse_range_header("bytes=0-7,10-20") is None

    def test_empty_spec(self) -> None:
        """A Range header with no range specifier after 'bytes=' should
        return None."""
        assert _parse_range_header("bytes=") is None


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestWriteViaPut:
    """store_app and node_app can be configured to accept PUT writes."""

    def test_put_writes_to_store(self, store: Store) -> None:
        """A PUT request to store_app with PUT enabled should write the
        request body into the store at the given key."""
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        payload = b"hello zarr"
        response = client.put("/some/key", content=payload)
        assert response.status_code == 204

        # Verify the data landed in the store.
        buf = sync(store.get("some/key", cpu.buffer_prototype))
        assert buf is not None
        assert buf.to_bytes() == payload

    def test_put_then_get_roundtrip(self, store: Store) -> None:
        """Data written via PUT should be retrievable via a subsequent GET
        at the same key."""
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        payload = b"\x00\x01\x02\x03"
        client.put("/data/blob", content=payload)

        response = client.get("/data/blob")
        assert response.status_code == 200
        assert response.content == payload

    def test_put_rejected_when_not_configured(self, store: Store) -> None:
        """PUT requests should return 405 Method Not Allowed when the server
        is created with the default methods (GET only)."""
        app = store_app(store)
        client = TestClient(app)

        response = client.put("/some/key", content=b"data")
        assert response.status_code == 405

    def test_put_on_node_validates_key(self, store: Store, group_with_arrays: zarr.Group) -> None:
        """PUT requests via node_app should be rejected with 404 when the
        target key is not a valid zarr key (metadata or chunk)."""
        app = node_app(group_with_arrays, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.put("/not_a_zarr_key.bin", content=b"data")
        assert response.status_code == 404

    def test_put_to_valid_chunk_key_succeeds(self, group_with_arrays: zarr.Group) -> None:
        """PUT requests via node_app to a valid chunk key should succeed
        with 204, and the written data should be retrievable via GET."""
        app = node_app(group_with_arrays, methods={"GET", "PUT"})
        client = TestClient(app)

        payload = b"\x00" * 32
        response = client.put("/regular/c/0/0", content=payload)
        assert response.status_code == 204

        # Confirm it round-trips.
        get_response = client.get("/regular/c/0/0")
        assert get_response.status_code == 200
        assert get_response.content == payload


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestStoreAppEdgeCases:
    """Edge cases for store_app."""

    def test_get_nonexistent_key_returns_404(self, store: Store) -> None:
        """GET for a key that does not exist in the store should return 404."""
        app = store_app(store)
        client = TestClient(app)

        response = client.get("/no/such/key")
        assert response.status_code == 404

    def test_empty_path_returns_404(self, store: Store) -> None:
        """GET to the root path '/' (empty key) should return 404 because
        an empty string is not a valid store key."""
        app = store_app(store)
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestNodeAppDirectArray:
    """Serve a single array directly (not through a group)."""

    def test_serve_nested_array_directly(self, store: Store) -> None:
        """When node_app is given a nested array (not a group), requests
        should use keys relative to that array's path. Metadata and in-bounds
        chunks should return 200, and out-of-bounds chunks should return 404."""
        root = zarr.open_group(store, mode="w")
        arr = zarr.create_array(
            root.store_path / "sub/nested",
            shape=(4,),
            chunks=(2,),
            dtype="f8",
        )
        arr[:] = np.arange(4, dtype="f8")

        # Serve the array directly — its prefix is "sub/nested".
        app = node_app(arr)
        client = TestClient(app)

        # Metadata should be accessible at the array root.
        response = client.get("/zarr.json")
        assert response.status_code == 200

        # Chunk keys are relative to the array.
        response = client.get("/c/0")
        assert response.status_code == 200

        response = client.get("/c/1")
        assert response.status_code == 200

        # Out of bounds.
        response = client.get("/c/99")
        assert response.status_code == 404

    def test_serve_root_array(self, store: Store) -> None:
        """When node_app is given an array stored at the root of a store
        (empty prefix), metadata and chunk keys should be accessible at
        their natural paths."""
        arr = zarr.create_array(
            store,
            shape=(6,),
            chunks=(3,),
            dtype="i4",
        )
        arr[:] = np.arange(6, dtype="i4")

        # Root-level array has prefix = "".
        app = node_app(arr)
        client = TestClient(app)

        response = client.get("/zarr.json")
        assert response.status_code == 200

        response = client.get("/c/0")
        assert response.status_code == 200

        response = client.get("/c/1")
        assert response.status_code == 200

        response = client.get("/c/2")
        assert response.status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestContentType:
    """Responses should have the correct Content-Type."""

    def test_metadata_has_json_content_type(self, group_with_arrays: zarr.Group) -> None:
        """Zarr metadata files (zarr.json) should be served with
        Content-Type: application/json."""
        app = node_app(group_with_arrays)
        client = TestClient(app)

        response = client.get("/zarr.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_chunk_has_octet_stream_content_type(self, group_with_arrays: zarr.Group) -> None:
        """Chunk data should be served with Content-Type: application/octet-stream
        since it is binary data."""
        arr = group_with_arrays["regular"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.ones((4, 4))

        app = node_app(group_with_arrays)
        client = TestClient(app)

        response = client.get("/regular/c/0/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestCorsMiddleware:
    """CORS middleware should add the expected headers."""

    def test_cors_headers_present(self, store: Store) -> None:
        """When cors_options are provided, responses should include the
        Access-Control-Allow-Origin header matching the request origin."""
        buf = cpu.buffer_prototype.buffer.from_bytes(b"data")
        sync(store.set("key", buf))

        cors = CorsOptions(allow_origins=["https://example.com"], allow_methods=["GET"])
        app = store_app(store, cors_options=cors)
        client = TestClient(app)

        response = client.get("/key", headers={"Origin": "https://example.com"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://example.com"

    def test_cors_preflight(self, store: Store) -> None:
        """CORS preflight OPTIONS requests should return 200 with the
        Access-Control-Allow-Origin header when CORS is configured."""
        cors = CorsOptions(allow_origins=["*"], allow_methods=["GET", "PUT"])
        app = store_app(store, methods={"GET", "PUT"}, cors_options=cors)
        client = TestClient(app)

        response = client.options(
            "/any/path",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "PUT",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_no_cors_headers_without_option(self, store: Store) -> None:
        """When no cors_options are provided, responses should not include
        any CORS headers, even if the request includes an Origin header."""
        buf = cpu.buffer_prototype.buffer.from_bytes(b"data")
        sync(store.set("key", buf))

        app = store_app(store)
        client = TestClient(app)

        response = client.get("/key", headers={"Origin": "https://example.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


def _metadata_key(zarr_format: ZarrFormat) -> str:
    """Return the metadata key for the given zarr format."""
    return "zarr.json" if zarr_format == 3 else ".zarray"


def _chunk_key(zarr_format: ZarrFormat, coords: str) -> str:
    """Return a chunk key for the given format.

    *coords* is a dot-separated string like ``"0.0"``.  For v3 this becomes
    ``"c/0/0"``; for v2 it is returned unchanged.
    """
    if zarr_format == 3:
        return "c/" + coords.replace(".", "/")
    return coords


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestNodeAppV2AndV3:
    """Test node_app with both v2 and v3 arrays side by side."""

    def test_metadata_accessible(self, store: Store, zarr_format: ZarrFormat) -> None:
        """The format-appropriate metadata key should be served with 200."""
        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8", zarr_format=zarr_format)
        app = node_app(arr)
        client = TestClient(app)

        response = client.get(f"/{_metadata_key(zarr_format)}")
        assert response.status_code == 200

    def test_chunk_accessible(self, store: Store, zarr_format: ZarrFormat) -> None:
        """An in-bounds chunk key should be served with 200 for both formats."""
        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8", zarr_format=zarr_format)
        arr[:] = np.ones(4)

        app = node_app(arr)
        client = TestClient(app)

        response = client.get(f"/{_chunk_key(zarr_format, '0')}")
        assert response.status_code == 200

    def test_out_of_bounds_chunk_returns_404(self, store: Store, zarr_format: ZarrFormat) -> None:
        """An out-of-bounds chunk key should return 404 for both formats."""
        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8", zarr_format=zarr_format)
        arr[:] = np.ones(4)

        app = node_app(arr)
        client = TestClient(app)

        response = client.get(f"/{_chunk_key(zarr_format, '99')}")
        assert response.status_code == 404

    def test_non_zarr_key_returns_404(self, store: Store, zarr_format: ZarrFormat) -> None:
        """A non-zarr key should return 404 regardless of format."""
        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8", zarr_format=zarr_format)
        non_zarr_buf = cpu.buffer_prototype.buffer.from_bytes(b"secret")
        sync(store.set("secret.txt", non_zarr_buf))

        app = node_app(arr)
        client = TestClient(app)

        response = client.get("/secret.txt")
        assert response.status_code == 404

    def test_data_roundtrip(self, store: Store, zarr_format: ZarrFormat) -> None:
        """Data written to an array should be readable via store_app for
        both formats."""
        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8", zarr_format=zarr_format)
        arr[:] = np.arange(4, dtype="f8")

        app = store_app(store)
        client = TestClient(app)

        # Metadata should be accessible.
        response = client.get(f"/{_metadata_key(zarr_format)}")
        assert response.status_code == 200

        # First chunk should be accessible.
        response = client.get(f"/{_chunk_key(zarr_format, '0')}")
        assert response.status_code == 200
        assert len(response.content) > 0


def _get_free_port() -> int:
    """Return an unused TCP port on localhost."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestServeBackground:
    """Test serve_store and serve_node with background=True."""

    def test_serve_store_background(self, store: Store) -> None:
        """serve_store(background=True) should return a BackgroundServer
        that responds to HTTP requests and can be used as a context manager."""
        import httpx

        from zarr.experimental.serve import serve_store

        buf = cpu.buffer_prototype.buffer.from_bytes(b"hello")
        sync(store.set("key", buf))

        port = _get_free_port()
        with serve_store(store, host="127.0.0.1", port=port, background=True) as server:
            assert server.host == "127.0.0.1"
            assert server.port == port
            assert server.url == f"http://127.0.0.1:{port}"

            response = httpx.get(f"{server.url}/key")
            assert response.status_code == 200
            assert response.content == b"hello"

    def test_serve_node_background(self, store: Store) -> None:
        """serve_node(background=True) should return a BackgroundServer
        that responds to HTTP requests and can be used as a context manager."""
        import httpx

        from zarr.experimental.serve import serve_node

        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8")
        arr[:] = np.arange(4, dtype="f8")

        port = _get_free_port()
        with serve_node(arr, host="127.0.0.1", port=port, background=True) as server:
            response = httpx.get(f"{server.url}/zarr.json")
            assert response.status_code == 200
