from __future__ import annotations

import asyncio
import errno
import os
import socket
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pytest
import zarr
from starlette.testclient import TestClient
from zarr.buffer import cpu
from zarr.storage import LocalStore, MemoryStore

from zarr_http_server._serve import (
    READ_ONLY_HTTP_METHODS,
    READ_WRITE_HTTP_METHODS,
    CorsOptions,
    ReadOnlyHTTPMethod,
    _parse_range_header,
    _RangeVerdict,
    node_app,
    store_app,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Coroutine, Iterator

    from zarr.abc.store import Store

ZarrFormat = Literal[2, 3]


def sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a store coroutine to completion (tests use MemoryStore only)."""
    return asyncio.run(coro)


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

    def test_out_of_bounds_chunk_key_returns_404(
        self, store: Store, group_with_arrays: zarr.Group
    ) -> None:
        """A chunk key that is syntactically valid but references indices beyond
        the array's chunk grid should return 404."""
        arr = group_with_arrays["regular"]
        assert isinstance(arr, zarr.Array)
        arr[:] = np.ones((4, 4))

        # Put real data at the out-of-grid key, so that a 404 can only come
        # from the bounds check -- not from the key merely being absent.
        planted = cpu.buffer_prototype.buffer.from_bytes(b"out of grid")
        sync(store.set("regular/c/99/99", planted))
        assert sync(store.get("regular/c/99/99", cpu.buffer_prototype)) is not None

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
class TestUnusableRangeHeadersAreIgnored:
    """A Range this server cannot turn into a read is ignored, not refused.

    RFC 9110 §14.2 requires ignoring a Range whose unit is unrecognized and
    permits ignoring one that will not parse; either way the answer is 200
    with the full representation. Refusing with 416 would tell a client the
    object is unreadable when only the request shape was unsupported -- and
    a multi-range request, which is legal to send and which proxies and
    download accelerators do send, would take that at face value.
    """

    @pytest.mark.parametrize(
        "header",
        [
            "bytes=abc-def",
            "bytes=-abc",
            "bytes=abc-",
            "bytes=0-7,10-20",
            "chars=0-7",
            "bytes=",
            "bytes=+0-1",
        ],
    )
    def test_unusable_range_serves_full_representation(self, store: Store, header: str) -> None:
        """Non-numeric bounds, multi-range, a non-'bytes' unit, an empty spec
        and a non-canonical byte position all fall back to a plain 200."""
        body = b"some data here"
        sync(store.set("key", cpu.buffer_prototype.buffer.from_bytes(body)))

        client = TestClient(store_app(store), raise_server_exceptions=False)

        response = client.get("/key", headers={"Range": header})
        assert response.status_code == 200
        assert response.content == body
        assert "content-range" not in response.headers


class TestParseRangeHeader:
    """Unit tests for _parse_range_header."""

    def test_parser_rejects_an_inverted_range(self) -> None:
        """Pin the parser itself: on a MemoryStore an unguarded inverted range
        happens to return b"" and still yields 416, so the status code alone
        cannot tell whether the guard is present."""
        assert _parse_range_header("bytes=5-2") is _RangeVerdict.UNSATISFIABLE
        assert _parse_range_header("bytes=0-0") is not _RangeVerdict.UNSATISFIABLE

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
        """An unrecognized range unit must be ignored, per RFC 9110 §14.2."""
        assert _parse_range_header("chars=0-7") is _RangeVerdict.IGNORE

    def test_garbage_values(self) -> None:
        """Non-numeric bounds are ignored rather than raising a ValueError."""
        assert _parse_range_header("bytes=abc-def") is _RangeVerdict.IGNORE

    def test_multi_range(self) -> None:
        """Multi-range requests (e.g. bytes=0-7,10-20) are legal to send; this
        server does not build multipart/byteranges, so it serves the whole
        representation instead of refusing."""
        assert _parse_range_header("bytes=0-7,10-20") is _RangeVerdict.IGNORE

    def test_empty_spec(self) -> None:
        """A Range header with no range specifier after 'bytes=' is ignored."""
        assert _parse_range_header("bytes=") is _RangeVerdict.IGNORE

    def test_non_canonical_byte_position(self) -> None:
        """`int` would accept these; RFC 9110 defines a byte position as
        1*DIGIT, so they are not ranges and the header is ignored."""
        assert _parse_range_header("bytes=+0-1") is _RangeVerdict.IGNORE
        assert _parse_range_header("bytes= 0-1") is _RangeVerdict.IGNORE
        assert _parse_range_header("bytes=0_0-1") is _RangeVerdict.IGNORE

    def test_oversized_start_is_unsatisfiable(self) -> None:
        """A first-byte-pos past any possible object names nothing."""
        assert _parse_range_header("bytes=99999999999999999999-") is _RangeVerdict.UNSATISFIABLE

    def test_zero_length_suffix_is_unsatisfiable(self) -> None:
        """ "The last zero bytes" names nothing."""
        assert _parse_range_header("bytes=-0") is _RangeVerdict.UNSATISFIABLE


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
class TestMethodValidation:
    """Only the methods the handler implements may be served."""

    def test_supported_methods_are_served(self, store: Store) -> None:
        """GET, PUT and HEAD each behave as their verb implies."""
        app = store_app(store, methods={"GET", "PUT", "HEAD"})
        client = TestClient(app)

        assert client.put("/zarr.json", content=b'{"a":1}').status_code == 204
        assert client.get("/zarr.json").content == b'{"a":1}'

        # HEAD reports the same status as GET but carries no body.
        head = client.head("/zarr.json")
        assert head.status_code == 200
        assert head.content == b""

    def test_empty_method_set_raises(self, store: Store) -> None:
        """Asking for no methods must be rejected rather than producing a
        server that answers every verb, including writes: Starlette treats a
        falsy `methods` on a Route as "match anything"."""
        for build in (
            lambda: store_app(store, methods=set()),
            lambda: node_app(zarr.open_group(store, mode="a"), methods=set()),
        ):
            with pytest.raises(ValueError, match="at least one"):
                build()

    @pytest.mark.parametrize("method", ["DELETE", "POST", "PATCH", "OPTIONS", "TRACE"])
    def test_unsupported_method_raises(self, store: Store, method: str) -> None:
        """A verb the handler cannot implement is rejected when the app is
        built, rather than silently answering as if it were a GET."""
        for build in (
            lambda: store_app(store, methods={"GET", method}),  # type: ignore[arg-type]
            lambda: node_app(zarr.open_group(store, mode="a"), methods={"GET", method}),  # type: ignore[arg-type]
        ):
            with pytest.raises(ValueError, match=method):
                build()


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

        # Plant data at the out-of-grid key so the 404 must come from the
        # bounds check rather than from the key being absent.
        key = _chunk_key(zarr_format, "99")
        sync(store.set(key, cpu.buffer_prototype.buffer.from_bytes(b"out of grid")))
        assert sync(store.get(key, cpu.buffer_prototype)) is not None

        app = node_app(arr)
        client = TestClient(app)

        response = client.get(f"/{key}")
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


class TestPathTraversalProtection:
    """store_app and node_app must reject path-traversal attempts before
    touching the store, regardless of URL-encoding tricks."""

    def test_get_traversal_outside_store_root_returns_404(self, tmp_path: Any) -> None:
        """A GET for a percent-encoded '../secret.txt' must not escape the
        store root and read a file outside it."""
        from zarr.storage import LocalStore

        root = tmp_path / "store_root"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret contents")

        store = LocalStore(root)
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.get("/..%2fsecret.txt")
        assert response.status_code == 404
        assert b"top secret" not in response.content

    def test_put_traversal_outside_store_root_returns_404(self, tmp_path: Any) -> None:
        """A PUT to a percent-encoded '../pwned.txt' must not escape the
        store root and write a file outside it."""
        from zarr.storage import LocalStore

        root = tmp_path / "store_root"
        root.mkdir()
        pwned = tmp_path / "pwned.txt"

        store = LocalStore(root)
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.put("/..%2fpwned.txt", content=b"pwned")
        assert response.status_code == 404
        assert not pwned.exists()

    @pytest.mark.parametrize(
        ("encoded_path", "climb_depth"),
        [
            ("/..%2fsecret.txt", 1),
            ("/%2e%2e/secret.txt", 1),
            ("/..%2f..%2fsecret.txt", 2),
        ],
    )
    def test_encoded_traversal_variants_return_404(
        self, tmp_path: Any, encoded_path: str, climb_depth: int
    ) -> None:
        """Various percent-encoded traversal spellings must all be rejected.

        The store root is nested exactly ``climb_depth`` directories below
        ``tmp_path`` and the secret lives at ``tmp_path/secret.txt`` -- the
        exact location each traversal's ".." segments resolve to -- so a
        case with a missing or deleted guard would actually reach the
        secret instead of just returning 404 for an unrelated reason (e.g.
        a two-level climb landing on a directory that happens to be empty).
        """
        from zarr.storage import LocalStore

        parts = [f"level{i}" for i in range(climb_depth - 1)] + ["store_root"]
        root = tmp_path.joinpath(*parts)
        root.mkdir(parents=True)
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret contents")

        store = LocalStore(root)
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.get(encoded_path)
        assert response.status_code == 404
        assert b"top secret" not in response.content

    def test_get_absolute_key_bypass_returns_404(self, tmp_path: Any) -> None:
        """A percent-encoded leading slash decodes to an ABSOLUTE path param
        (e.g. request '/%2fetc%2fhostname' -> path param '/etc/hostname').
        '/etc/hostname'.split('/') -> ['', 'etc', 'hostname'] has no '.' or
        '..' segment, so the two-element guard misses it, but LocalStore
        resolves an absolute key by discarding its root entirely -- an
        arbitrary-file read. The empty leading segment must be rejected."""
        from zarr.storage import LocalStore

        root = tmp_path / "store_root"
        root.mkdir()
        secret = tmp_path / "secret_abs.txt"
        secret.write_text("top secret absolute contents")

        store = LocalStore(root)
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        # Mirror the exploit: percent-encode every "/" (including the
        # leading one) in the absolute secret path as "%2f".
        encoded_path = "/" + str(secret).replace("/", "%2f")

        response = client.get(encoded_path)
        assert response.status_code == 404
        assert b"top secret absolute" not in response.content
        assert secret.read_text() == "top secret absolute contents"

    def test_put_absolute_key_bypass_returns_404(self, tmp_path: Any) -> None:
        """Same absolute-key vector as above, but for PUT: a percent-encoded
        leading slash must not allow writing a file outside the store root."""
        from zarr.storage import LocalStore

        root = tmp_path / "store_root"
        root.mkdir()
        pwned = tmp_path / "pwned_abs.txt"

        store = LocalStore(root)
        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        encoded_path = "/" + str(pwned).replace("/", "%2f")

        response = client.put(encoded_path, content=b"pwned")
        assert response.status_code == 404
        assert not pwned.exists()

    @pytest.mark.parametrize(
        "encoded_path",
        [
            "/..%5C..%5Cwin.ini",
            "/%5CWindows%5Cwin.ini",
            "/C:/Windows/win.ini",
            "/C:%5CWindows",
            "/%5C%5Chost%5Cshare%5Cx",
        ],
    )
    def test_backslash_and_drive_traversal_variants_return_404(self, encoded_path: str) -> None:
        """Backslash is a path separator on Windows, and a drive-qualified or
        root-relative key discards a filesystem store's root entirely on
        Windows, even though POSIX only ever treats '/' as a separator. The
        guard must reject these purely from the string, before the store is
        ever touched -- verified here by making the store raise if called."""
        from unittest.mock import AsyncMock

        from zarr.storage import MemoryStore

        store = MemoryStore()
        store.get = AsyncMock(side_effect=AssertionError("store.get should not be called"))  # type: ignore[method-assign]
        store.set = AsyncMock(side_effect=AssertionError("store.set should not be called"))  # type: ignore[method-assign]

        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.get(encoded_path)
        assert response.status_code == 404

    def test_put_backslash_traversal_returns_404(self) -> None:
        """A PUT to a backslash-encoded '..\\..\\pwned.txt' must be rejected
        before the store is touched, mirroring the GET case above."""
        from unittest.mock import AsyncMock

        from zarr.storage import MemoryStore

        store = MemoryStore()
        store.set = AsyncMock(side_effect=AssertionError("store.set should not be called"))  # type: ignore[method-assign]

        app = store_app(store, methods={"GET", "PUT"})
        client = TestClient(app)

        response = client.put("/..%5C..%5Cpwned.txt", content=b"pwned")
        assert response.status_code == 404


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

        from zarr_http_server import serve_store

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

        from zarr_http_server import serve_node

        arr = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="f8")
        arr[:] = np.arange(4, dtype="f8")

        port = _get_free_port()
        with serve_node(arr, host="127.0.0.1", port=port, background=True) as server:
            response = httpx.get(f"{server.url}/zarr.json")
            assert response.status_code == 200


class TestStoreFailuresAreNotReportedAsMisses:
    """Under the v3 spec an absent chunk is an uninitialized one, and a reader
    is right to substitute the array's fill value for it. A 404 therefore
    asserts something about the store's contents, and an I/O failure must not
    borrow it -- that would have a correct client materialize fill values over
    data that exists."""

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses the permission bits under test")
    def test_unreadable_key_is_a_server_error(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "key").write_bytes(b"real data")
        os.chmod(root / "key", 0o000)

        client = TestClient(store_app(LocalStore(str(root))), raise_server_exceptions=False)
        try:
            assert client.get("/key").status_code >= 500
        finally:
            os.chmod(root / "key", 0o600)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses the permission bits under test")
    def test_unwritable_store_is_a_server_error(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "ro"
        root.mkdir()
        os.chmod(root, 0o500)

        client = TestClient(
            store_app(LocalStore(str(root)), methods={"GET", "PUT"}),
            raise_server_exceptions=False,
        )
        try:
            assert client.put("/key", content=b"data").status_code >= 500
        finally:
            os.chmod(root, 0o700)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestShardGridBounds:
    """A sharded array's storage grid is its shard grid, not its chunk grid."""

    def test_out_of_shard_grid_key_returns_404(self, store: Store) -> None:
        arr = zarr.create_array(store, shape=(8, 8), chunks=(2, 2), shards=(4, 4), dtype="i4")
        arr[:] = np.arange(64, dtype="i4").reshape(8, 8)

        # (8,8) with (4,4) shards has a 2x2 shard grid, so c/3/3 is out of it.
        # Plant data there so the 404 must come from the bounds check.
        sync(store.set("c/3/3", cpu.buffer_prototype.buffer.from_bytes(b"PLANTED")))
        assert sync(store.get("c/3/3", cpu.buffer_prototype)) is not None

        client = TestClient(node_app(arr))
        assert client.get("/c/0/0").status_code == 200
        assert client.get("/c/3/3").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestWrongArityChunkKeys:
    """A chunk key with the wrong number of coordinates is invalid, and must
    not reach the grid comparison -- zip(strict=True) would raise there."""

    @pytest.mark.parametrize("key", ["c/0", "c/0/0/0", "c/0/0/0/0"])
    def test_wrong_arity_returns_404(self, store: Store, key: str) -> None:
        arr = zarr.create_array(store, shape=(4, 4), chunks=(2, 2), dtype="f8")
        arr[:] = np.ones((4, 4))

        sync(store.set(key, cpu.buffer_prototype.buffer.from_bytes(b"PLANTED")))

        client = TestClient(node_app(arr), raise_server_exceptions=False)
        assert client.get(f"/{key}").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestNodeNamesContainingAColon:
    """A leading `<x>:` is a drive reference to `ntpath`, so the guard rejects
    it on every platform to keep it a pure string gate in front of any store.
    The documented cost is that such a name is unreachable in the first
    segment -- but only there."""

    def test_colon_named_node_in_a_later_segment_is_served(self, store: Store) -> None:
        root = zarr.open_group(store, mode="w")
        sub = root.create_group("sub")
        sub.create_array("a:b", shape=(2,), chunks=(2,), dtype="f8")

        client = TestClient(node_app(root))
        assert client.get("/sub/a:b/zarr.json").status_code == 200

    def test_colon_named_node_in_the_first_segment_is_rejected(self, store: Store) -> None:
        root = zarr.open_group(store, mode="w")
        root.create_array("a:b", shape=(2,), chunks=(2,), dtype="f8")

        client = TestClient(node_app(root))
        assert client.get("/a:b/zarr.json").status_code == 404


class TestChunkedBodyIsCapped:
    """A chunked request carries no Content-Length, so the cap has to hold
    while the body is being read rather than after it is buffered."""

    def test_chunked_body_over_cap_is_rejected(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(str(tmp_path / "root"))
        client = TestClient(
            store_app(store, methods={"GET", "PUT"}, max_body_size=64),
            raise_server_exceptions=False,
        )

        def body() -> Iterator[bytes]:
            for _ in range(20):
                yield b"x" * 32

        # httpx sends an iterator body with Transfer-Encoding: chunked.
        assert client.put("/key", content=body()).status_code == 413
        assert not (tmp_path / "root" / "key").exists()


class TestHostileKeysAreNotServerErrors:
    """A key the store cannot express is a miss, not a server fault: the
    server must never answer 5xx for input a client can choose freely.

    This needs a filesystem-backed store -- a `MemoryStore` accepts any key
    as a dict key, so only `LocalStore` surfaces the underlying errors (an
    embedded NUL, a name longer than the filesystem allows).
    """

    @pytest.mark.parametrize(
        "path", ["/x%00y", "/ok.txt%00", "/" + "a" * 3000, "/" + "b" * 3000 + "/zarr.json"]
    )
    def test_unexpressable_key_returns_404_not_500(self, tmp_path: pathlib.Path, path: str) -> None:
        store = LocalStore(str(tmp_path / "root"))
        client = TestClient(store_app(store, methods={"GET", "PUT"}))

        assert client.get(path).status_code == 404
        assert client.put(path, content=b"x").status_code == 404


class TestGenericStoreFailuresAreNotMisses:
    """Only an error that answers about the *name* may become a 404.

    `ENAMETOOLONG` says no such name is expressible, which is an answer about
    the key. `EINVAL` is POSIX's catch-all and is reachable on a perfectly
    ordinary short key -- a bad seek, an unsupported filesystem feature -- so
    reporting it as absence would have a v3 reader write fill values over a
    chunk that exists but could not be read.
    """

    @staticmethod
    def _store_failing_with(code: int) -> Store:
        class Failing(MemoryStore):
            async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
                raise OSError(code, os.strerror(code))

            async def set(self, key: str, value: Any) -> None:
                raise OSError(code, os.strerror(code))

        return Failing()

    def test_einval_is_not_reported_as_absent(self) -> None:
        """The regression this class exists for."""
        client = TestClient(
            store_app(self._store_failing_with(errno.EINVAL), methods={"GET", "PUT"}),
            raise_server_exceptions=False,
        )

        assert client.get("/c/0/0").status_code >= 500
        assert client.put("/c/0/0", content=b"data").status_code >= 500

    def test_enametoolong_is_still_a_miss(self) -> None:
        """A name the store cannot express holds nothing, so 404 is honest."""
        client = TestClient(
            store_app(self._store_failing_with(errno.ENAMETOOLONG), methods={"GET", "PUT"}),
            raise_server_exceptions=False,
        )

        assert client.get("/c/0/0").status_code == 404
        assert client.put("/c/0/0", content=b"data").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestPutBodyLimit:
    """`Store.set` takes a whole buffer, so an unbounded body would let one
    request size the server's memory use."""

    def test_body_over_the_limit_is_rejected(self, store: Store) -> None:
        client = TestClient(store_app(store, methods={"GET", "PUT"}, max_body_size=64))

        assert client.put("/key", content=b"x" * 65).status_code == 413
        # Nothing was written.
        assert sync(store.get("key", cpu.buffer_prototype)) is None
        # A body within the limit still succeeds.
        assert client.put("/key", content=b"x" * 64).status_code == 204

    def test_limit_can_be_lifted(self, store: Store) -> None:
        client = TestClient(store_app(store, methods={"GET", "PUT"}, max_body_size=None))
        assert client.put("/key", content=b"x" * 5000).status_code == 204


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestRangeResponseCorrectness:
    """A 206 must describe which bytes it carries, and a range that cannot be
    satisfied must say so rather than returning an empty 206."""

    def test_206_carries_content_range(self, store: Store) -> None:
        sync(store.set("key", cpu.buffer_prototype.buffer.from_bytes(b"0123456789")))
        client = TestClient(store_app(store))

        response = client.get("/key", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["Content-Range"] == "bytes 2-5/*"

    def test_range_beyond_end_is_416(self, store: Store) -> None:
        sync(store.set("key", cpu.buffer_prototype.buffer.from_bytes(b"0123456789")))
        client = TestClient(store_app(store))

        assert client.get("/key", headers={"Range": "bytes=1000-2000"}).status_code == 416

    def test_inverted_range_is_416(self, store: Store) -> None:
        sync(store.set("key", cpu.buffer_prototype.buffer.from_bytes(b"0123456789")))
        client = TestClient(store_app(store))

        assert client.get("/key", headers={"Range": "bytes=5-2"}).status_code == 416


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestZeroDimensionalArray:
    """A 0-d array has exactly one chunk, spelled `0` in v2 and `c` in v3."""

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_sole_chunk_is_served(self, store: Store, zarr_format: ZarrFormat) -> None:
        arr = zarr.create_array(store, shape=(), dtype="i4", zarr_format=zarr_format)
        arr[...] = 7

        client = TestClient(node_app(arr))
        chunk_key = "0" if zarr_format == 2 else "c"

        assert client.get(f"/{chunk_key}").status_code == 200
        # A 1-d coordinate is not valid for a 0-d grid.
        assert client.get("/0/0").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestGroupAndArrayMetadataKeysAreDistinct:
    """A v2 group owns `.zgroup`; `.zarray` belongs to arrays, and vice versa."""

    def test_node_does_not_claim_the_other_kind_of_metadata(self, store: Store) -> None:
        root = zarr.open_group(store, mode="w", zarr_format=2)
        arr = root.create_array("a", shape=(2,), chunks=(2,), dtype="f8")

        # Plant both documents so a 404 reflects the key set, not absence.
        sync(store.set(".zarray", cpu.buffer_prototype.buffer.from_bytes(b"{}")))
        sync(store.set("a/.zgroup", cpu.buffer_prototype.buffer.from_bytes(b"{}")))

        assert TestClient(node_app(root)).get("/.zarray").status_code == 404
        assert TestClient(node_app(arr)).get("/.zgroup").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestBackgroundServerReportsBoundPort:
    """`port=0` asks the OS for a free port, so the server must report the
    port it actually bound rather than the zero it was asked for."""

    def test_port_zero_reports_the_bound_port(self, store: Store) -> None:
        import httpx

        from zarr_http_server import serve_store

        sync(store.set("key", cpu.buffer_prototype.buffer.from_bytes(b"hello")))

        with serve_store(store, host="127.0.0.1", port=0, background=True) as server:
            assert server.port != 0
            assert server.url == f"http://127.0.0.1:{server.port}"
            # The reported URL is the one that actually serves the data.
            assert httpx.get(f"{server.url}/key").content == b"hello"


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestUnopenableChildIsAnError:
    """A child that cannot be *judged* must not be reported as absent.

    404 is a claim about the store's contents, and under the v3 spec an
    absent chunk is an uninitialized one -- so a correct reader answers 404
    by silently substituting the array's fill value. Returning it for a child
    whose metadata could not be read would materialize zeros over data that
    exists. Only a genuinely missing member is a 404; corrupt metadata, an
    I/O error, or a codec this process lacks all surface as 5xx.
    """

    def test_child_with_unparseable_metadata_is_not_reported_as_missing(self, store: Store) -> None:
        """A corrupt child metadata document must not yield 404."""
        root = zarr.open_group(store, mode="w")
        root.create_array("good", shape=(2,), chunks=(2,), dtype="f8")
        sync(store.set("junk/zarr.json", cpu.buffer_prototype.buffer.from_bytes(b"not json")))

        client = TestClient(node_app(root), raise_server_exceptions=False)

        assert client.get("/good/zarr.json").status_code == 200
        assert client.get("/junk/zarr.json").status_code >= 500
        assert client.get("/junk/c/0").status_code >= 500

    def test_absent_child_is_reported_as_missing(self, store: Store) -> None:
        """A member that simply is not there is still a plain 404."""
        root = zarr.open_group(store, mode="w")
        root.create_array("good", shape=(2,), chunks=(2,), dtype="f8")

        client = TestClient(node_app(root), raise_server_exceptions=False)

        assert client.get("/nope/zarr.json").status_code == 404
        assert client.get("/junk/c/0").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestReadBackWithZarrClient:
    """The round-trip the README leads with: serve an array, then open it
    with a zarr client over HTTP.

    This needs an HTTP-capable fsspec, which is a client-side concern that
    `zarr-http-server` deliberately does not depend on -- it lives in the `docs`
    dependency group, alongside the other deps the README examples need.
    """

    def test_served_array_reads_back_identically(self, store: Store) -> None:
        """`zarr.open_array(server.url)` should return the same data that was
        served, so the README's headline example stays true."""
        pytest.importorskip("fsspec")
        pytest.importorskip("aiohttp")

        from zarr_http_server import serve_node

        expected = np.arange(100, dtype="uint8").reshape(10, 10)
        arr = zarr.create_array(store, data=expected, chunks=(5, 5), write_data=True)

        port = _get_free_port()
        with serve_node(arr, host="127.0.0.1", port=port, background=True) as server:
            remote = zarr.open_array(server.url, mode="r")
            np.testing.assert_array_equal(remote[:], expected)


class TestBackgroundServerBoundedShutdown:
    """BackgroundServer.shutdown() must not hang forever on a slow or stuck
    in-flight request."""

    def test_shutdown_returns_promptly_with_slow_inflight_request(self) -> None:
        """A request that takes far longer than shutdown_timeout must not
        prevent shutdown() from returning within roughly shutdown_timeout,
        via uvicorn's force_exit rather than an unbounded thread join."""
        import asyncio
        import threading
        import time

        import httpx
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Route

        from zarr_http_server._serve import _start_server

        async def slow(request: Any) -> Response:
            # Sleeps far longer than shutdown_timeout below, so a correct
            # implementation must force the connection closed rather than
            # wait for this to finish.
            await asyncio.sleep(5)
            return Response(status_code=204)

        app = Starlette(routes=[Route("/slow", slow, methods=["GET"])])
        port = _get_free_port()
        server = _start_server(
            app, host="127.0.0.1", port=port, background=True, shutdown_timeout=1
        )
        assert server is not None

        request_errors: list[BaseException] = []

        def make_slow_request() -> None:
            try:
                httpx.get(f"http://127.0.0.1:{port}/slow", timeout=10)
            except Exception as exc:  # noqa: BLE001 -- connection drop when the server force-closes is expected
                request_errors.append(exc)

        request_thread = threading.Thread(target=make_slow_request, daemon=True)
        request_thread.start()
        time.sleep(0.2)  # give the request time to actually start

        start = time.monotonic()
        server.shutdown()
        elapsed = time.monotonic() - start

        # shutdown_timeout=1 plus some scheduling slack; well under the
        # 5-second handler sleep an unbounded join would have waited out.
        assert elapsed < 3.0, f"shutdown() took {elapsed:.2f}s, expected well under 3s"

        request_thread.join(timeout=10)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestCorsOptionsCoverTheMiddleware:
    """`CorsOptions` exposes every `CORSMiddleware` parameter, with our own
    defaults only for the two the server knows about."""

    _ORIGIN = "https://viewer.example"

    def _client(self, store: Store, cors: CorsOptions) -> TestClient:
        return TestClient(store_app(store, methods={"GET"}, cors_options=cors))

    def test_ranged_response_is_readable_cross_origin(self, store: Store) -> None:
        """`Content-Range` is not CORS-safelisted, so without `expose_headers`
        a browser reads the bytes but cannot learn which bytes it got."""
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"0123456789")))
        client = self._client(store, {"allow_origins": [self._ORIGIN], "allow_methods": ["GET"]})

        response = client.get("/k", headers={"Origin": self._ORIGIN, "Range": "bytes=0-3"})

        assert response.status_code == 206
        assert response.headers["access-control-expose-headers"] == "Content-Range"

    def test_range_survives_a_preflight(self, store: Store) -> None:
        """A preflight naming `Range` must be allowed, not answered 400."""
        client = self._client(store, {"allow_origins": [self._ORIGIN], "allow_methods": ["GET"]})

        preflight = client.options(
            "/k",
            headers={
                "Origin": self._ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "range",
            },
        )

        assert preflight.status_code == 200
        assert "Range" in preflight.headers["access-control-allow-headers"]

    def test_caller_value_replaces_the_default(self, store: Store) -> None:
        """Our defaults apply only to absent keys; a supplied key wins outright
        so `expose_headers: []` means "expose nothing", not "expose ours"."""
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))
        base: CorsOptions = {"allow_origins": [self._ORIGIN], "allow_methods": ["GET"]}

        empty = self._client(store, {**base, "expose_headers": []})
        assert (
            "access-control-expose-headers"
            not in empty.get("/k", headers={"Origin": self._ORIGIN}).headers
        )

        custom = self._client(store, {**base, "expose_headers": ["X-Custom"]})
        assert (
            custom.get("/k", headers={"Origin": self._ORIGIN}).headers[
                "access-control-expose-headers"
            ]
            == "X-Custom"
        )

    def test_parameters_beyond_the_original_two_are_reachable(self, store: Store) -> None:
        """The regression this class exists for: `CorsOptions` used to carry
        only `allow_origins` and `allow_methods`, sealing the rest away."""
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))
        client = self._client(
            store,
            {
                "allow_origin_regex": r"https://.*\.example",
                "allow_credentials": True,
                "max_age": 30,
            },
        )
        origin = "https://sub.example"

        response = client.get("/k", headers={"Origin": origin})
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["access-control-allow-credentials"] == "true"

        preflight = client.options(
            "/k", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
        )
        assert preflight.headers["access-control-max-age"] == "30"


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestUvicornOptionsAreNotSealedOff:
    """`uvicorn.Config` takes ~50 parameters; naming four of them and dropping
    the rest would put TLS, proxy headers, `root_path` and log level out of
    reach entirely."""

    def test_options_reach_uvicorn_config(self, store: Store) -> None:
        """A key this signature does not name still lands on the Config."""
        from zarr_http_server import serve_store

        server = serve_store(
            store,
            host="127.0.0.1",
            port=0,
            background=True,
            uvicorn_options={"root_path": "/api", "log_level": "warning"},
        )
        assert server is not None
        try:
            config = server._server.config
            assert config.root_path == "/api"
            assert config.log_level == "warning"
            # Ours still apply where the caller did not override them.
            assert config.timeout_graceful_shutdown == 5
        finally:
            server.shutdown()

    def test_caller_options_win_over_ours(self, store: Store) -> None:
        """The merge order is ours-then-theirs, so a caller can override even
        an option this signature sets itself."""
        from zarr_http_server import serve_store

        server = serve_store(
            store,
            host="127.0.0.1",
            port=0,
            background=True,
            shutdown_timeout=5,
            uvicorn_options={"timeout_graceful_shutdown": 11},
        )
        assert server is not None
        try:
            assert server._server.config.timeout_graceful_shutdown == 11
        finally:
            server.shutdown()

    @pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs unix domain sockets")
    def test_non_tcp_bind_reports_no_url(self, store: Store, tmp_path: pathlib.Path) -> None:
        """A unix-socket bind has no host and port, so `url` must say so rather
        than naming an address nothing is listening on."""
        import httpx

        from zarr_http_server import serve_store

        sock = str(tmp_path / "s.sock")
        server = serve_store(store, background=True, uvicorn_options={"uds": sock})
        assert server is not None
        try:
            assert server.url is None
            assert server.host is None
            assert server.port is None
            # ...and it really is serving, just not over TCP.
            with httpx.Client(transport=httpx.HTTPTransport(uds=sock)) as client:
                response = client.get("http://localhost/zarr.json", timeout=10)
            assert response.status_code in (200, 404)
        finally:
            server.shutdown()


class TestHeadDoesNotTransferTheBody:
    """A HEAD body is discarded at the wire, so building one is pure waste."""

    def test_head_does_not_read_the_value(self, tmp_path: pathlib.Path) -> None:
        """The regression this class exists for: HEAD used to fall through to
        the GET handler and pull the whole object to report its length."""
        read = {"bytes": 0}

        class CountingLocal(LocalStore):
            async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
                buf = await super().get(key, prototype, byte_range)
                if buf is not None:
                    read["bytes"] += len(buf)
                return buf

        store = CountingLocal(str(tmp_path / "root"))
        payload = b"x" * 100_000
        sync(store.set("big", cpu.buffer_prototype.buffer.from_bytes(payload)))
        client = TestClient(store_app(store))

        read["bytes"] = 0
        response = client.head("/big")

        assert response.status_code == 200
        assert response.headers["content-length"] == str(len(payload))
        assert read["bytes"] == 0, "HEAD read the value to report its length"

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    def test_head_of_a_missing_key_is_404(self, store: Store) -> None:
        client = TestClient(store_app(store))
        assert client.head("/nope").status_code == 404


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestMetadataContentType:
    """Metadata documents are JSON in every zarr format, not just v3."""

    @pytest.mark.parametrize(
        ("zarr_format", "key"), [(3, "zarr.json"), (2, ".zarray"), (2, ".zattrs")]
    )
    def test_metadata_is_served_as_json(
        self, store: Store, zarr_format: ZarrFormat, key: str
    ) -> None:
        zarr.create_array(
            store, name="a", shape=(4,), chunks=(2,), dtype="i4", zarr_format=zarr_format
        )
        sync(store.set(f"a/{key}", cpu.buffer_prototype.buffer.from_bytes(b"{}")))

        response = TestClient(store_app(store)).get(f"/a/{key}")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestCorsAllowMethodsMatchTheRoute:
    """Advertising a method the route rejects is a promise the server cannot
    keep: the browser caches the preflight and every later call 405s."""

    def test_wildcard_expands_to_what_is_served(self, store: Store) -> None:
        """`"*"` is the idiomatic "everything this app does", so it expands to
        exactly that rather than to every verb Starlette knows."""
        app = store_app(
            store, methods={"GET"}, cors_options={"allow_origins": ["*"], "allow_methods": ["*"]}
        )

        preflight = TestClient(app).options(
            "/k", headers={"Origin": "https://e.test", "Access-Control-Request-Method": "GET"}
        )

        advertised = preflight.headers["access-control-allow-methods"]
        assert set(advertised.replace(" ", "").split(",")) == {"GET", "HEAD"}

    def test_advertising_an_unserved_method_is_rejected(self, store: Store) -> None:
        with pytest.raises(ValueError, match="does not serve"):
            store_app(
                store,
                methods={"GET"},
                cors_options={"allow_origins": ["*"], "allow_methods": ["GET", "DELETE"]},
            )

    def test_head_counts_as_served_when_get_is(self, store: Store) -> None:
        """Starlette routes HEAD wherever GET goes, so naming it is not an error."""
        store_app(
            store,
            methods={"GET"},
            cors_options={"allow_origins": ["*"], "allow_methods": ["GET", "HEAD"]},
        )

    def test_absent_allow_methods_is_left_alone(self, store: Store) -> None:
        """Starlette's GET-only default stands; widening it to everything
        served would newly advertise PUT on a write-enabled app."""
        app = store_app(store, methods={"GET", "PUT"}, cors_options={"allow_origins": ["*"]})

        preflight = TestClient(app).options(
            "/k", headers={"Origin": "https://e.test", "Access-Control-Request-Method": "GET"}
        )

        assert "PUT" not in preflight.headers["access-control-allow-methods"]


class TestReadOnlyServing:
    """The guarantees a read-only deployment rests on.

    Two independent layers: `methods` decides what the route answers, and the
    store decides whether a write could succeed at all. The second is the one
    that survives a misconfiguration of the first, so both are pinned here.
    """

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    @pytest.mark.parametrize("method", ["PUT", "POST", "DELETE", "PATCH"])
    def test_default_app_refuses_every_mutating_method(self, store: Store, method: str) -> None:
        """The default is read-only: no argument is needed to get there, and
        nothing a client sends can write."""
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))
        before = sync(store.get("k", cpu.buffer_prototype)).to_bytes()
        client = TestClient(store_app(store), raise_server_exceptions=False)

        response = client.request(method, "/k", content=b"overwritten")

        assert response.status_code == 405
        assert sync(store.get("k", cpu.buffer_prototype)).to_bytes() == before

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    def test_post_can_never_be_enabled(self, store: Store) -> None:
        """POST is not merely unrouted, it is unconfigurable: there is no
        handler behavior for it, so asking is an error rather than a no-op."""
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            store_app(store, methods={"GET", "POST"})  # type: ignore[arg-type]

    def test_read_only_store_refuses_writes_independently_of_methods(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The layer that survives getting `methods` wrong.

        Constructed through the private builder because the public entry
        points now reject this combination outright; the handler check stays
        as the backstop for a store whose `read_only` is not fixed.
        """
        from zarr_http_server._serve import _make_starlette_app

        writable = LocalStore(str(tmp_path / "root"))
        sync(writable.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))

        app = _make_starlette_app(methods={"GET", "PUT"})
        app.state.store = writable.with_read_only(True)
        app.state.node = None
        app.state.prefix = ""
        app.state.max_body_size = None

        response = TestClient(app, raise_server_exceptions=False).put("/k", content=b"x")

        assert response.status_code == 403
        assert sync(writable.get("k", cpu.buffer_prototype)).to_bytes() == b"data"

    def test_put_on_a_read_only_store_is_rejected_at_construction(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A write that could never succeed is a configuration error, not a
        runtime 403 delivered to whoever happens to try first."""
        store = LocalStore(str(tmp_path / "root")).with_read_only(True)

        with pytest.raises(ValueError, match="store is read-only"):
            store_app(store, methods={"GET", "PUT"})

    def test_read_only_node_is_rejected_at_construction(self, tmp_path: pathlib.Path) -> None:
        """The same check applies to a node, whose store it inherits."""
        store = LocalStore(str(tmp_path / "root"))
        zarr.create_array(store, shape=(4,), chunks=(2,), dtype="i4", compressors=None)
        read_only_array = zarr.open_array(store, mode="r")

        with pytest.raises(ValueError, match="store is read-only"):
            node_app(read_only_array, methods={"GET", "PUT"})


class TestMethodSetConstants:
    """Named method sets let a call site state its intent, and make writable
    deployments findable: every writable app must name one."""

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    def test_read_only_constant_matches_the_default(self, store: Store) -> None:
        """Passing it explicitly and omitting `methods` are the same server, so
        saying so out loud costs nothing."""
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))

        default = TestClient(store_app(store), raise_server_exceptions=False)
        named = TestClient(
            store_app(store, methods=READ_ONLY_HTTP_METHODS), raise_server_exceptions=False
        )

        for method in ["GET", "HEAD", "PUT", "POST", "DELETE", "PATCH"]:
            assert default.request(method, "/k").status_code == (
                named.request(method, "/k").status_code
            )

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    @pytest.mark.parametrize("method", ["PUT", "POST", "DELETE", "PATCH"])
    def test_read_only_constant_refuses_writes(self, store: Store, method: str) -> None:
        sync(store.set("k", cpu.buffer_prototype.buffer.from_bytes(b"data")))
        client = TestClient(
            store_app(store, methods=READ_ONLY_HTTP_METHODS), raise_server_exceptions=False
        )

        assert client.request(method, "/k", content=b"x").status_code == 405
        assert sync(store.get("k", cpu.buffer_prototype)).to_bytes() == b"data"

    @pytest.mark.parametrize("store", ["memory"], indirect=True)
    def test_read_write_constant_permits_exactly_put(self, store: Store) -> None:
        """It grants writes -- and still not POST, which has no handler."""
        client = TestClient(
            store_app(store, methods=READ_WRITE_HTTP_METHODS), raise_server_exceptions=False
        )

        assert client.put("/k", content=b"data").status_code == 204
        assert client.post("/k", content=b"data").status_code == 405
        assert sync(store.get("k", cpu.buffer_prototype)).to_bytes() == b"data"

    def test_constants_cannot_be_mutated_by_a_caller(self) -> None:
        """Frozen, so one caller cannot widen the default for every other."""
        assert isinstance(READ_ONLY_HTTP_METHODS, frozenset)
        assert isinstance(READ_WRITE_HTTP_METHODS, frozenset)
        assert "PUT" not in READ_ONLY_HTTP_METHODS

    def test_constants_match_the_types_they_model(self) -> None:
        """The sets are derived from the Literals, so they cannot disagree
        about what this server serves. Pinning the contents here makes
        widening either type a deliberate, visible edit."""
        assert frozenset(get_args(ReadOnlyHTTPMethod)) == READ_ONLY_HTTP_METHODS
        assert set(READ_ONLY_HTTP_METHODS) == {"GET", "HEAD"}
        assert set(READ_WRITE_HTTP_METHODS) == {"GET", "HEAD", "PUT"}
        # Read-only is a strict subset: the only difference is the write verb.
        assert READ_ONLY_HTTP_METHODS < READ_WRITE_HTTP_METHODS
        assert {"PUT"} == READ_WRITE_HTTP_METHODS - READ_ONLY_HTTP_METHODS
