"""Property-based tests driven against a real HTTP endpoint.

Every test here talks to an actual uvicorn server over a socket, not to an
in-process ASGI client, so the properties cover the parts of the stack that
only exist on the wire: header parsing, method dispatch, and status codes.

Each property checks **two** things after a request -- the response, and the
state of the backing store. That pairing is the point. A server can answer
correctly and corrupt the store, or refuse a request and write anyway, and a
response-only assertion sees neither. The bug that motivated this module did
exactly that: a `PUT` to a non-canonically spelled chunk key ("c/00/00"
instead of "c/0/0") answered `204 No Content` and stored the body under a key
no reader ever looks up, so the client saw success and the data was invisible.

The vocabulary below splits keys into two families:

* **in-band** -- keys the served node genuinely owns: its metadata documents
  and the canonical spelling of each chunk key in its grid. These must be
  served, and writes to them must be visible to a zarr reader.
* **out-of-band** -- everything else: non-canonical spellings of a valid
  chunk key, coordinates outside the grid, traversal attempts, and keys
  belonging to a sibling node. These must be refused *and* must leave the
  store byte-for-byte unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import httpx
import numpy as np
import pytest
import zarr
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from zarr.buffer import cpu
from zarr.core.sync import sync

from zarr_http_server import serve_node, serve_store
from zarr_http_server._keys import _shard_grid_shape

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr import Array
    from zarr.abc.store import Store

# Real HTTP round trips are slower and jumpier than hypothesis' default
# deadline allows, and a CI runner under load makes that worse.
_HTTP = settings(deadline=None, max_examples=50)

# An uncompressed array so a chunk's bytes are exactly its raw values: a PUT
# body of the right length is a legitimate chunk, which lets these tests
# assert that a write is readable through a zarr client rather than merely
# present in the store.
_SHAPE = (6, 4)
_CHUNKS = (2, 2)
_DTYPE = "int32"


@dataclass(frozen=True)
class Served:
    """A running server plus the handles needed to reason about its keys."""

    url: str
    store: Store
    array: Array[Any]
    kind: Literal["store", "node"]

    http_prefix: str
    """Prepended to an array-relative key to form the request path."""

    store_prefix: str
    """Prepended to an array-relative key to form the backing store key."""

    def path(self, array_relative_key: str) -> str:
        return f"{self.http_prefix}{array_relative_key}"

    def store_key(self, array_relative_key: str) -> str:
        return f"{self.store_prefix}{array_relative_key}"


def _snapshot(store: Store) -> dict[str, bytes]:
    """Every key in *store* mapped to its bytes.

    Comparing two snapshots is how these tests assert that a rejected request
    changed nothing -- not just that it added no key, but that it modified
    none either.
    """

    async def _read() -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        async for key in store.list():
            buf = await store.get(key, cpu.buffer_prototype)
            if buf is not None:
                out[key] = buf.to_bytes()
        return out

    return sync(_read())


@pytest.fixture(
    scope="module",
    params=[
        ("store", 2, "memory"),
        ("store", 3, "memory"),
        ("node", 2, "memory"),
        ("node", 3, "memory"),
        # LocalStore reads through the filesystem, which sizes its read from
        # the range rather than from the object and so raises on an over-wide
        # one. MemoryStore just slices a `bytes` and is happy with any bound,
        # so a memory-only matrix cannot tell a clamped range from a refused
        # one -- the store backend is part of what these properties test.
        ("store", 3, "local"),
        ("node", 3, "local"),
    ],
    ids=["store-v2", "store-v3", "node-v2", "node-v3", "store-v3-local", "node-v3-local"],
)
def served(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Served]:
    """One real server per (app kind, zarr format, store backend).

    Module-scoped because hypothesis drives hundreds of requests per test and
    a server per example would dominate the runtime. `port=0` lets the OS
    choose the port and `server.url` reports what it actually bound, which
    avoids the bind-then-close race of picking a port up front.
    """
    kind, zarr_format, backend = request.param

    store: Store
    if backend == "local":
        root_dir = tmp_path_factory.mktemp(f"{kind}-v{zarr_format}")
        store = zarr.storage.LocalStore(root_dir)
    else:
        store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    array = root.create_array(
        "inside", shape=_SHAPE, chunks=_CHUNKS, dtype=_DTYPE, compressors=None
    )
    array[:] = np.arange(int(np.prod(_SHAPE)), dtype=_DTYPE).reshape(_SHAPE)
    # A sibling the node_app must never serve, and whose keys therefore make
    # good out-of-band probes.
    sibling = root.create_array(
        "outside", shape=_CHUNKS, chunks=_CHUNKS, dtype=_DTYPE, compressors=None
    )
    sibling[:] = 1

    if kind == "node":
        server = serve_node(
            array, host="127.0.0.1", port=0, background=True, methods={"GET", "PUT"}
        )
        http_prefix = ""
    else:
        server = serve_store(
            store, host="127.0.0.1", port=0, background=True, methods={"GET", "PUT"}
        )
        http_prefix = "inside/"
    assert server is not None

    try:
        yield Served(
            url=server.url,
            store=store,
            array=array,
            kind=kind,
            http_prefix=http_prefix,
            store_prefix="inside/",
        )
    finally:
        server.shutdown()


def _grid(array: Array[Any]) -> tuple[int, ...]:
    return _shard_grid_shape(array)


@st.composite
def chunk_coords(draw: st.DrawFn, array: Array[Any]) -> tuple[int, ...]:
    """Coordinates of a chunk that exists in *array*'s storage grid."""
    return tuple(draw(st.integers(min_value=0, max_value=g - 1)) for g in _grid(array))


@st.composite
def out_of_grid_coords(draw: st.DrawFn, array: Array[Any]) -> tuple[int, ...]:
    """Coordinates outside the grid, so the key is well-formed but names nothing."""
    grid = _grid(array)
    axis = draw(st.integers(min_value=0, max_value=len(grid) - 1))
    coords = list(draw(chunk_coords(array)))
    coords[axis] = draw(st.integers(min_value=grid[axis], max_value=grid[axis] + 50))
    return tuple(coords)


# The confusability with ASCII digits is the point: `int()` accepts these as
# decimal digits, so they name a different store key while decoding to the
# same coordinate. That is the defect these strategies probe for.
_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_FULLWIDTH = "０１２３４５６７８９"  # noqa: RUF001


def _respellings(digits: str) -> list[str]:
    """Strings `int()` maps to the same value as *digits*, spelled differently.

    These are exactly the spellings that make decode-only validation unsafe:
    each names a *different* store key while decoding to the same coordinate.
    """
    value = int(digits)
    return [
        f"0{digits}",
        f"00{digits}",
        f"+{digits}",
        f" {digits}",
        f"{digits} ",
        f"\t{digits}",
        "".join(_ARABIC_INDIC[int(d)] for d in digits),
        "".join(_FULLWIDTH[int(d)] for d in digits),
        *([f"-{digits}"] if value == 0 else []),
    ]


@st.composite
def non_canonical_chunk_keys(draw: st.DrawFn, array: Array[Any]) -> str:
    """A chunk key that decodes into the grid but is not zarr's own spelling.

    Built by taking the canonical key and re-spelling one of its digit runs,
    which keeps this strategy independent of the chunk key encoding's grammar.
    """
    coords = draw(chunk_coords(array))
    canonical = array.metadata.encode_chunk_key(coords)
    runs = list(re.finditer(r"\d+", canonical))
    assume(runs)
    run = draw(st.sampled_from(runs))
    replacement = draw(st.sampled_from(_respellings(run.group())))
    key = canonical[: run.start()] + replacement + canonical[run.end() :]
    assume(key != canonical)
    return key


TRAVERSAL_KEYS = [
    # Percent-encoded on purpose. An HTTP client resolves dot-segments before
    # it sends -- httpx turns "../secret" into "/secret" and ".." into "/" per
    # RFC 3986 §5.2.4 -- so a literal "../" probe never reaches the server as
    # traversal and asserts nothing. Encoded, it survives the client intact
    # and Starlette decodes it back into a real ".." segment on arrival, which
    # is the form the server's own guard has to catch.
    "..%2Fsecret",
    "%2e%2e%2Fsecret",
    "%2e%2e%2F%2e%2e%2Fetc%2Fpasswd",
    "%2e%2e",
    "%2e",
    "%2e%2Fzarr.json",
    "a%2F..%2F..%2Fb",
    "%2Fabsolute",
    "sub%2F%2Fempty",
    "C%3A%2Fwindows",
    "..%5Cwindows",
    "a%5C..%5C..%5Cb",
    "%00nul",
]


def _chunk_payload(array: Array[Any]) -> bytes:
    """Bytes of a full, uncompressed chunk for *array*."""
    return np.zeros(_CHUNKS, dtype=array.dtype).tobytes()


def _url(served: Served, array_relative_key: str) -> str:
    """Request URL for a key, percent-encoded so it survives the wire verbatim.

    Generated keys contain characters a URL cannot carry literally -- a tab
    makes httpx raise `InvalidURL`, and a space or a non-ASCII digit would be
    re-encoded on the way out anyway. Encoding here means Starlette decodes
    the path param back to exactly the key the strategy produced, so the
    property really is "for any key K, a request for K is refused" rather than
    "for any key the URL parser happened to leave alone".
    """
    return f"{served.url}/{quote(served.path(array_relative_key), safe='/')}"


def _require_node_scope(served: Served) -> None:
    """Skip a property that only a node-scoped app can satisfy.

    `store_app` proxies the store's raw key space and has no array semantics
    to validate against, so every syntactically acceptable key is in-band for
    it by contract -- including a non-canonical chunk spelling. Only
    `node_app` claims to serve exactly one node's keys, so only `node_app` can
    be held to what that set contains.
    """
    if served.kind != "node":
        pytest.skip("store_app serves the raw key space; node scoping does not apply")


class TestInBandRequests:
    """Keys the node owns are served, and writes to them are visible."""

    @given(data=st.data())
    @_HTTP
    def test_get_of_a_canonical_key_returns_the_stored_bytes(
        self, served: Served, data: st.DataObject
    ) -> None:
        """A GET of an in-band key returns exactly what the store holds, and
        reading never changes the store."""
        coords = data.draw(chunk_coords(served.array))
        relative = served.array.metadata.encode_chunk_key(coords)

        before = _snapshot(served.store)
        response = httpx.get(_url(served, relative), timeout=30)

        expected = before.get(served.store_key(relative))
        if expected is None:
            assert response.status_code == 404
        else:
            assert response.status_code == 200
            assert response.content == expected
        assert _snapshot(served.store) == before

    @given(data=st.data())
    @_HTTP
    def test_put_of_a_canonical_key_is_stored_and_readable(
        self, served: Served, data: st.DataObject
    ) -> None:
        """The headline property: a PUT that reports success must be visible.

        Success means three things at once -- a 2xx, the bytes landing under
        the key the client named, and a zarr client subsequently reading back
        the values that were written. The original defect satisfied the first
        and failed the other two.
        """
        coords = data.draw(chunk_coords(served.array))
        fill = data.draw(st.integers(min_value=-(2**31), max_value=2**31 - 1))
        relative = served.array.metadata.encode_chunk_key(coords)
        payload = np.full(_CHUNKS, fill, dtype=served.array.dtype).tobytes()

        before = _snapshot(served.store)
        response = httpx.put(_url(served, relative), content=payload, timeout=30)
        assert response.status_code == 204

        after = _snapshot(served.store)
        key = served.store_key(relative)
        assert after[key] == payload, "the body did not land under the key the client named"
        assert set(after) - set(before) <= {key}, "the write touched a key the client did not name"

        # The write is not merely present, it is legible: reopen the array and
        # read the chunk the coordinates address.
        reread = zarr.open_array(served.store, path="inside")
        block = tuple(slice(c * s, (c + 1) * s) for c, s in zip(coords, _CHUNKS, strict=True))
        assert np.array_equal(reread[block], np.full(_CHUNKS, fill, dtype=served.array.dtype))

    @given(data=st.data())
    @_HTTP
    def test_range_of_a_stored_key_returns_the_matching_slice(
        self, served: Served, data: st.DataObject
    ) -> None:
        """A satisfiable range returns exactly the bytes it names, and says so
        in Content-Range."""
        relative = served.array.metadata.encode_chunk_key(tuple(0 for _ in _grid(served.array)))
        key = served.store_key(relative)
        before = _snapshot(served.store)
        assume(key in before)
        body = before[key]

        start = data.draw(st.integers(min_value=0, max_value=len(body) - 1))
        end = data.draw(st.integers(min_value=start, max_value=len(body) - 1))

        response = httpx.get(
            _url(served, relative), headers={"Range": f"bytes={start}-{end}"}, timeout=30
        )

        assert response.status_code == 206
        assert response.content == body[start : end + 1]
        # RFC 9110 §15.3.7: a single-part 206 must carry Content-Range, and it
        # must describe the bytes actually returned.
        content_range = response.headers["content-range"]
        assert content_range.startswith(f"bytes {start}-{start + len(response.content) - 1}/")
        assert _snapshot(served.store) == before

    @given(suffix=st.integers(min_value=1, max_value=200))
    @_HTTP
    def test_suffix_range_returns_the_tail_and_locates_it(
        self, served: Served, suffix: int
    ) -> None:
        """A suffix range must report where in the object its bytes came from.

        Zarr's sharding codec reads a shard index this way, so a 206 without
        Content-Range leaves the reader unable to tell a clamped whole-object
        read from the tail it asked for.
        """
        relative = served.array.metadata.encode_chunk_key(tuple(0 for _ in _grid(served.array)))
        key = served.store_key(relative)
        before = _snapshot(served.store)
        assume(key in before)
        body = before[key]

        response = httpx.get(
            _url(served, relative), headers={"Range": f"bytes=-{suffix}"}, timeout=30
        )

        assert response.status_code == 206
        expected = body[-suffix:] if suffix <= len(body) else body
        assert response.content == expected
        first = len(body) - len(expected)
        assert response.headers["content-range"].startswith(f"bytes {first}-{len(body) - 1}/")
        assert _snapshot(served.store) == before


class TestOutOfBandRequests:
    """Keys the node does not own are refused, and change nothing."""

    @given(data=st.data())
    @_HTTP
    def test_non_canonical_chunk_key_is_refused_and_writes_nothing(
        self, served: Served, data: st.DataObject
    ) -> None:
        """The regression that motivated this module.

        A key that decodes into the grid but is spelled differently from
        zarr's own rendering names a store key no reader consults. Accepting a
        write to it reports success and loses the data, so it must be refused
        and the store must be untouched.
        """
        _require_node_scope(served)
        relative = data.draw(non_canonical_chunk_keys(served.array))
        payload = _chunk_payload(served.array)

        before = _snapshot(served.store)
        put = httpx.put(_url(served, relative), content=payload, timeout=30)
        assert put.status_code == 404
        assert _snapshot(served.store) == before, "a refused write still modified the store"

        get = httpx.get(_url(served, relative), timeout=30)
        assert get.status_code == 404
        assert _snapshot(served.store) == before

    @given(data=st.data())
    @_HTTP
    def test_out_of_grid_chunk_key_is_refused_and_writes_nothing(
        self, served: Served, data: st.DataObject
    ) -> None:
        """Coordinates past the end of the grid address no chunk of this array."""
        _require_node_scope(served)
        coords = data.draw(out_of_grid_coords(served.array))
        relative = served.array.metadata.encode_chunk_key(coords)

        before = _snapshot(served.store)
        put = httpx.put(_url(served, relative), content=_chunk_payload(served.array), timeout=30)
        get = httpx.get(_url(served, relative), timeout=30)

        assert put.status_code == 404
        assert get.status_code == 404
        assert _snapshot(served.store) == before

    @given(key=st.sampled_from(TRAVERSAL_KEYS))
    @_HTTP
    def test_traversal_key_is_refused_and_writes_nothing(self, served: Served, key: str) -> None:
        """Nothing that tries to leave the served scope may be served or written."""
        before = _snapshot(served.store)
        put = httpx.put(f"{served.url}/{key}", content=b"payload", timeout=30)
        get = httpx.get(f"{served.url}/{key}", timeout=30)

        assert put.status_code in (403, 404, 405), f"{key!r} was accepted for writing"
        assert get.status_code in (403, 404, 405), f"{key!r} was served"
        assert _snapshot(served.store) == before

    @given(data=st.data())
    @_HTTP
    def test_node_app_never_serves_a_sibling(self, served: Served, data: st.DataObject) -> None:
        """A node_app is scoped to one node, so a sibling's keys are invisible
        even though they exist in the same store."""
        if served.kind != "node":
            pytest.skip("store_app deliberately serves the whole store")

        coords = data.draw(chunk_coords(served.array))
        chunk = served.array.metadata.encode_chunk_key(coords)
        relative = data.draw(
            st.sampled_from(
                [
                    f"outside/{chunk}",
                    "outside/zarr.json",
                    "outside/.zarray",
                    f"../outside/{chunk}",
                ]
            )
        )

        before = _snapshot(served.store)
        get = httpx.get(f"{served.url}/{relative}", timeout=30)
        put = httpx.put(f"{served.url}/{relative}", content=b"payload", timeout=30)

        assert get.status_code == 404
        assert put.status_code == 404
        assert _snapshot(served.store) == before


class TestMethodsAndRanges:
    """Transport-level invariants that hold for every key."""

    @given(
        method=st.sampled_from(["DELETE", "POST", "PATCH", "OPTIONS"]),
        data=st.data(),
    )
    @_HTTP
    def test_unconfigured_method_is_refused_and_writes_nothing(
        self, served: Served, method: str, data: st.DataObject
    ) -> None:
        """Only the methods the app was configured with may reach the store."""
        coords = data.draw(chunk_coords(served.array))
        relative = served.array.metadata.encode_chunk_key(coords)

        before = _snapshot(served.store)
        response = httpx.request(method, _url(served, relative), content=b"payload", timeout=30)

        assert response.status_code == 405
        assert _snapshot(served.store) == before

    @given(
        header=st.sampled_from(
            [
                "bytes=abc-def",
                "bytes=0-1,4-5",
                "chars=0-7",
                "bytes=",
                "nonsense",
                "bytes=+0-1",
            ]
        )
    )
    @_HTTP
    def test_unusable_range_serves_the_whole_object(self, served: Served, header: str) -> None:
        """RFC 9110 §14.2: a Range the server cannot use is ignored, not refused."""
        relative = served.array.metadata.encode_chunk_key(tuple(0 for _ in _grid(served.array)))
        before = _snapshot(served.store)
        key = served.store_key(relative)
        assume(key in before)

        response = httpx.get(_url(served, relative), headers={"Range": header}, timeout=30)

        assert response.status_code == 200
        assert response.content == before[key]
        assert "content-range" not in response.headers

    @given(
        header=st.sampled_from(
            [
                "bytes=5-2",
                "bytes=-0",
                "bytes=99999999999999999999-",
                "bytes=100000-100001",
            ]
        )
    )
    @_HTTP
    def test_unsatisfiable_range_is_refused(self, served: Served, header: str) -> None:
        """A well-formed range that names nothing readable is a 416."""
        relative = served.array.metadata.encode_chunk_key(tuple(0 for _ in _grid(served.array)))
        before = _snapshot(served.store)
        assume(served.store_key(relative) in before)

        response = httpx.get(_url(served, relative), headers={"Range": header}, timeout=30)

        assert response.status_code == 416
        assert _snapshot(served.store) == before

    @given(end=st.integers(min_value=10**19, max_value=10**30))
    @_HTTP
    def test_absurdly_wide_range_is_clamped_not_refused(self, served: Served, end: int) -> None:
        """RFC 9110 §14.1.2 clamps a last-byte-pos past the end of the object,
        so an over-wide range reads to EOF rather than erroring."""
        relative = served.array.metadata.encode_chunk_key(tuple(0 for _ in _grid(served.array)))
        key = served.store_key(relative)
        before = _snapshot(served.store)
        assume(key in before)

        response = httpx.get(
            _url(served, relative), headers={"Range": f"bytes=0-{end}"}, timeout=30
        )

        assert response.status_code == 206
        assert response.content == before[key]
        assert _snapshot(served.store) == before
