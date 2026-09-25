from __future__ import annotations

import asyncio
import errno
import logging
import ntpath
import socket
import sys
import threading
import time
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast, get_args

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.buffer import cpu

from zarr_http_server._keys import array_metadata_keys, group_metadata_keys, is_valid_node_key

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from zarr import Array, Group
    from zarr.abc.store import ByteRequest, Store

__all__ = [
    "AUTO_PORT",
    "DEFAULT_MAX_BODY_SIZE",
    "DEFAULT_PORT",
    "READ_ONLY_HTTP_METHODS",
    "READ_WRITE_HTTP_METHODS",
    "BackgroundServer",
    "CorsOptions",
    "HTTPMethod",
    "ReadOnlyHTTPMethod",
    "node_app",
    "serve",
    "serve_background",
    "store_app",
]


class CorsOptions(TypedDict, total=False):
    """Options forwarded to Starlette's `CORSMiddleware`.

    Every parameter the middleware accepts appears here, so configuring CORS
    never requires reaching around this package. Keys left out fall back to
    the defaults described below; a key that is present is used verbatim,
    including an empty list.

    Two defaults differ from Starlette's own, because this server knows
    something its caller should not have to. It emits `Content-Range` on
    every ranged response, which is *not* a CORS-safelisted response header --
    with Starlette's empty `expose_headers` a browser client can read the
    bytes but not learn which bytes it got. And it accepts a `Range` request
    header, which Starlette's empty `allow_headers` would reject at preflight.

    * `expose_headers` defaults to `["Content-Range"]`
    * `allow_headers` defaults to `["Range"]`

    Everything else defaults to the middleware's own value: no origins, `GET`
    only, no credentials, no origin regex, no private-network access, and a
    600-second preflight cache.
    """

    allow_origins: list[str]
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool
    allow_origin_regex: str | None
    allow_private_network: bool
    expose_headers: list[str]
    max_age: int


_CORS_DEFAULTS: CorsOptions = {
    "expose_headers": ["Content-Range"],
    "allow_headers": ["Range"],
}


ReadOnlyHTTPMethod = Literal["GET", "HEAD"]
"""An HTTP method that cannot modify the store.

Distinguished from `HTTPMethod` in the type domain, not only at runtime, so a
read-only interface can be *declared* rather than merely configured: a
parameter annotated `AbstractSet[ReadOnlyHTTPMethod]` cannot be handed `"PUT"`
without a type error, whatever the value turns out to be at runtime.

`HEAD` belongs here because Starlette routes it wherever `GET` goes, which is
what RFC 9110 §9.3.2 asks of an origin server. It is answered from the value's
size rather than by building and discarding a body.
"""

_WriteHTTPMethod = Literal["PUT"]
"""An HTTP method that modifies the store.

Private because nothing needs to name "the write methods" on its own -- it
exists so `HTTPMethod` can be defined as the union rather than as a third
hand-written list of the same strings.
"""

HTTPMethod = ReadOnlyHTTPMethod | _WriteHTTPMethod
"""An HTTP method this server implements.

`GET` and `HEAD` read a key; `PUT` writes one. Other verbs are not accepted:
the handler has no behavior for them, so serving them would silently answer
as if they were `GET`.
"""

# Derived from the types above rather than restated, so the runtime sets and
# the static types cannot disagree about what this server serves. Adding a
# method to a Literal is then the only edit needed.
READ_ONLY_HTTP_METHODS: frozenset[ReadOnlyHTTPMethod] = frozenset(get_args(ReadOnlyHTTPMethod))
"""Methods that only read. The default for every app in this package.

Naming the set makes a read-only deployment say so at the call site, rather
than being the absence of an argument:

    store_app(store, methods=READ_ONLY_HTTP_METHODS)

Typed as a set of `ReadOnlyHTTPMethod`, so a caller building on it keeps the
static guarantee: adding `"PUT"` to a `frozenset[ReadOnlyHTTPMethod]` is a
type error, not a runtime surprise.

The stronger guarantee is a read-only store, which holds however `methods` is
configured -- see `store.with_read_only(True)`.
"""

READ_WRITE_HTTP_METHODS: frozenset[HTTPMethod] = frozenset(
    get_args(ReadOnlyHTTPMethod) + get_args(_WriteHTTPMethod)
)
"""Methods that read and write. Serving these grants clients write access.

Every writable app must name a method set, so this constant is also what makes
writable deployments findable: grepping for `READ_WRITE_HTTP_METHODS` (or for
`methods=` generally) turns up every place that opts in.
"""

_SUPPORTED_METHODS: frozenset[str] = READ_WRITE_HTTP_METHODS

_LOGGER = logging.getLogger("uvicorn.error")
"""uvicorn's own logger, so a port fallback appears alongside its startup lines."""

DEFAULT_PORT = 8000
"""Port tried first when `port="auto"`."""

AUTO_PORT: Literal["auto"] = "auto"
"""Sentinel for `port`: prefer `DEFAULT_PORT`, but settle for any free port.

An explicit port is a requirement -- it binds that port or fails -- because a
caller who names one usually has something else expecting the server there.
`"auto"` says the opposite: no particular port is needed, so a collision
should not stop the server from starting. `port=0` keeps its usual meaning of
"any free port", with no preference.
"""

_STARTUP_TIMEOUT = 5.0
"""Seconds to wait for a background server to report that it is listening."""

_STARTUP_ABANDON_TIMEOUT = 5.0
"""Seconds to wait for a server that failed to start to stop again."""

_SHUTDOWN_JOIN_MARGIN = 1.0
"""Seconds to wait beyond uvicorn's graceful bound before forcing shutdown.

uvicorn spends a fixed ~0.2s tearing down (a 0.1s loop tick plus a 0.1s
sleep) before its own `timeout_graceful_shutdown` wait begins, so a join that
merely equals that bound is guaranteed to expire first and escalate to
`force_exit` -- which makes uvicorn skip ASGI lifespan shutdown.
"""

DEFAULT_MAX_BODY_SIZE = 256 * 1024 * 1024
"""Default cap on a `PUT` body, in bytes.

`Store.set` takes a whole `Buffer`, so an accepted body is held in memory in
full; the body is read incrementally and abandoned once it passes this cap,
so one request cannot size the server's memory use. Pass
`max_body_size=None` to lift the cap and read the body whole.
"""


class BackgroundServer:
    """A running background HTTP server that can be used as a context manager.

    Wraps a ``uvicorn.Server`` running in a daemon thread.  When used as a
    context manager the server is shut down automatically on exit.

    Parameters
    ----------
    server : uvicorn.Server
        The running uvicorn server instance.
    thread : threading.Thread
        The daemon thread running the server.
    host : str or None
        The host the server was asked to bind, or ``None`` when it is not
        listening on a TCP socket.
    port : int or None
        The port actually bound, or ``None`` when the server is not listening
        on a TCP socket.
    scheme : str, optional
        URL scheme the server is reachable over. Defaults to ``"http"``.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully during
        :meth:`shutdown` before forcing the server closed. Defaults to ``5``.

    Examples
    --------
    >>> with serve_background(node_app(arr)) as server:  # doctest: +SKIP
    ...     print(f"Listening on {server.host}:{server.port}")
    ...     # server is shut down when the block exits
    """

    def __init__(
        self,
        server: uvicorn.Server,
        thread: threading.Thread,
        *,
        host: str | None,
        port: int | None,
        scheme: str = "http",
        shutdown_timeout: int = 5,
    ) -> None:
        self._server = server
        self._thread = thread
        self.host = host
        self.port = port
        self.scheme = scheme
        self._shutdown_timeout = shutdown_timeout

    @property
    def url(self) -> str | None:
        """The base URL of the running server.

        ``None`` when the server is not listening on a TCP socket -- a unix
        socket or an inherited file descriptor has no host and port, and
        inventing one would be a URL that connects to nothing.
        """
        if self.host is None or self.port is None:
            return None
        return f"{self.scheme}://{self.host}:{self.port}"

    def shutdown(self) -> None:
        """Signal the server to shut down and wait for it to stop.

        Waits for the server thread to exit on its own, then escalates to
        ``force_exit`` if it has not, so a request wedged outside uvicorn's
        loop cannot block here forever.

        Raises
        ------
        RuntimeError
            If the thread is still running after both waits. Returning
            normally would report success for a server that is still bound to
            its port and still serving, which the caller cannot detect any
            other way.
        """
        self._server.should_exit = True
        # Outlast uvicorn's own graceful wait rather than matching it. uvicorn
        # spends roughly 0.2s on teardown (a 0.1s loop tick plus a 0.1s sleep)
        # *before* its `timeout_graceful_shutdown` wait even begins, so an
        # equal bound here always expires first -- escalating to force_exit on
        # the path that is supposed to be the orderly one, which makes uvicorn
        # skip ASGI lifespan shutdown entirely.
        self._thread.join(timeout=self._graceful_timeout + _SHUTDOWN_JOIN_MARGIN)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=self._shutdown_timeout)

        if self._thread.is_alive():
            raise RuntimeError(
                "Server thread did not stop within "
                f"{self._graceful_timeout + _SHUTDOWN_JOIN_MARGIN + self._shutdown_timeout:.1f}s, "
                "even after force_exit. The server may still be serving and "
                "holding its port; a request blocked in a store call cannot be "
                "cancelled from here."
            )

    @property
    def _graceful_timeout(self) -> float:
        """uvicorn's own graceful-shutdown bound, whoever configured it."""
        configured = self._server.config.timeout_graceful_shutdown
        return float(configured) if configured is not None else float(self._shutdown_timeout)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()


class _RangeVerdict(Enum):
    """The outcome of a Range header that does not name a readable range."""

    IGNORE = auto()
    """Serve the full representation with 200, as if no Range had been sent."""

    UNSATISFIABLE = auto()
    """Answer 416: the range is well-formed but names nothing readable."""


_MAX_BYTE_POS = sys.maxsize
"""Largest byte position this server will pass to a store.

Range bounds arrive as arbitrary-precision Python integers, but a store
ultimately turns them into an index-sized `seek`/`read`. Feeding an oversized
value through raises `OverflowError`/`ValueError` from deep inside the store
rather than producing a response, so bounds are clamped or rejected here.
"""


def _parse_int(text: str) -> int | None:
    """Parse a byte position, accepting only the canonical spelling of one.

    `int` is lenient in ways an HTTP byte position is not -- it accepts
    surrounding whitespace, a leading `+`/`-`, underscore separators, and
    non-ASCII decimal digits -- so `bytes=+0-1` and `bytes=0_0-1` would parse.
    RFC 9110 defines a byte position as 1*DIGIT.
    """
    if not text.isascii() or not text.isdigit():
        return None
    return int(text)


def _parse_range_header(range_header: str) -> ByteRequest | _RangeVerdict:
    """Parse an HTTP Range header into a ByteRequest.

    A header this server cannot turn into a single read is *ignored* rather
    than rejected. RFC 9110 §14.2 requires a server to ignore a Range whose
    unit it does not recognize, and permits ignoring one it cannot parse; in
    both cases the correct answer is the full representation, not 416.
    Answering 416 would tell a client the object is unreadable when it is
    merely the request that was unsupported -- and a client coalescing two
    chunk reads into one multi-range request would take that at face value.

    416 is reserved for a well-formed range that genuinely names nothing.

    Parameters
    ----------
    range_header : str
        The value of the Range header, e.g. ``"bytes=0-99"`` or ``"bytes=-100"``.

    Returns
    -------
    ByteRequest or _RangeVerdict
        A ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``
        for a readable range, otherwise the verdict to apply.
    """
    if not range_header.startswith("bytes="):
        # An unrecognized range unit; RFC 9110 §14.2 says MUST ignore.
        return _RangeVerdict.IGNORE
    range_spec = range_header[len("bytes=") :]
    if "," in range_spec:
        # A multipart range. Legal to send, and legal to answer with the whole
        # representation; this server does not build multipart/byteranges.
        return _RangeVerdict.IGNORE

    if range_spec.startswith("-"):
        # suffix request: bytes=-N
        suffix = _parse_int(range_spec[1:])
        if suffix is None:
            return _RangeVerdict.IGNORE
        if suffix == 0:
            # "the last zero bytes" names nothing.
            return _RangeVerdict.UNSATISFIABLE
        return SuffixByteRequest(suffix=min(suffix, _MAX_BYTE_POS))

    parts = range_spec.split("-", 1)
    if len(parts) != 2:
        return _RangeVerdict.IGNORE
    start_str, end_str = parts
    start = _parse_int(start_str)
    if start is None:
        return _RangeVerdict.IGNORE
    if start > _MAX_BYTE_POS:
        # No object can be this long, so the range starts past every end.
        return _RangeVerdict.UNSATISFIABLE
    if end_str == "":
        # offset request: bytes=N-
        return OffsetByteRequest(offset=start)
    end_pos = _parse_int(end_str)
    if end_pos is None:
        return _RangeVerdict.IGNORE
    # HTTP end is inclusive, ByteRequest end is exclusive. A last-byte-pos at
    # or past the end of the object is satisfiable -- RFC 9110 §14.1.2 says to
    # clamp it -- so an oversized bound is capped rather than refused.
    end = min(end_pos, _MAX_BYTE_POS - 1) + 1
    if start >= end:
        # An inverted range like "bytes=5-2" is unsatisfiable, not a read
        # of negative length.
        return _RangeVerdict.UNSATISFIABLE
    return RangeByteRequest(start=start, end=end)


def _is_drive_qualified(path: str) -> bool:
    """Whether *path* carries a drive or UNC prefix that would discard a store root.

    A drive-qualified key such as `C:/Windows`, or the drive-relative `a:b`,
    replaces the root it is joined to rather than extending it. This is
    rejected on every platform, not just Windows: the check is a string-level
    gate in front of an arbitrary `Store`, and this package cannot know how a
    given implementation resolves keys.

    The cost is that a node whose *first* path segment looks like `<x>:...`
    is unreachable -- `ntpath` treats any single character before a colon as
    a drive, so there is no safe subset to admit. Such names are legal but
    rare, and later segments are unaffected (`sub/a:b` is served normally).

    Parameters
    ----------
    path : str
        The candidate key, with separators already folded to `/`.

    Returns
    -------
    bool
    """
    return ntpath.splitdrive(path)[0] != ""


def _names_nothing(exc: OSError) -> bool:
    """Whether an `OSError` answers about the *name*, rather than reporting failure.

    `ENAMETOOLONG` is the store saying no such name is expressible here. That
    is an answer about the key -- nothing can be stored under it, so a miss is
    honest -- and it is unreachable for real data, because `encode_chunk_key`
    never produces a segment near a filesystem's length limit. Answering 404
    therefore cannot make a reader substitute fill values over a chunk that
    exists, and it keeps a client from turning a freely chosen key into a 5xx.

    Every other `errno` describes a failure to complete the operation and must
    surface as one. `EINVAL` was accepted here and was the dangerous case: it
    is POSIX's catch-all, reachable on a perfectly ordinary short key through
    a bad seek or an unsupported filesystem feature. Under the v3 spec an
    absent chunk is an uninitialized one, so reporting such a failure as 404
    has a correct reader write fill values over data that is merely
    unreadable. When in doubt the store's own signal is the one to trust:
    `None` means absent, a raised error means the request could not be
    answered.

    Parameters
    ----------
    exc : OSError
        The error raised by the store.

    Returns
    -------
    bool
    """
    return exc.errno == errno.ENAMETOOLONG


def _content_range(byte_range: RangeByteRequest | OffsetByteRequest, length: int) -> str:
    """Build a `Content-Range` value for a 206 response.

    Parameters
    ----------
    byte_range : RangeByteRequest or OffsetByteRequest
        The range that was served. A suffix request is resolved to an absolute
        range before reaching here, because RFC 9110 §15.3.7 requires every
        single-part 206 to carry a `Content-Range` and a suffix's first-byte
        position is not knowable without the object's size.
    length : int
        The number of bytes actually returned.

    Returns
    -------
    str
        A `bytes <first>-<last>/*` value.
    """
    start = byte_range.start if isinstance(byte_range, RangeByteRequest) else byte_range.offset
    # The total length is unknown here; RFC 9110 permits "*" in its place.
    return f"bytes {start}-{start + length - 1}/*"


async def _resolve_suffix(
    store: Store, path: str, byte_range: SuffixByteRequest
) -> RangeByteRequest | _RangeVerdict:
    """Turn a suffix request into an absolute range using the object's size.

    A suffix range names its bytes relative to an end this server does not
    otherwise need to know. Resolving it here is what lets the 206 carry a
    `Content-Range`, which RFC 9110 requires and which a caller reading a
    shard index needs in order to locate what it was given.
    """
    try:
        size = await store.getsize(path)
    except FileNotFoundError:
        return _RangeVerdict.UNSATISFIABLE
    if size == 0:
        return _RangeVerdict.UNSATISFIABLE
    # A suffix longer than the object is satisfiable and yields the whole
    # object, per RFC 9110 §14.1.2.
    return RangeByteRequest(start=max(0, size - byte_range.suffix), end=size)


_JSON_BASENAMES = (
    array_metadata_keys(2)
    | array_metadata_keys(3)
    | group_metadata_keys(2)
    | group_metadata_keys(3)
)
"""Metadata documents that are JSON, for every zarr format.

Derived from the same tables that decide which keys a node owns, so a v2
array's `.zarray` is typed as JSON rather than as opaque bytes -- and adding a
document in one place cannot leave the media type behind in the other.
"""


def content_type_for(path: str) -> str:
    """Media type for a store key, chosen by its basename."""
    if path.rsplit("/", 1)[-1] in _JSON_BASENAMES:
        return "application/json"
    return "application/octet-stream"


async def _head_response(store: Store, path: str, content_type: str) -> Response:
    """Answer a HEAD without transferring the value.

    A HEAD body is discarded at the wire, so routing HEAD through the GET
    handler reads the whole object -- megabytes of chunk or shard -- to report
    a length. `Store.getsize` is a `stat` on a filesystem store and an
    info/HEAD call on a remote one.
    """
    from starlette.responses import Response

    try:
        size = await store.getsize(path)
    except FileNotFoundError:
        return Response(status_code=404)
    except OSError as exc:
        if not _names_nothing(exc):
            raise
        return Response(status_code=404)
    return Response(status_code=200, media_type=content_type, headers={"Content-Length": str(size)})


async def _get_response(
    store: Store,
    path: str,
    byte_range: RangeByteRequest | OffsetByteRequest | None = None,
) -> Response:
    """Fetch a key from the store and return an HTTP response."""
    from starlette.responses import Response

    proto = cpu.buffer_prototype
    content_type = content_type_for(path)

    try:
        buf = await store.get(path, proto, byte_range=byte_range)
    except (MemoryError, OverflowError):
        # The client's last-byte-pos is wider than the store can materialize
        # -- it sizes its read from the range, not from the object. RFC 9110
        # §14.1.2 makes a last-byte-pos at or past the end of the object
        # satisfiable and clamps it to the end, so re-read from the same start
        # to EOF, which is what the clamped range denotes. Refusing with 416
        # would deny a request that is merely over-wide, and a client asking
        # for "from here to well past the end" is asking a normal question.
        if not isinstance(byte_range, RangeByteRequest):
            raise
        byte_range = OffsetByteRequest(offset=byte_range.start)
        buf = await store.get(path, proto, byte_range=byte_range)
    except OSError as exc:
        if not _names_nothing(exc):
            # A real I/O failure, which must not be reported as a miss. Under
            # the v3 spec an absent chunk is an uninitialized one, and a
            # reader is right to substitute the array's fill value for it --
            # so 404 asserts something about the store's contents. An
            # unreadable chunk is not an uninitialized chunk, and answering
            # 404 would have a correct client silently materialize fill
            # values over data that exists.
            raise
        return Response(status_code=404)
    if buf is None:
        return Response(status_code=404)

    if byte_range is None:
        return Response(content=buf.to_bytes(), status_code=200, media_type=content_type)

    body = buf.to_bytes()
    if len(body) == 0:
        # The range lies wholly beyond the end of the object.
        return Response(status_code=416)

    headers = {"Content-Range": _content_range(byte_range, len(body))}
    return Response(content=body, status_code=206, media_type=content_type, headers=headers)


async def _handle_request(request: Request) -> Response:
    """Handle a request, optionally filtering by node validity."""
    from starlette.responses import Response

    store: Store = request.app.state.store
    node: Array[Any] | Group | None = request.app.state.node
    prefix: str = request.app.state.prefix
    path = request.path_params.get("path", "")

    # Reject non-canonical / traversal-prone keys before touching the store.
    # Starlette percent-decodes path params, so "..%2f" arrives as a literal
    # ".." segment and "%2f"-encoded leading slashes arrive as an empty leading
    # segment (making the key absolute, which escapes a filesystem store root).
    # Backslashes are separators on Windows and drive-qualified or
    # root-relative keys discard a filesystem store's root entirely, so fold
    # separators before checking segments -- mirroring zarr's normalize_path
    # (src/zarr/storage/_utils.py) plus a drive check it doesn't need. Legitimate
    # zarr keys never contain empty, ".", or ".." segments, or a drive letter.
    segments = path.replace("\\", "/").split("/")
    if any(segment in ("", ".", "..") for segment in segments) or _is_drive_qualified(path):
        return Response(status_code=404)

    # A NUL can never appear in a store key, and reaches the filesystem layer
    # as a raised error rather than a miss. Rejecting it here keeps that out
    # of the store, so the store's own errors always mean real I/O trouble.
    if "\x00" in path:
        return Response(status_code=404)

    # If serving a node, validate the key before touching the store. Group
    # validation opens children through zarr's synchronous API, which drives
    # the store to completion and would otherwise block the event loop for
    # the duration -- serializing every concurrent request behind it, and
    # outlasting the shutdown timeout.
    if node is not None and not await asyncio.to_thread(is_valid_node_key, node, path):
        return Response(status_code=404)

    # Resolve the full store key by prepending the node's prefix.
    store_key = f"{prefix}/{path}" if prefix else path

    if request.method == "PUT":
        if store.read_only:
            # The store will refuse this with a ValueError from deep inside
            # `set`, which is a 500 -- a server fault. Refusing to write to a
            # read-only store is not a fault, it is the answer.
            return Response(status_code=403)

        max_body_size: int | None = request.app.state.max_body_size
        if max_body_size is None:
            body = await request.body()
        else:
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > max_body_size:
                return Response(status_code=413)

            # Read incrementally and stop at the cap. `request.body()` would
            # buffer the whole body first, which a chunked request can use to
            # exceed the cap by any amount before it is ever checked.
            chunks: list[bytes] = []
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > max_body_size:
                    return Response(status_code=413)
                chunks.append(chunk)
            body = b"".join(chunks)

        buf = cpu.buffer_prototype.buffer.from_bytes(body)
        try:
            await store.set(store_key, buf)
        except OSError as exc:
            if not _names_nothing(exc):
                # A real write failure -- a full disk, a read-only mount, a
                # permissions problem. Reporting it as 404 would tell the
                # client the write is pointless rather than failed.
                raise
            return Response(status_code=404)
        return Response(status_code=204)

    if request.method == "HEAD":
        # A HEAD body is discarded at the wire, so reading the value to build
        # one transfers the whole object to answer a question about its size.
        # `getsize` is a stat on a filesystem store and a HEAD/info call on a
        # remote one.
        return await _head_response(store, store_key, content_type_for(path))

    range_header = request.headers.get("range")
    byte_range: RangeByteRequest | OffsetByteRequest | None = None
    if range_header is not None:
        parsed = _parse_range_header(range_header)
        if parsed is _RangeVerdict.UNSATISFIABLE:
            return Response(status_code=416)
        if isinstance(parsed, SuffixByteRequest):
            parsed = await _resolve_suffix(store, store_key, parsed)
            if parsed is _RangeVerdict.UNSATISFIABLE:
                return Response(status_code=416)
        # _RangeVerdict.IGNORE falls through with byte_range still None, which
        # serves the full representation with 200.
        if not isinstance(parsed, _RangeVerdict):
            byte_range = parsed

    return await _get_response(store, store_key, byte_range)


def _make_starlette_app(
    *,
    methods: AbstractSet[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette app with the request handler.

    Raises
    ------
    ValueError
        If `methods` contains anything outside `GET`, `PUT`, and `HEAD`.
    """
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Route

    if methods is None:
        methods = READ_ONLY_HTTP_METHODS

    # An empty set must not reach Starlette: `Route` treats a falsy `methods`
    # as "match every method", so asking for no methods would serve them all.
    if not methods:
        raise ValueError(
            "methods must name at least one HTTP method; "
            f"accepted methods are {', '.join(sorted(_SUPPORTED_METHODS))}."
        )

    unsupported = sorted(set(methods) - _SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(
            f"Unsupported HTTP method(s): {', '.join(unsupported)}. "
            f"Accepted methods are {', '.join(sorted(_SUPPORTED_METHODS))}."
        )

    app = Starlette(
        routes=[Route("/{path:path}", _handle_request, methods=list(methods))],
    )

    if cors_options is not None:
        # Typed loosely on purpose: unpacking a merged TypedDict loses the
        # per-key types, so the looseness is contained to this block rather
        # than spread across casts at each use.
        merged: dict[str, Any] = {**_CORS_DEFAULTS, **cors_options}
        if "allow_methods" in merged:
            # Only when the caller said something. Absent, Starlette's own
            # `GET`-only default stands: widening it to everything served
            # would newly advertise `PUT` cross-origin on a write-enabled app
            # that never asked for it.
            merged["allow_methods"] = _reconcile_allow_methods(
                merged["allow_methods"], served=_served_methods(methods)
            )
        app.add_middleware(
            CORSMiddleware,
            # Our defaults first, so a key the caller supplied wins outright
            # rather than being merged into -- an explicit `expose_headers: []`
            # means "expose nothing", not "expose our default".
            **merged,
        )
    return app


def _reject_writes_to_a_read_only_store(
    store: Store, methods: AbstractSet[HTTPMethod] | None
) -> None:
    """Refuse a configuration whose writes can never succeed.

    A store's `read_only` is fixed when it is built, so asking to serve `PUT`
    from one is a contradiction that would only reveal itself as a 403 on the
    first write a client attempts -- possibly long after deployment, and to
    the client rather than to whoever misconfigured it. Saying so at
    construction matches how unsupported `methods` and contradictory
    `cors_options` are already handled.

    Raises
    ------
    ValueError
        If `methods` asks for `PUT` on a read-only store.
    """
    if methods is not None and "PUT" in methods and store.read_only:
        raise ValueError(
            "methods asks for PUT, but the store is read-only, so no write "
            "could ever succeed. Drop PUT to serve reads, or pass a writable "
            "store (`store.with_read_only(False)`)."
        )


def _served_methods(methods: AbstractSet[HTTPMethod]) -> frozenset[str]:
    """The methods the route will actually answer.

    Starlette adds `HEAD` to any route that serves `GET`, which RFC 9110
    §9.3.2 asks of every origin server, so `HEAD` is served whenever `GET` is
    whether or not it was named.
    """
    served = set(methods)
    if "GET" in served:
        served.add("HEAD")
    return frozenset(served)


def _reconcile_allow_methods(allow_methods: list[str], *, served: frozenset[str]) -> list[str]:
    """Check `cors_options["allow_methods"]` against what the route serves.

    An advertised method the route rejects is a promise the server cannot
    keep: a browser caches the preflight and every later cross-origin call
    fails with 405 after a successful handshake. The reverse is worse -- the
    same silence lets `allow_methods=["*"]` on a write-enabled app hand every
    origin on the internet write access, which is exactly the footgun the
    `methods` validation above exists to prevent.

    `"*"` expands to what is actually served rather than being rejected: it
    is the idiomatic spelling of "everything this app does", and the app
    cannot do more than it serves.
    """
    if "*" in allow_methods:
        return sorted(served)

    unserved = sorted(set(allow_methods) - served)
    if unserved:
        raise ValueError(
            f"cors_options['allow_methods'] advertises {', '.join(unserved)}, "
            f"which this app does not serve (it serves {', '.join(sorted(served))}). "
            "A browser would cache that preflight and every such request would "
            "then fail with 405."
        )
    return list(allow_methods)


def _build_server(
    app: Starlette,
    *,
    host: str,
    port: int | Literal["auto"],
    shutdown_timeout: int,
    uvicorn_options: Mapping[str, object] | None,
) -> tuple[uvicorn.Server, dict[str, object], socket.socket | None]:
    """Configure a `uvicorn.Server` for *app*, and report how it will bind.

    Returns the options actually used -- a key from `uvicorn_options` may have
    replaced one passed here -- and, for `port="auto"`, the socket already
    bound on the caller's behalf, which must be handed to `Server.run`.
    """
    import uvicorn

    options: dict[str, object] = {
        "host": host,
        "port": port,
        "timeout_graceful_shutdown": shutdown_timeout,
    }
    if uvicorn_options is not None:
        options.update(uvicorn_options)

    sock: socket.socket | None = None
    if options.get("port") == AUTO_PORT:
        if _binds_without_a_port(options):
            # A uds or fd bind ignores host and port; leave a valid int in
            # place of the sentinel so `Config` still type-checks.
            options["port"] = DEFAULT_PORT
        else:
            # Bind here rather than probing and handing uvicorn a port number:
            # probing would release the port before uvicorn claimed it, which
            # is the bind-then-close race that makes "find a free port" helpers
            # flaky. Holding the socket means nothing can take it in between.
            sock = _bind_preferred_or_free(str(options["host"]), DEFAULT_PORT)
            # Keep Config agreeing with reality, so uvicorn's own "running on
            # ..." line names the port it is really serving.
            options["port"] = sock.getsockname()[1]

    # uvicorn.Config's parameters are individually typed and there are ~50 of
    # them; `Mapping[str, object]` is the honest type for the public argument,
    # so the cast is confined to the call itself.
    return uvicorn.Server(uvicorn.Config(app, **cast("dict[str, Any]", options))), options, sock


def _binds_without_a_port(options: Mapping[str, object]) -> bool:
    """Whether these options bind something other than a TCP host and port."""
    return options.get("uds") is not None or options.get("fd") is not None


def _bind_preferred_or_free(host: str, preferred: int) -> socket.socket:
    """Bind *preferred* on *host* if it is free, otherwise any free port.

    The address family comes from `getaddrinfo` rather than being assumed:
    hard-coding `AF_INET` would fail for an IPv6 host such as ``"::1"``.
    """
    for candidate in (preferred, 0):
        family, socktype, proto, _, sockaddr = socket.getaddrinfo(
            host, candidate, type=socket.SOCK_STREAM
        )[0]
        sock = socket.socket(family, socktype, proto)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(sockaddr)
        except OSError:
            sock.close()
            if candidate == 0:
                # Nothing is free, which is a real failure rather than a
                # reason to keep looking.
                raise
            _LOGGER.info(
                "port %d is in use; binding a free port instead. Pass an "
                "explicit port to require a particular one.",
                preferred,
            )
            continue
        return sock
    raise AssertionError("unreachable")  # pragma: no cover


def serve(
    app: Starlette,
    *,
    host: str = "127.0.0.1",
    port: int | Literal["auto"] = AUTO_PORT,
    shutdown_timeout: int = 5,
    uvicorn_options: Mapping[str, object] | None = None,
) -> None:
    """Run an ASGI app under Uvicorn, blocking until it is stopped.

    Returns only when the server stops, so this is the shape for a process
    whose job is to serve -- a script, a container entrypoint. Use
    :func:`serve_background` when the caller has more to do.

    Build the app first with :func:`store_app` or :func:`node_app`, or compose
    several of them; what an app serves is settled when it is built, and this
    only decides how it runs.

    .. code-block:: python

        serve(store_app(store), host="0.0.0.0", port=8000)

    Parameters
    ----------
    app : Starlette
        The ASGI app to run.
    host : str, optional
        The host to bind to. Defaults to ``"127.0.0.1"``.
    port : int, optional
        The port to bind to. Defaults to ``8000``.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully when the
        server is shut down, via uvicorn's ``timeout_graceful_shutdown``.
        Defaults to ``5``.
    uvicorn_options : Mapping[str, object], optional
        Extra options passed straight to `uvicorn.Config`, merged over the
        ones set here (`host`, `port`, `timeout_graceful_shutdown`), so a
        caller key wins. This is the escape hatch for anything uvicorn can do
        that this signature does not name -- TLS via `ssl_keyfile` /
        `ssl_certfile`, `proxy_headers` and `forwarded_allow_ips` behind a
        reverse proxy, `root_path` when mounted under a prefix, `log_level`,
        `limit_concurrency`, or a `uds` / `fd` bind.
    """
    server, _, sock = _build_server(
        app,
        host=host,
        port=port,
        shutdown_timeout=shutdown_timeout,
        uvicorn_options=uvicorn_options,
    )
    server.run(sockets=[sock] if sock is not None else None)


def serve_background(
    app: Starlette,
    *,
    host: str = "127.0.0.1",
    port: int | Literal["auto"] = AUTO_PORT,
    shutdown_timeout: int = 5,
    uvicorn_options: Mapping[str, object] | None = None,
) -> BackgroundServer:
    """Start an ASGI app under Uvicorn in a daemon thread and return at once.

    Returns once the socket is listening, so the next statement can use the
    server. The returned handle is also a context manager:

    .. code-block:: python

        with serve_background(node_app(array)) as server:
            httpx.get(f"{server.url}/zarr.json")

    In a notebook, where the server must outlive the cell that started it,
    keep the handle instead and call :meth:`BackgroundServer.shutdown` later.

    Parameters
    ----------
    app : Starlette
        The ASGI app to run.
    host : str, optional
        The host to bind to. Defaults to ``"127.0.0.1"``.
    port : int, optional
        The port to bind to. Defaults to ``0``, which asks the OS for a free
        one -- unlike :func:`serve`, whose caller usually needs a port others
        already know. A background server is normally reached through
        :attr:`BackgroundServer.url`, and a fixed default would make starting
        a second one, or re-running a notebook cell, fail on a port collision.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully, both for
        uvicorn's ``timeout_graceful_shutdown`` and for the wait
        :meth:`BackgroundServer.shutdown` uses before forcing the thread
        closed. Defaults to ``5``.
    uvicorn_options : Mapping[str, object], optional
        Extra options passed straight to `uvicorn.Config`, merged over the
        ones set here so a caller key wins. See :func:`serve`. Binding
        somewhere other than a TCP host and port leaves
        :attr:`BackgroundServer.url` as ``None``.

    Returns
    -------
    BackgroundServer
        A handle for the running server.

    Raises
    ------
    RuntimeError
        If the server does not start -- most often because the port is
        already in use.
    """
    server, options, sock = _build_server(
        app,
        host=host,
        port=port,
        shutdown_timeout=shutdown_timeout,
        uvicorn_options=uvicorn_options,
    )

    # uvicorn skips signal-handler installation off the main thread
    # (Server.capture_signals), so no workaround is needed here.
    thread = threading.Thread(
        target=partial(server.run, sockets=[sock] if sock is not None else None), daemon=True
    )
    thread.start()

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while not server.started:
        if not thread.is_alive():
            # uvicorn logs the underlying error and calls sys.exit, which in a
            # thread ends it without surfacing anything to the caller. The
            # overwhelmingly common cause is a port already in use.
            raise RuntimeError(
                f"Server thread exited before startup completed; {host}:{port} "
                "may already be in use. See the server log for the cause."
            )
        if time.monotonic() > deadline:
            # The thread is still alive here, unlike the branch above, and it
            # is about to finish starting. Raising without stopping it would
            # leave a server bound to the port with no handle to shut it down
            # -- a daemon thread serving for the rest of the process, and a
            # retry on the same port failing with the other error above.
            server.should_exit = True
            server.force_exit = True
            thread.join(timeout=_STARTUP_ABANDON_TIMEOUT)
            raise RuntimeError(
                f"Server failed to start within {_STARTUP_TIMEOUT:g} seconds; "
                "it has been signalled to stop."
            )
        time.sleep(0.01)

    # Report the port the socket actually bound rather than the one asked for,
    # so `port=0` ("pick a free port") yields a usable `url`. A unix-socket or
    # file-descriptor bind has no host and port at all, so both stay None and
    # `url` reports None rather than naming an address nothing is listening on.
    bound_port: int | None = None
    for bound in server.servers:
        for sock in bound.sockets:
            sockname = sock.getsockname()
            if isinstance(sockname, tuple) and len(sockname) >= 2:
                bound_port = int(sockname[1])
            break
        break

    # The host is taken from the request rather than the socket: a wildcard
    # bind reports "0.0.0.0", which is not an address a client can connect to.
    requested_host = options.get("host")
    bound_host = (
        str(requested_host) if bound_port is not None and requested_host is not None else None
    )

    return BackgroundServer(
        server,
        thread,
        host=bound_host,
        port=bound_port,
        scheme="https" if server.config.is_ssl else "http",
        shutdown_timeout=shutdown_timeout,
    )


def store_app(
    store: Store,
    *,
    methods: AbstractSet[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
    max_body_size: int | None = DEFAULT_MAX_BODY_SIZE,
) -> Starlette:
    """Create a Starlette ASGI app that serves every key in a zarr ``Store``.

    Parameters
    ----------
    store : Store
        The zarr store to serve.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept: any of `"GET"`, `"HEAD"`, and `"PUT"`.
        Defaults to `{"GET"}`, which also serves `HEAD`. Passing any other
        method raises `ValueError`.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.
    max_body_size : int or None, optional
        Largest `PUT` body to accept, in bytes; larger requests get a 413.
        `Store.set` takes a whole `Buffer`, so bodies cannot be streamed and
        are held in memory in full. Defaults to `DEFAULT_MAX_BODY_SIZE`;
        pass `None` to lift the cap.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    _reject_writes_to_a_read_only_store(store, methods)
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = store
    app.state.node = None
    app.state.prefix = ""
    app.state.max_body_size = max_body_size
    return app


def node_app(
    node: Array[Any] | Group,
    *,
    methods: AbstractSet[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
    max_body_size: int | None = DEFAULT_MAX_BODY_SIZE,
) -> Starlette:
    """Create a Starlette ASGI app that serves only the keys belonging to a
    zarr ``Array`` or ``Group``.

    For an ``Array``, the served keys are the metadata document(s) and all
    chunk (or shard) keys whose coordinates fall within the array's grid.

    For a ``Group``, the served keys are the group's own metadata plus any
    path that resolves through the group's members to a valid array metadata
    document or chunk key.

    Requests for keys outside this set receive a 404 response, even if the
    underlying store contains data at that path.

    Parameters
    ----------
    node : Array or Group
        The zarr array or group to serve.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept: any of `"GET"`, `"HEAD"`, and `"PUT"`.
        Defaults to `{"GET"}`, which also serves `HEAD`. Passing any other
        method raises `ValueError`.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.
    max_body_size : int or None, optional
        Largest `PUT` body to accept, in bytes; larger requests get a 413.
        `Store.set` takes a whole `Buffer`, so bodies cannot be streamed and
        are held in memory in full. Defaults to `DEFAULT_MAX_BODY_SIZE`;
        pass `None` to lift the cap.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    _reject_writes_to_a_read_only_store(node.store_path.store, methods)
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = node.store_path.store
    app.state.node = node
    app.state.prefix = node.store_path.path
    app.state.max_body_size = max_body_size
    return app
