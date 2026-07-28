from __future__ import annotations

import ntpath
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, overload

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.buffer import cpu

from zarr_server._keys import is_valid_node_key

if TYPE_CHECKING:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from zarr import Array, Group
    from zarr.abc.store import ByteRequest, Store

__all__ = [
    "BackgroundServer",
    "CorsOptions",
    "HTTPMethod",
    "node_app",
    "serve_node",
    "serve_store",
    "store_app",
]


class CorsOptions(TypedDict):
    allow_origins: list[str]
    allow_methods: list[str]


HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


class BackgroundServer:
    """A running background HTTP server that can be used as a context manager.

    Wraps a ``uvicorn.Server`` running in a daemon thread.  When used as a
    context manager the server is shut down automatically on exit.

    Parameters
    ----------
    server : uvicorn.Server
        The running uvicorn server instance.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully during
        :meth:`shutdown` before forcing the server closed. Defaults to ``5``.

    Examples
    --------
    >>> with serve_node(arr, background=True) as server:  # doctest: +SKIP
    ...     print(f"Listening on {server.host}:{server.port}")
    ...     # server is shut down when the block exits
    """

    def __init__(
        self,
        server: uvicorn.Server,
        thread: threading.Thread,
        *,
        host: str,
        port: int,
        shutdown_timeout: int = 5,
    ) -> None:
        self._server = server
        self._thread = thread
        self.host = host
        self.port = port
        self._shutdown_timeout = shutdown_timeout

    @property
    def url(self) -> str:
        """The base URL of the running server."""
        return f"http://{self.host}:{self.port}"

    def shutdown(self) -> None:
        """Signal the server to shut down and wait for it to stop.

        Waits up to ``shutdown_timeout`` seconds (set when the server was
        started) for the server thread to exit on its own -- uvicorn's own
        graceful-shutdown wait for in-flight requests is bounded by the same
        value via ``timeout_graceful_shutdown``. If the thread is still
        alive after that, ``force_exit`` is set on the underlying uvicorn
        server to tear it down immediately rather than block forever on a
        stuck or slow in-flight request.
        """
        self._server.should_exit = True
        self._thread.join(timeout=self._shutdown_timeout)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()


def _parse_range_header(range_header: str) -> ByteRequest | None:
    """Parse an HTTP Range header into a ByteRequest.

    Parameters
    ----------
    range_header : str
        The value of the Range header, e.g. ``"bytes=0-99"`` or ``"bytes=-100"``.

    Returns
    -------
    ByteRequest or None
        A ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``,
        or ``None`` if the header cannot be parsed.
    """
    if not range_header.startswith("bytes="):
        return None
    range_spec = range_header[len("bytes=") :]
    try:
        if range_spec.startswith("-"):
            # suffix request: bytes=-N
            suffix = int(range_spec[1:])
            return SuffixByteRequest(suffix=suffix)
        parts = range_spec.split("-", 1)
        if len(parts) != 2:
            return None
        start_str, end_str = parts
        start = int(start_str)
        if end_str == "":
            # offset request: bytes=N-
            return OffsetByteRequest(offset=start)
        # range request: bytes=N-M (HTTP end is inclusive, ByteRequest end is exclusive)
        end = int(end_str) + 1
        return RangeByteRequest(start=start, end=end)
    except ValueError:
        return None


async def _get_response(store: Store, path: str, byte_range: ByteRequest | None = None) -> Response:
    """Fetch a key from the store and return an HTTP response."""
    from starlette.responses import Response

    proto = cpu.buffer_prototype
    content_type = "application/json" if path.endswith("zarr.json") else "application/octet-stream"

    buf = await store.get(path, proto, byte_range=byte_range)
    if buf is None:
        return Response(status_code=404)

    status_code = 206 if byte_range is not None else 200
    return Response(content=buf.to_bytes(), status_code=status_code, media_type=content_type)


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
    if any(segment in ("", ".", "..") for segment in segments) or ntpath.splitdrive(path)[0]:
        return Response(status_code=404)

    # If serving a node, validate the key before touching the store.
    if node is not None and not is_valid_node_key(node, path):
        return Response(status_code=404)

    # Resolve the full store key by prepending the node's prefix.
    store_key = f"{prefix}/{path}" if prefix else path

    if request.method == "PUT":
        body = await request.body()
        buf = cpu.buffer_prototype.buffer.from_bytes(body)
        await store.set(store_key, buf)
        return Response(status_code=204)

    range_header = request.headers.get("range")
    byte_range: ByteRequest | None = None
    if range_header is not None:
        byte_range = _parse_range_header(range_header)
        if byte_range is None:
            return Response(status_code=416)

    return await _get_response(store, store_key, byte_range)


def _make_starlette_app(
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette app with the request handler."""
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Route

    if methods is None:
        methods = {"GET"}

    app = Starlette(
        routes=[Route("/{path:path}", _handle_request, methods=list(methods))],
    )

    if cors_options is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_options["allow_origins"],
            allow_methods=cors_options["allow_methods"],
        )
    return app


def _start_server(
    app: Starlette,
    *,
    host: str,
    port: int,
    background: bool,
    shutdown_timeout: int = 5,
) -> BackgroundServer | None:
    """Create a uvicorn server for *app* and either block or run in a daemon thread.

    ``shutdown_timeout`` bounds uvicorn's own graceful-shutdown wait for
    in-flight requests (``timeout_graceful_shutdown``), and, for a
    background server, is also the bound :meth:`BackgroundServer.shutdown`
    uses before forcing the server thread closed.
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, timeout_graceful_shutdown=shutdown_timeout)
    server = uvicorn.Server(config)

    if not background:
        server.run()
        return None

    # uvicorn skips signal-handler installation off the main thread
    # (Server.capture_signals), so no workaround is needed here.
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Server failed to start within 5 seconds")
        time.sleep(0.01)

    return BackgroundServer(server, thread, host=host, port=port, shutdown_timeout=shutdown_timeout)


def store_app(
    store: Store,
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette ASGI app that serves every key in a zarr ``Store``.

    Parameters
    ----------
    store : Store
        The zarr store to serve.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = store
    app.state.node = None
    app.state.prefix = ""
    return app


def node_app(
    node: Array[Any] | Group,
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
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
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = node.store_path.store
    app.state.node = node
    app.state.prefix = node.store_path.path
    return app


@overload
def serve_store(
    store: Store,
    *,
    host: str = ...,
    port: int = ...,
    methods: set[HTTPMethod] | None = ...,
    cors_options: CorsOptions | None = ...,
    background: Literal[False] = ...,
    shutdown_timeout: int = ...,
) -> None: ...


@overload
def serve_store(
    store: Store,
    *,
    host: str = ...,
    port: int = ...,
    methods: set[HTTPMethod] | None = ...,
    cors_options: CorsOptions | None = ...,
    background: Literal[True],
    shutdown_timeout: int = ...,
) -> BackgroundServer: ...


def serve_store(
    store: Store,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
    background: bool = False,
    shutdown_timeout: int = 5,
) -> BackgroundServer | None:
    """Serve every key in a zarr ``Store`` over HTTP.

    Builds a Starlette ASGI app (see :func:`store_app`) and starts a
    `Uvicorn <https://www.uvicorn.org/>`_ server.

    Parameters
    ----------
    store : Store
        The zarr store to serve.
    host : str, optional
        The host to bind to. Defaults to ``"127.0.0.1"``.
    port : int, optional
        The port to bind to. Defaults to ``8000``.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.
    background : bool, optional
        If ``False`` (the default), the server blocks until shut down.
        If ``True``, the server runs in a daemon thread and this function
        returns immediately.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully before
        the server is forced closed, whether on `BackgroundServer.shutdown`
        or (for a blocking server) on receiving a shutdown signal. Defaults
        to ``5``.

    Returns
    -------
    BackgroundServer or None
        The running server when ``background=True``, or ``None`` when
        the server has been shut down after blocking.  The
        ``BackgroundServer`` can be used as a context manager for
        automatic shutdown.
    """
    app = store_app(store, methods=methods, cors_options=cors_options)
    return _start_server(
        app, host=host, port=port, background=background, shutdown_timeout=shutdown_timeout
    )


@overload
def serve_node(
    node: Array[Any] | Group,
    *,
    host: str = ...,
    port: int = ...,
    methods: set[HTTPMethod] | None = ...,
    cors_options: CorsOptions | None = ...,
    background: Literal[False] = ...,
    shutdown_timeout: int = ...,
) -> None: ...


@overload
def serve_node(
    node: Array[Any] | Group,
    *,
    host: str = ...,
    port: int = ...,
    methods: set[HTTPMethod] | None = ...,
    cors_options: CorsOptions | None = ...,
    background: Literal[True],
    shutdown_timeout: int = ...,
) -> BackgroundServer: ...


def serve_node(
    node: Array[Any] | Group,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
    background: bool = False,
    shutdown_timeout: int = 5,
) -> BackgroundServer | None:
    """Serve only the keys belonging to a zarr ``Array`` or ``Group`` over HTTP.

    Builds a Starlette ASGI app (see :func:`node_app`) and starts a
    `Uvicorn <https://www.uvicorn.org/>`_ server.

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
    host : str, optional
        The host to bind to. Defaults to ``"127.0.0.1"``.
    port : int, optional
        The port to bind to. Defaults to ``8000``.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.
    background : bool, optional
        If ``False`` (the default), the server blocks until shut down.
        If ``True``, the server runs in a daemon thread and this function
        returns immediately.
    shutdown_timeout : int, optional
        Seconds to wait for in-flight requests to finish gracefully before
        the server is forced closed, whether on `BackgroundServer.shutdown`
        or (for a blocking server) on receiving a shutdown signal. Defaults
        to ``5``.

    Returns
    -------
    BackgroundServer or None
        The running server when ``background=True``, or ``None`` when
        the server has been shut down after blocking.  The
        ``BackgroundServer`` can be used as a context manager for
        automatic shutdown.
    """
    app = node_app(node, methods=methods, cors_options=cors_options)
    return _start_server(
        app, host=host, port=port, background=background, shutdown_timeout=shutdown_timeout
    )
