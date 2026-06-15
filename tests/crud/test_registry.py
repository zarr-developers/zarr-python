from __future__ import annotations

import pytest

from zarr.crud import CrudBackend, NodeExistsError, get_backend, register_backend


def test_node_exists_error_is_value_error() -> None:
    assert issubclass(NodeExistsError, ValueError)


def test_default_backend_is_reference() -> None:
    # the reference backend is registered at import and is the configured default
    be = get_backend()
    assert be is get_backend("reference")


def test_get_unknown_backend_raises() -> None:
    with pytest.raises(KeyError, match="no CRUD backend"):
        get_backend("does-not-exist")


def test_register_and_resolve_instance() -> None:
    class Dummy:
        pass

    dummy = Dummy()
    register_backend("dummy-test", dummy)  # type: ignore[arg-type]
    try:
        assert get_backend("dummy-test") is dummy  # type: ignore[comparison-overlap]
    finally:
        from zarr.crud import _registry

        _registry._BACKENDS.pop("dummy-test", None)


def test_protocol_is_runtime_checkable() -> None:
    # ReferenceBackend (registered as "reference") structurally satisfies the protocol
    assert isinstance(get_backend("reference"), CrudBackend)
