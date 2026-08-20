from __future__ import annotations

import pytest

from zarr.registry import (
    _CODEC_PACKAGE_PREFIXES,
    _CODEC_PACKAGES,
    _NUMCODEC_PACKAGE_PREFIXES,
    _missing_codec_message,
    _packages_for_codec,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("n5_default", ("zarr-n5",)),
        ("gribberish", ("gribberish",)),
        ("imagecodecs_jpeg2k", ("virtual-tiff",)),
        ("omfiles.pfor", ("omfiles",)),
        ("any-numcodecs.array-array", ("zarr-any-numcodecs",)),
        ("totally-made-up", ()),
    ],
)
def test_packages_for_codec_v3(name: str, expected: tuple[str, ...]) -> None:
    """Exact names and prefixes both resolve; unknown names resolve to nothing."""
    assert _packages_for_codec(name, zarr_format=3) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("wavpack", ("wavpack-numcodecs",)),
        ("grib", ("kerchunk",)),
        ("zfpy", ("numcodecs[zfpy]",)),
        ("crc32c", ("numcodecs[crc32c]",)),
        ("gribscan.rawgrib", ("gribscan",)),
        ("imagecodecs_jpeg2k", ("imagecodecs-numcodecs",)),
        ("totally-made-up", ()),
    ],
)
def test_packages_for_numcodec_v2(name: str, expected: tuple[str, ...]) -> None:
    """Zarr format 2 codec ids resolve against the numcodecs table."""
    assert _packages_for_codec(name, zarr_format=2) == expected


def test_packages_for_codec_is_format_specific() -> None:
    """The same name can mean different packages in each format's registry."""
    assert _packages_for_codec("imagecodecs_jpeg2k", zarr_format=3) == ("virtual-tiff",)
    assert _packages_for_codec("imagecodecs_jpeg2k", zarr_format=2) == ("imagecodecs-numcodecs",)
    # `crc32c` is a codec zarr implements in format 3, so only format 2 gets a hint for it.
    assert _packages_for_codec("crc32c", zarr_format=3) == ()


def test_missing_codec_message_with_known_packages() -> None:
    """The message names the codec, the docs page, and every known package."""
    msg = _missing_codec_message("n5_default", zarr_format=3)
    assert "'n5_default'" in msg
    assert "extending" in msg
    assert "registers a codec implementation with zarr." in msg
    assert "Known packages supporting this codec: zarr-n5" in msg


def test_missing_codec_message_without_known_packages() -> None:
    """With no known package we still explain how to register one by hand."""
    msg = _missing_codec_message("totally-made-up", zarr_format=3)
    assert "'totally-made-up'" in msg
    assert "Known packages" not in msg


def test_missing_codec_message_for_zarr_format_2() -> None:
    """Format 2 codecs live in the numcodecs registry, so the message must say so."""
    msg = _missing_codec_message("wavpack", zarr_format=2)
    assert "numcodecs.readthedocs.io" in msg
    assert "registers a codec implementation with numcodecs." in msg
    assert "Known packages supporting this codec: wavpack-numcodecs" in msg


def test_no_prefix_shadows_another_prefix() -> None:
    """First-match prefix lookup is only deterministic while no prefix contains another."""
    for table in (_CODEC_PACKAGE_PREFIXES, _NUMCODEC_PACKAGE_PREFIXES):
        for a in table:
            for b in table:
                assert a == b or not a.startswith(b)


def test_mapping_does_not_shadow_builtin_codecs() -> None:
    """A codec zarr implements itself must never appear in the format 3 hint table."""
    import zarr.codecs  # noqa: F401  (importing registers the built-in codecs)
    from zarr.registry import _codec_registries

    implemented = {name for name, reg in _codec_registries.items() if len(reg) > 0}
    assert implemented, "expected importing zarr.codecs to populate the registry"
    assert not (implemented & set(_CODEC_PACKAGES))
    for prefix in _CODEC_PACKAGE_PREFIXES:
        assert not any(name.startswith(prefix) for name in implemented)


def test_get_codec_class_unknown_raises_with_package_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered codec with a known package names that package in the error."""
    from collections import defaultdict

    import zarr.registry
    from zarr.errors import UnknownCodecError
    from zarr.registry import Registry, get_codec_class

    monkeypatch.setattr(zarr.registry, "_codec_registries", defaultdict(Registry))
    with pytest.raises(UnknownCodecError, match="Known packages supporting this codec: zarr-n5"):
        get_codec_class("n5_default")


def test_get_codec_class_unknown_raises_without_package_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered codec we know nothing about still explains manual registration."""
    from collections import defaultdict

    import zarr.registry
    from zarr.errors import UnknownCodecError
    from zarr.registry import Registry, get_codec_class

    monkeypatch.setattr(zarr.registry, "_codec_registries", defaultdict(Registry))
    with pytest.raises(UnknownCodecError, match="An implementation for codec 'nope' is not"):
        get_codec_class("nope")


def test_unknown_codec_error_is_exported() -> None:
    """UnknownCodecError is public API now that we tell users to catch it."""
    import zarr.errors

    assert "UnknownCodecError" in zarr.errors.__all__


def test_get_numcodec_unknown_raises_with_package_hint() -> None:
    """A Zarr format 2 codec id we don't have names the package that provides it."""
    from zarr.errors import UnknownCodecError
    from zarr.registry import get_numcodec

    with pytest.raises(
        UnknownCodecError, match="Known packages supporting this codec: wavpack-numcodecs"
    ):
        get_numcodec({"id": "wavpack"})


def test_get_numcodec_unknown_points_at_numcodecs_registry() -> None:
    """The Zarr format 2 message links the numcodecs registry, not zarr's extending guide."""
    from zarr.errors import UnknownCodecError
    from zarr.registry import get_numcodec

    with pytest.raises(UnknownCodecError, match="numcodecs.readthedocs.io"):
        get_numcodec({"id": "definitely-not-a-real-codec"})


def test_get_numcodec_known_codec_still_works() -> None:
    """The happy path is untouched."""
    from numcodecs import GZip

    from zarr.registry import get_numcodec

    assert get_numcodec({"id": "gzip", "level": 2}) == GZip(level=2)  # type: ignore[typeddict-unknown-key]


def test_get_numcodec_without_an_id_keeps_the_numcodecs_error() -> None:
    """With no codec id there is nothing to look up, so numcodecs' own error stands."""
    from zarr.errors import UnknownCodecError
    from zarr.registry import get_numcodec

    # Not asserting on numcodecs' exception class: it only gained a dedicated
    # `numcodecs.errors.UnknownCodecError` in 0.15.1, and zarr supports numcodecs >= 0.14.
    with pytest.raises(ValueError) as excinfo:
        get_numcodec({"level": 2})  # type: ignore[typeddict-item,typeddict-unknown-key]
    assert not isinstance(excinfo.value, UnknownCodecError)


async def test_open_array_with_missing_v3_codec_reports_package() -> None:
    """Opening a Zarr format 3 array naming a codec we lack points at the package."""
    import json

    import zarr
    from zarr.core.buffer import default_buffer_prototype
    from zarr.errors import UnknownCodecError
    from zarr.storage import MemoryStore

    store = MemoryStore()
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "data_type": "float64",
        "fill_value": 0.0,
        "codecs": [{"name": "bytes"}, {"name": "n5_default"}],
        "attributes": {},
    }
    await store.set(
        "zarr.json",
        default_buffer_prototype().buffer.from_bytes(json.dumps(metadata).encode()),
    )
    with pytest.raises(UnknownCodecError, match="zarr-n5"):
        zarr.open_array(store=store, mode="r")


async def test_open_array_with_missing_v2_codec_reports_package() -> None:
    """Same, for a Zarr format 2 array whose compressor id we lack."""
    import json

    import zarr
    from zarr.core.buffer import default_buffer_prototype
    from zarr.errors import UnknownCodecError
    from zarr.storage import MemoryStore

    store = MemoryStore()
    metadata = {
        "zarr_format": 2,
        "shape": [4],
        "chunks": [4],
        "dtype": "<f8",
        "compressor": {"id": "wavpack"},
        "fill_value": 0.0,
        "order": "C",
        "filters": None,
    }
    await store.set(
        ".zarray",
        default_buffer_prototype().buffer.from_bytes(json.dumps(metadata).encode()),
    )
    with pytest.raises(UnknownCodecError, match="wavpack-numcodecs"):
        zarr.open_array(store=store, mode="r")


def test_get_codec_class_with_unregistered_config_pin_raises_unknown_codec_error() -> None:
    """A config pinning an implementation that isn't registered must not leak a KeyError."""
    from zarr.core.config import config
    from zarr.errors import UnknownCodecError
    from zarr.registry import get_codec_class

    with config.set({"codecs.bytes": "some.package.RemovedBytesCodec"}):
        with pytest.raises(UnknownCodecError, match="RemovedBytesCodec"):
            get_codec_class("bytes")


def test_open_array_with_unregistered_config_pin_raises_unknown_codec_error() -> None:
    """The same, through the array-open path that used to convert the KeyError for us."""
    from zarr.core.config import config
    from zarr.core.metadata.v3 import parse_codecs
    from zarr.errors import UnknownCodecError

    with config.set({"codecs.bytes": "some.package.RemovedBytesCodec"}):
        with pytest.raises(UnknownCodecError, match="RemovedBytesCodec"):
            parse_codecs([{"name": "bytes"}])


def test_is_missing_numcodec_error_ignores_unrelated_value_errors() -> None:
    """A ValueError about a bad configuration is not a missing codec, on any numcodecs."""
    from zarr.registry import _is_missing_numcodec_error

    assert not _is_missing_numcodec_error(ValueError("level must be between 0 and 9"))
