from __future__ import annotations

import json

import numcodecs.registry
import pytest

import zarr
import zarr.registry
from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import BadConfigError, config
from zarr.core.metadata.v3 import parse_codecs
from zarr.errors import UnknownCodecError
from zarr.registry import (
    _CODEC_PACKAGE_PREFIXES,
    _CODEC_PACKAGES,
    _NUMCODEC_PACKAGE_PREFIXES,
    _NUMCODECS_CODEC_DOCS_URL,
    _ZARR_CODEC_DOCS_URL,
    _missing_codec_message,
    _packages_for_codec,
    get_codec_class,
    get_numcodec,
)
from zarr.storage import MemoryStore


@pytest.fixture
def unregistered_v3_codec(monkeypatch: pytest.MonkeyPatch) -> str:
    """Guarantee a Zarr format 3 codec name resolves to nothing.

    The registry is entry-point driven, so a name the hint table advertises resolves for real
    in any environment where the advertised package happens to be installed.
    """
    name = "n5_default"
    monkeypatch.setitem(zarr.registry._codec_registries, name, zarr.registry.Registry())
    return name


@pytest.fixture
def unregistered_v2_codec(monkeypatch: pytest.MonkeyPatch) -> str:
    """As above, for a numcodecs codec id."""
    codec_id = "wavpack"
    monkeypatch.delitem(numcodecs.registry.codec_registry, codec_id, raising=False)
    monkeypatch.delitem(numcodecs.registry.entries, codec_id, raising=False)
    return codec_id


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
    assert msg == (
        "An implementation for codec 'n5_default' is not available. Register one explicitly "
        f"using the codec registry (see {_ZARR_CODEC_DOCS_URL}), or install a Python package "
        "that registers a codec implementation with zarr. Known packages supporting this "
        "codec: zarr-n5."
    )


def test_missing_codec_message_without_known_packages() -> None:
    """With no known package we still explain how to register one by hand."""
    msg = _missing_codec_message("totally-made-up", zarr_format=3)
    assert msg == (
        "An implementation for codec 'totally-made-up' is not available. Register one "
        f"explicitly using the codec registry (see {_ZARR_CODEC_DOCS_URL}), or install a "
        "Python package that registers a codec implementation with zarr."
    )
    assert "Known packages" not in msg


def test_missing_codec_message_for_zarr_format_2() -> None:
    """Format 2 codecs live in the numcodecs registry, so the message must say so."""
    msg = _missing_codec_message("wavpack", zarr_format=2)
    assert msg == (
        "An implementation for codec 'wavpack' is not available. Register one explicitly "
        f"using the codec registry (see {_NUMCODECS_CODEC_DOCS_URL}), or install a Python "
        "package that registers a codec implementation with numcodecs. Known packages "
        "supporting this codec: wavpack-numcodecs."
    )


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

    # Select on the implementing class's module rather than on "is this registry non-empty":
    # a third-party entry-point codec lazy-loaded by an earlier test would otherwise show up
    # here and fail the assertion even though it shadows nothing.
    implemented = {
        name
        for name, reg in _codec_registries.items()
        if any(cls.__module__.startswith("zarr.") for cls in reg.values())
    }
    assert implemented, "expected importing zarr.codecs to populate the registry"
    assert not (implemented & set(_CODEC_PACKAGES))
    for prefix in _CODEC_PACKAGE_PREFIXES:
        assert not any(name.startswith(prefix) for name in implemented)


def test_get_codec_class_unknown_raises_with_package_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered codec with a known package names that package in the error."""
    from collections import defaultdict

    monkeypatch.setattr(zarr.registry, "_codec_registries", defaultdict(zarr.registry.Registry))
    with pytest.raises(UnknownCodecError, match="Known packages supporting this codec: zarr-n5"):
        get_codec_class("n5_default")


def test_get_codec_class_unknown_raises_without_package_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered codec we know nothing about still explains manual registration."""
    from collections import defaultdict

    monkeypatch.setattr(zarr.registry, "_codec_registries", defaultdict(zarr.registry.Registry))
    with pytest.raises(UnknownCodecError, match="An implementation for codec 'nope' is not"):
        get_codec_class("nope")


def test_unknown_codec_error_is_exported() -> None:
    """UnknownCodecError is public API now that we tell users to catch it."""
    import zarr.errors

    assert "UnknownCodecError" in zarr.errors.__all__


def test_get_numcodec_unknown_raises_with_package_hint(unregistered_v2_codec: str) -> None:
    """A Zarr format 2 codec id we don't have names the package that provides it."""
    with pytest.raises(
        UnknownCodecError, match="Known packages supporting this codec: wavpack-numcodecs"
    ):
        get_numcodec({"id": unregistered_v2_codec})


def test_get_numcodec_unknown_points_at_numcodecs_registry() -> None:
    """The Zarr format 2 message links the numcodecs registry, not zarr's extending guide."""
    with pytest.raises(UnknownCodecError) as excinfo:
        get_numcodec({"id": "definitely-not-a-real-codec"})
    assert _NUMCODECS_CODEC_DOCS_URL in str(excinfo.value)
    assert _ZARR_CODEC_DOCS_URL not in str(excinfo.value)


def test_get_numcodec_known_codec_still_works() -> None:
    """The happy path is untouched."""
    from numcodecs import GZip

    assert get_numcodec({"id": "gzip", "level": 2}) == GZip(level=2)  # type: ignore[typeddict-unknown-key]


def test_get_numcodec_without_an_id_keeps_the_numcodecs_error() -> None:
    """With no codec id there is nothing to look up, so numcodecs' own error stands."""
    # Not asserting on numcodecs' exception class: it only gained a dedicated
    # `numcodecs.errors.UnknownCodecError` in 0.15.1, and zarr supports numcodecs >= 0.14.
    with pytest.raises(ValueError) as excinfo:
        get_numcodec({"level": 2})  # type: ignore[typeddict-item,typeddict-unknown-key]
    assert not isinstance(excinfo.value, UnknownCodecError)


def test_get_numcodec_does_not_relabel_a_bad_configuration() -> None:
    """A registered codec rejecting its config is not a missing codec.

    Telling the user to install a package they already have would be the same misleading
    error this module exists to remove, pointing the other way.
    """
    with pytest.raises(ValueError) as excinfo:
        get_numcodec({"id": "bitround", "keepbits": -1})  # type: ignore[typeddict-unknown-key]
    assert not isinstance(excinfo.value, UnknownCodecError)
    assert str(excinfo.value) == "keepbits must be zero or positive"


@pytest.mark.parametrize("via", ["zarr", "numcodecs"])
def test_get_numcodec_does_not_relabel_a_missing_inner_codec(via: str) -> None:
    """A wrapper codec failing on an inner codec must not be blamed on the outer id.

    The outer id is registered, so relabelling it would advertise a package the user has
    already installed. Covers both routes a real wrapper takes to resolve its inner codec:
    zarr's ``get_numcodec`` and numcodecs' own ``get_codec``.
    """
    import numcodecs.registry
    from numcodecs.abc import Codec

    resolve = get_numcodec if via == "zarr" else numcodecs.registry.get_codec

    class WrapperCodec(Codec):  # type: ignore[misc]
        codec_id = "test_wrapper"

        @classmethod
        def from_config(cls, config: dict[str, object]) -> WrapperCodec:
            resolve({"id": "some-missing-inner-codec"})
            return cls()

        def encode(self, buf: object) -> object:
            return buf

        def decode(self, buf: object, out: object = None) -> object:
            return buf

    numcodecs.registry.register_codec(WrapperCodec, codec_id="test_wrapper")
    try:
        with pytest.raises(ValueError) as excinfo:
            get_numcodec({"id": "test_wrapper"})
        assert "some-missing-inner-codec" in str(excinfo.value)
        assert "test_wrapper" not in str(excinfo.value)
    finally:
        numcodecs.registry.codec_registry.pop("test_wrapper", None)


@pytest.mark.parametrize("data", ["abc", ["ab"]])
def test_get_numcodec_non_mapping_input_still_raises_value_error(data: object) -> None:
    """Input that is not a mapping must reach numcodecs rather than raising AttributeError.

    Reading ``id`` off the input unconditionally turned numcodecs' ValueError into an
    AttributeError, which is not a ValueError and so escaped handlers that caught it. Both
    params end in a ValueError from numcodecs, by different routes: ``"abc"`` fails
    ``dict(config)`` coercion, while ``["ab"]`` coerces to ``{"a": "b"}`` and then has no id.
    """
    with pytest.raises(ValueError):
        get_numcodec(data)  # type: ignore[arg-type]


def test_imagecodecs_prefix_does_not_over_match_in_zarr_format_3() -> None:
    """virtual-tiff provides 15 of the 81 `imagecodecs_*` names; the rest must get no hint.

    Recommending virtual-tiff for a name it does not provide is worse than saying nothing.
    """
    assert _packages_for_codec("imagecodecs_jpeg2k", zarr_format=3) == ("virtual-tiff",)
    for name in ("imagecodecs_jpegls", "imagecodecs_avif", "imagecodecs_blosc"):
        assert _packages_for_codec(name, zarr_format=3) == ()
        assert _packages_for_codec(name, zarr_format=2) == ("imagecodecs-numcodecs",)


def test_resolve_codec_reports_missing_codec() -> None:
    """The other public entry point into `get_codec_class` gets the message too."""
    import zarr

    with pytest.raises(UnknownCodecError, match="Known packages supporting this codec: zarr-n5"):
        zarr.create_array({}, shape=(4,), dtype="uint8", compressors=[{"name": "n5_default"}])


async def test_open_array_with_missing_v3_codec_reports_package(
    unregistered_v3_codec: str,
) -> None:
    """Opening a Zarr format 3 array naming a codec we lack points at the package."""
    store = MemoryStore()
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "data_type": "float64",
        "fill_value": 0.0,
        "codecs": [{"name": "bytes"}, {"name": unregistered_v3_codec}],
        "attributes": {},
    }
    await store.set(
        "zarr.json",
        default_buffer_prototype().buffer.from_bytes(json.dumps(metadata).encode()),
    )
    with pytest.raises(UnknownCodecError, match="Known packages supporting this codec: zarr-n5"):
        zarr.open_array(store=store, mode="r")


async def test_open_array_with_missing_v2_codec_reports_package(
    unregistered_v2_codec: str,
) -> None:
    """Same, for a Zarr format 2 array whose compressor id we lack."""
    store = MemoryStore()
    metadata = {
        "zarr_format": 2,
        "shape": [4],
        "chunks": [4],
        "dtype": "<f8",
        "compressor": {"id": unregistered_v2_codec},
        "fill_value": 0.0,
        "order": "C",
        "filters": None,
    }
    await store.set(
        ".zarray",
        default_buffer_prototype().buffer.from_bytes(json.dumps(metadata).encode()),
    )
    with pytest.raises(
        UnknownCodecError, match="Known packages supporting this codec: wavpack-numcodecs"
    ):
        zarr.open_array(store=store, mode="r")


def test_get_codec_class_with_unregistered_config_pin_raises_bad_config_error() -> None:
    """A config pinning an implementation that isn't registered is a config error."""
    with config.set({"codecs.bytes": "some.package.RemovedBytesCodec"}):
        with pytest.raises(BadConfigError, match="RemovedBytesCodec"):
            get_codec_class("bytes")


def test_parse_codecs_with_unregistered_config_pin_raises_bad_config_error() -> None:
    """The same, through the array-open path, and never as a bare KeyError."""
    with config.set({"codecs.bytes": "some.package.RemovedBytesCodec"}):
        with pytest.raises(BadConfigError, match="RemovedBytesCodec"):
            parse_codecs([{"name": "bytes"}])


def test_parse_codecs_converts_keyerror_from_from_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A codec whose from_dict indexes a malformed config must not leak a KeyError.

    A bare KeyError out of metadata parsing is caught by the array-then-group fallback in
    `zarr.api.asynchronous.open`, which then reports an unrelated group error.
    """
    from zarr.codecs import BytesCodec
    from zarr.errors import MetadataValidationError
    from zarr.registry import register_codec

    class PickyCodec(BytesCodec):
        @classmethod
        def from_dict(cls, data: object) -> PickyCodec:
            data["configuration"]["required_option"]  # type: ignore[index]
            return cls()

    monkeypatch.setitem(zarr.registry._codec_registries, "test_picky", zarr.registry.Registry())
    register_codec("test_picky", PickyCodec)
    with pytest.raises(MetadataValidationError, match="test_picky.*required_option"):
        parse_codecs([{"name": "test_picky", "configuration": {}}])
