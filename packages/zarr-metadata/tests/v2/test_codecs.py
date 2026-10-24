"""Reading each v2 codec from its numcodecs configuration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    RefusedField,
    UnclaimedField,
    canonical_of,
    resolve,
)
from zarr_metadata.v2.codec import CODECS_V2, ZarrV2CodecMetadata
from zarr_metadata.v2.definition import ZarrV2CodecDefinition, ZarrV2Context

CONTEXT = ZarrV2Context(codecs=CODECS_V2)
Loc = tuple[str | int, ...]

# The output of numcodecs 0.16.5 `get_config()` for a default instance, where one exists, and
# other configurations that numcodecs 0.15 and 0.16 read and round-trip.
EXAMPLES: dict[str, tuple[dict[str, Any], ...]] = {
    # -1 is zlib's default-level constant, which numcodecs writes as given.
    "zlib": ({"id": "zlib", "level": 1}, {"id": "zlib"}, {"id": "zlib", "level": -1}),
    "gzip": ({"id": "gzip", "level": 9}, {"id": "gzip", "level": -1}),
    "bz2": ({"id": "bz2", "level": 1},),
    "lzma": ({"id": "lzma", "format": 1, "check": -1, "preset": None, "filters": None},),
    "blosc": (
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        {"id": "blosc", "clevel": 5},
        {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2, "blocksize": 0, "typesize": 4},
        # numcodecs writes a `shuffle` given as a bool as it was given.
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": True, "blocksize": 0},
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": False, "blocksize": 0},
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": -1},
    ),
    # zstd clamps a level outside its own range.
    "zstd": (
        {"id": "zstd", "level": 0, "checksum": False},
        {"id": "zstd", "level": 23, "checksum": False},
        {"id": "zstd", "level": -200000, "checksum": False},
        {"id": "zstd", "level": 2**31 - 1, "checksum": False},
    ),
    # numcodecs passes `acceleration` to C as an `int`, and lz4 reads one below 1 as 1.
    "lz4": (
        {"id": "lz4", "acceleration": 1},
        {"id": "lz4", "acceleration": 2**31 - 1},
        {"id": "lz4", "acceleration": -(2**31)},
    ),
    # An element size of 1 or less leaves the bytes as they are. numcodecs passes a larger
    # one to C as a `Py_ssize_t`.
    "shuffle": (
        {"id": "shuffle", "elementsize": 4},
        {"id": "shuffle", "elementsize": 0},
        {"id": "shuffle", "elementsize": -(2**63) - 1},
        {"id": "shuffle", "elementsize": 2**63 - 1},
    ),
    "delta": (
        {"id": "delta", "dtype": "<i4", "astype": "<i4"},
        {"id": "delta", "dtype": "<f8"},
        {"id": "delta", "dtype": "<f4", "astype": None},
        {"id": "delta", "dtype": "|b1", "astype": "|b1"},
        {"id": "delta", "dtype": "<M8[ns]", "astype": "<M8[ns]"},
        # numcodecs writes a time type without a unit as `np.dtype("m8").str` gives it.
        {"id": "delta", "dtype": "<m8", "astype": "<m8"},
    ),
    "fixedscaleoffset": (
        {"id": "fixedscaleoffset", "scale": 10, "offset": 0, "dtype": "<f8", "astype": "<f8"},
        {"id": "fixedscaleoffset", "scale": 10, "offset": 1000, "dtype": "<f8", "astype": "|u1"},
        {"id": "fixedscaleoffset", "scale": 1, "offset": 0, "dtype": "<f4", "astype": None},
    ),
    "quantize": (
        {"id": "quantize", "digits": 3, "dtype": "<f8", "astype": "<f8"},
        {"id": "quantize", "digits": -1, "dtype": "<f4", "astype": "<f4"},
        {"id": "quantize", "digits": 2, "dtype": "<f4", "astype": None},
    ),
    "bitround": ({"id": "bitround", "keepbits": 5},),
    "astype": (
        {"id": "astype", "encode_dtype": "<i2", "decode_dtype": "<f4"},
        {"id": "astype", "encode_dtype": "<M8", "decode_dtype": ">M8"},
    ),
    "packbits": ({"id": "packbits"},),
    "vlen-utf8": ({"id": "vlen-utf8"},),
    "vlen-bytes": ({"id": "vlen-bytes"},),
    "vlen-array": ({"id": "vlen-array", "dtype": "<i4"},),
    "crc32": (
        {"id": "crc32"},
        {"id": "crc32", "location": "end"},
        {"id": "crc32", "location": None},
    ),
    "crc32c": ({"id": "crc32c"},),
    "adler32": ({"id": "adler32"},),
    "fletcher32": ({"id": "fletcher32"},),
}
CASES = [(name, field) for name, fields in EXAMPLES.items() for field in fields]


def test_every_codec_has_an_example() -> None:
    """Every definition in `CODECS_V2` has an example below."""
    assert {definition.name for definition in CODECS_V2} == set(EXAMPLES)


@pytest.mark.parametrize(
    ("name", "field"), CASES, ids=[f"{n}:{i}" for i, (n, _) in enumerate(CASES)]
)
def test_every_example_reads_and_is_written_back_as_it_was(
    name: str, field: ZarrV2CodecMetadata
) -> None:
    """A numcodecs configuration is read by the definition its `id` names, with no problem, and is written back unchanged: the `id` next to the parameters. Its canonical form reads as the same codec."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, CONTEXT)
    assert isinstance(resolved, AcceptedField)
    assert resolved.definition.name == name
    assert problems == ()
    assert resolved.to_json() == field
    assert resolve(canonical_of(resolved, problems), ZarrV2CodecDefinition, CONTEXT)[0] == resolved


BLOSC_DEFAULTS: dict[str, Any] = {
    "id": "blosc",
    "cname": "lz4",
    "clevel": 5,
    "shuffle": 1,
    "blocksize": 0,
}
CHECKSUM_DEFAULTS: dict[str, Any] = {"location": "start"}


@pytest.mark.parametrize(
    ("field", "canonical"),
    [
        ({"id": "zlib"}, {"id": "zlib", "level": 1}),
        ({"id": "gzip"}, {"id": "gzip", "level": 1}),
        ({"id": "gzip", "level": 9}, {"id": "gzip", "level": 9}),
        ({"id": "bz2"}, {"id": "bz2", "level": 1}),
        (
            {"id": "lzma"},
            {"id": "lzma", "format": 1, "check": -1, "preset": None, "filters": None},
        ),
        ({"id": "blosc"}, BLOSC_DEFAULTS),
        ({"id": "blosc", "shuffle": True}, BLOSC_DEFAULTS),
        ({"id": "blosc", "shuffle": False}, {**BLOSC_DEFAULTS, "shuffle": 0}),
        ({"id": "blosc", "typesize": None}, BLOSC_DEFAULTS),
        ({"id": "blosc", "typesize": 4}, {**BLOSC_DEFAULTS, "typesize": 4}),
        ({"id": "blosc", "shuffle": 0, "typesize": 4}, {**BLOSC_DEFAULTS, "shuffle": 0}),
        ({"id": "blosc", "shuffle": False, "typesize": 4}, {**BLOSC_DEFAULTS, "shuffle": 0}),
        (
            {"id": "blosc", "shuffle": 2, "typesize": 4},
            {**BLOSC_DEFAULTS, "shuffle": 2, "typesize": 4},
        ),
        ({"id": "zstd"}, {"id": "zstd", "level": 0, "checksum": False}),
        ({"id": "zstd", "level": 0}, {"id": "zstd", "level": 0, "checksum": False}),
        ({"id": "lz4"}, {"id": "lz4", "acceleration": 1}),
        ({"id": "shuffle"}, {"id": "shuffle", "elementsize": 4}),
        ({"id": "delta", "dtype": "<i4"}, {"id": "delta", "dtype": "<i4", "astype": "<i4"}),
        (
            {"id": "delta", "dtype": "<i4", "astype": None},
            {"id": "delta", "dtype": "<i4", "astype": "<i4"},
        ),
        ({"id": "delta", "dtype": "<i1"}, {"id": "delta", "dtype": "|i1", "astype": "|i1"}),
        (
            {"id": "delta", "dtype": "|i1", "astype": "<u1"},
            {"id": "delta", "dtype": "|i1", "astype": "|u1"},
        ),
        (
            {"id": "delta", "dtype": "<M8[μs]"},
            {"id": "delta", "dtype": "<M8[us]", "astype": "<M8[us]"},
        ),
        (
            {"id": "astype", "encode_dtype": "<u1", "decode_dtype": "<b1"},
            {"id": "astype", "encode_dtype": "|u1", "decode_dtype": "|b1"},
        ),
        ({"id": "vlen-array", "dtype": "<i1"}, {"id": "vlen-array", "dtype": "|i1"}),
        (
            {"id": "fixedscaleoffset", "offset": 0, "scale": 1, "dtype": "<f8"},
            {
                "id": "fixedscaleoffset",
                "offset": 0.0,
                "scale": 1.0,
                "dtype": "<f8",
                "astype": "<f8",
            },
        ),
        (
            {"id": "fixedscaleoffset", "offset": 2, "scale": 10, "dtype": "<f8", "astype": "<u1"},
            {
                "id": "fixedscaleoffset",
                "offset": 2.0,
                "scale": 10.0,
                "dtype": "<f8",
                "astype": "|u1",
            },
        ),
        (
            {"id": "quantize", "digits": 2, "dtype": "<f4", "astype": None},
            {"id": "quantize", "digits": 2, "dtype": "<f4", "astype": "<f4"},
        ),
        ({"id": "crc32"}, {"id": "crc32", **CHECKSUM_DEFAULTS}),
        ({"id": "crc32", "location": None}, {"id": "crc32", **CHECKSUM_DEFAULTS}),
        ({"id": "crc32c"}, {"id": "crc32c", **CHECKSUM_DEFAULTS}),
        ({"id": "adler32"}, {"id": "adler32", **CHECKSUM_DEFAULTS}),
        ({"id": "bitround", "keepbits": 5}, {"id": "bitround", "keepbits": 5}),
        ({"id": "fletcher32"}, {"id": "fletcher32"}),
    ],
)
def test_the_canonical_form_fills_in_defaults_and_writes_each_value_one_way(
    field: dict[str, Any], canonical: dict[str, Any]
) -> None:
    """The canonical form of a codec fills in each parameter that numcodecs gives a default, where the configuration omits it or writes null, as `get_config()` writes it. A blosc `shuffle` of `true` or `false` is 1 or 0, which numcodecs reads it as, and an `astype` left out is the `dtype`. It writes each value that numcodecs reads alike in one way: a dtype parameter as NumPy spells it (`<i1` is `|i1`, `μs` is `us`), as an array's dtype compares; a fixedscaleoffset `offset` and `scale` as floats; and a blosc `typesize` is dropped under `shuffle` 0, where nothing is shuffled. So a configuration and its canonical form are one codec: they compare equal, with equal hashes. A codec with nothing to fill in or respell is unchanged."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, CONTEXT)
    filled, _ = resolve(canonical, ZarrV2CodecDefinition, CONTEXT)
    assert canonical_of(resolved, problems) == canonical
    assert resolved == filled
    assert hash(resolved) == hash(filled)


def test_a_parameter_set_away_from_its_default_is_another_codec() -> None:
    """A parameter written with another value than its numcodecs default gives another codec."""
    resolved, _ = resolve({"id": "zstd", "level": 3}, ZarrV2CodecDefinition, CONTEXT)
    default, _ = resolve({"id": "zstd"}, ZarrV2CodecDefinition, CONTEXT)
    assert resolved != default


@pytest.mark.parametrize(
    "field",
    [{"id": "categorize", "labels": ["a"]}, {"id": "pickle"}, {"id": "n5_wrapper", "inner": 1}],
)
def test_an_id_the_package_does_not_model_is_unclaimed(field: dict[str, Any]) -> None:
    """A codec id that no definition in scope claims is read as `UnclaimedField`. Its parameters are kept and not checked, as for an unclaimed v3 extension."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, CONTEXT)
    assert isinstance(resolved, UnclaimedField)
    assert problems == ()
    assert json.dumps(resolved.to_json()) == json.dumps(field)


@pytest.mark.parametrize(
    ("field", "at", "kind"),
    [
        ({"id": "zlib", "level": 10}, ("c", "level"), "invalid_value"),
        ({"id": "gzip", "level": -2}, ("c", "level"), "invalid_value"),
        ({"id": ""}, ("c", "id"), "invalid_value"),
        ({"id": "bz2", "level": 0}, ("c", "level"), "invalid_value"),
        ({"id": "blosc", "cname": "brotli"}, ("c", "cname"), "invalid_value"),
        ({"id": "blosc", "clevel": -1}, ("c", "clevel"), "invalid_value"),
        ({"id": "blosc", "shuffle": 3}, ("c", "shuffle"), "invalid_value"),
        ({"id": "blosc", "blocksize": 2**31}, ("c", "blocksize"), "invalid_value"),
        ({"id": "blosc", "shuffle": None}, ("c", "shuffle"), "invalid_type"),
        ({"id": "zstd", "level": 2**31}, ("c", "level"), "invalid_value"),
        ({"id": "zstd", "level": -(2**31) - 1}, ("c", "level"), "invalid_value"),
        ({"id": "lz4", "acceleration": 2**31}, ("c", "acceleration"), "invalid_value"),
        ({"id": "lz4", "acceleration": -(2**31) - 1}, ("c", "acceleration"), "invalid_value"),
        ({"id": "shuffle", "elementsize": 2**63}, ("c", "elementsize"), "invalid_value"),
        ({"id": "shuffle", "elementsize": 1.5}, ("c", "elementsize"), "invalid_type"),
        ({"id": "delta"}, ("c", "dtype"), "missing_key"),
        ({"id": "delta", "dtype": "float32"}, ("c", "dtype"), "invalid_value"),
        ({"id": "delta", "dtype": "<f3"}, ("c", "dtype"), "invalid_value"),
        ({"id": "delta", "dtype": "<f4", "astype": "|f4"}, ("c", "astype"), "invalid_value"),
        ({"id": "vlen-array", "dtype": "<i16"}, ("c", "dtype"), "invalid_value"),
        (
            {"id": "astype", "encode_dtype": "|M8", "decode_dtype": "<f8"},
            ("c", "encode_dtype"),
            "invalid_value",
        ),
        (
            {"id": "astype", "encode_dtype": "<i2", "decode_dtype": "f4"},
            ("c", "decode_dtype"),
            "invalid_value",
        ),
        ({"id": "quantize", "digits": -309, "dtype": "<f8"}, ("c", "digits"), "invalid_value"),
        ({"id": "quantize", "digits": 308, "dtype": "<f8"}, ("c", "digits"), "invalid_value"),
        ({"id": "quantize", "digits": 3, "dtype": "<i8"}, ("c", "dtype"), "invalid_value"),
        ({"id": "bitround", "keepbits": -1}, ("c", "keepbits"), "invalid_value"),
        ({"id": "crc32", "location": "middle"}, ("c", "location"), "invalid_value"),
        ({"id": "packbits", "extra": 1}, ("c", "extra"), "unknown_key"),
        ({"id": "vlen-array"}, ("c", "dtype"), "missing_key"),
        ({"id": "zlib", "level": "1"}, ("c", "level"), "invalid_type"),
    ],
)
def test_error_a_parameter_outside_what_numcodecs_takes_is_a_problem(
    field: dict[str, Any], at: Loc, kind: str
) -> None:
    """Each of these is reported at the parameter's location inside the field: a parameter out of the range numcodecs reads, a parameter of the wrong type, a parameter that is missing when numcodecs has no default for it, a dtype parameter that an array's dtype could not be, and a key that no codec declares."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, CONTEXT, ("c",))
    assert [(p.loc, p.kind) for p in problems] == [(at, kind)]
    assert isinstance(resolved, AcceptedField if kind == "unknown_key" else RefusedField)
