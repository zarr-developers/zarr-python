"""Every v2 codec, read as its numcodecs configuration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2.codec import V2_CODECS, ZarrV2CodecMetadata
from zarr_metadata.v3.definition import Context, Read, Refused, Unclaimed, canonical_of, resolve

SCOPE = Context.of(*V2_CODECS)
Loc = tuple[str | int, ...]

# numcodecs 0.16.5 `get_config()` of a default instance, where one exists.
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
    ),
    "zstd": ({"id": "zstd", "level": 0, "checksum": False},),
    "lz4": ({"id": "lz4", "acceleration": 1},),
    "shuffle": ({"id": "shuffle", "elementsize": 4},),
    "delta": ({"id": "delta", "dtype": "<i4", "astype": "<i4"}, {"id": "delta", "dtype": "<f8"}),
    "fixedscaleoffset": (
        {"id": "fixedscaleoffset", "scale": 10, "offset": 0, "dtype": "<f8", "astype": "<f8"},
    ),
    "quantize": ({"id": "quantize", "digits": 3, "dtype": "<f8", "astype": "<f8"},),
    "bitround": ({"id": "bitround", "keepbits": 5},),
    "astype": ({"id": "astype", "encode_dtype": "<i2", "decode_dtype": "<f4"},),
    "packbits": ({"id": "packbits"},),
    "vlen-utf8": ({"id": "vlen-utf8"},),
    "vlen-bytes": ({"id": "vlen-bytes"},),
    "vlen-array": ({"id": "vlen-array", "dtype": "<i4"},),
    "crc32": ({"id": "crc32"}, {"id": "crc32", "location": "end"}),
    "crc32c": ({"id": "crc32c"},),
    "adler32": ({"id": "adler32"},),
    "fletcher32": ({"id": "fletcher32"},),
}
CASES = [(name, field) for name, fields in EXAMPLES.items() for field in fields]


def test_every_codec_has_an_example() -> None:
    """Every definition in `V2_CODECS` is exercised below."""
    assert {definition.name for definition in V2_CODECS} == set(EXAMPLES)


@pytest.mark.parametrize(
    ("name", "field"), CASES, ids=[f"{n}:{i}" for i, (n, _) in enumerate(CASES)]
)
def test_every_example_reads_and_is_written_back_as_it_was(
    name: str, field: ZarrV2CodecMetadata
) -> None:
    """A numcodecs configuration reads by the definition its id names, with no problem, and is written back as it was: the id beside the parameters, which have one spelling."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, SCOPE)
    assert isinstance(resolved, Read)
    assert resolved.definition.name == name
    assert problems == ()
    assert resolved.to_json() == field
    assert canonical_of(resolved, problems) == field


@pytest.mark.parametrize(
    "field",
    [{"id": "categorize", "labels": ["a"]}, {"id": "pickle"}, {"id": "n5_wrapper", "inner": 1}],
)
def test_an_id_the_package_does_not_model_is_unclaimed(field: dict[str, Any]) -> None:
    """A codec id nothing in scope claims reads as `Unclaimed`, its parameters kept and unjudged, as a v3 extension nothing claims is."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, SCOPE)
    assert isinstance(resolved, Unclaimed)
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
        ({"id": "blosc", "shuffle": 3}, ("c", "shuffle"), "invalid_value"),
        ({"id": "blosc", "blocksize": -1}, ("c", "blocksize"), "invalid_value"),
        ({"id": "zstd", "level": 23}, ("c", "level"), "invalid_value"),
        ({"id": "shuffle", "elementsize": 0}, ("c", "elementsize"), "invalid_value"),
        ({"id": "delta"}, ("c", "dtype"), "missing_key"),
        ({"id": "delta", "dtype": "float32"}, ("c", "dtype"), "invalid_value"),
        (
            {"id": "astype", "encode_dtype": "<i2", "decode_dtype": "f4"},
            ("c", "decode_dtype"),
            "invalid_value",
        ),
        ({"id": "quantize", "digits": -1, "dtype": "<f8"}, ("c", "digits"), "invalid_value"),
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
    """A parameter out of its range, of the wrong type, missing when numcodecs has no default, a dtype parameter that is no typestr, or a key no codec declares is reported at the parameter, beside the field."""
    resolved, problems = resolve(field, ZarrV2CodecDefinition, SCOPE, ("c",))
    assert [(p.loc, p.kind) for p in problems] == [(at, kind)]
    assert isinstance(resolved, Read if kind == "unknown_key" else Refused)
