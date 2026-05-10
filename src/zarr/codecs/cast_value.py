"""Cast-value array-to-array codec.

Value-converts array elements to a new data type during encoding,
and back to the original data type during decoding, with configurable
rounding, out-of-range handling, and explicit scalar mappings.

Requires the optional ``cast-value-rs`` package for the actual casting
logic. Install it with: ``pip install cast-value-rs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype import get_data_type_from_json

if TYPE_CHECKING:
    from typing import NotRequired, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata

    class ScalarMapJSON(TypedDict):
        encode: NotRequired[list[tuple[object, object]]]
        decode: NotRequired[list[tuple[object, object]]]


RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]


class ScalarMap(TypedDict, total=False):
    """
    The normalized, in-memory form of a scalar map.
    """

    encode: Mapping[str | float | int, str | float | int]
    decode: Mapping[str | float | int, str | float | int]


# see https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value
CAST_VALUE_INT_DTYPES: Final[set[str]] = {
    # signed
    "int2",
    "int4",
    "int8",
    "int16",
    "int32",
    "int64",
    # unsigned
    "uint2",
    "uint4",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
"""Integer dtype identifiers permitted as the source or target of `cast_value`.

Membership in this set drives the `out_of_range="wrap"` rule, which the
spec restricts to integral targets that use two's-complement representation
for modular arithmetic.
"""

CAST_VALUE_FLOAT_DTYPES: Final[set[str]] = {
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "bfloat16",
    "float16",
    "float32",
    "float64",
}
"""Floating-point dtype identifiers permitted as the source or target of `cast_value`."""

PERMITTED_DATA_TYPE_NAMES: Final[set[str]] = CAST_VALUE_INT_DTYPES | CAST_VALUE_FLOAT_DTYPES
"""All dtype identifiers the `cast_value` codec is defined for."""


def parse_scalar_map(obj: ScalarMapJSON | ScalarMap) -> ScalarMap:
    """
    Parse a scalar map into its normalized dict-of-dicts form.

    Accepts either the JSON form (lists of tuples) or an already-normalized form
    (dicts). For example, ``{"encode": [("NaN", 0)]}`` becomes
    ``{"encode": {"NaN": 0}}``.
    """
    result: ScalarMap = {}
    for direction in ("encode", "decode"):
        if direction in obj:
            entries = obj[direction]
            if entries is not None:
                if isinstance(entries, Mapping):
                    result[direction] = entries
                else:
                    result[direction] = dict(entries)  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# Backend: cast-value-rs
# ---------------------------------------------------------------------------

try:
    from cast_value_rs import cast_array as cast_array_rs

    _HAS_RUST_BACKEND = True
except ModuleNotFoundError:
    _HAS_RUST_BACKEND = False


def _check_representable(
    value: JSON,
    zdtype: ZDType[TBaseDType, TBaseScalar],
    label: str,
) -> None:
    """Raise ``ValueError`` if *value* cannot be parsed by *zdtype*."""
    try:
        zdtype.from_json_scalar(value, zarr_format=3)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"{label} {value!r} is not representable in dtype {zdtype.to_native_dtype()}."
        ) from e


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CastValue(ArrayArrayCodec):
    """Cast-value array-to-array codec.

    Value-converts array elements to a new data type during encoding,
    and back to the original data type during decoding.

    Requires the `cast-value-rs` package for the actual casting logic.

    Parameters
    ----------
    data_type : str or ZDType
        Target zarr v3 data type. Strings are looked up by spec name
        (e.g. "uint8", "float32"); a `ZDType` instance is used as-is.
    rounding : RoundingMode
        How to round when exact representation is impossible. Default is
        "nearest-even".
    out_of_range : OutOfRangeMode or None
        What to do when a value is outside the target's range. `None` means
        error; "clamp" clips to range; "wrap" uses modular arithmetic
        (only valid for integer types). Default is `None`.
    scalar_map : ScalarMap, ScalarMapJSON, or None
        Explicit mapping from input scalars to output scalars. Default is
        `None`.

    Attributes
    ----------
    dtype : ZDType
        Resolved target data type (a `ZDType` instance, regardless of
        whether the constructor received a string or a `ZDType`).
    rounding : RoundingMode
        The rounding mode, as supplied to the constructor.
    out_of_range : OutOfRangeMode or None
        The out-of-range behaviour, as supplied to the constructor.
    scalar_map : ScalarMap or None
        Parsed scalar map (always normalized to `ScalarMap` form).

    References
    ----------

    - The `cast_value` codec spec: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value
    """

    is_fixed_size = True

    dtype: ZDType[TBaseDType, TBaseScalar]
    rounding: RoundingMode
    out_of_range: OutOfRangeMode | None
    scalar_map: ScalarMap | None

    def __init__(
        self,
        *,
        data_type: str | ZDType[TBaseDType, TBaseScalar],
        rounding: RoundingMode = "nearest-even",
        out_of_range: OutOfRangeMode | None = None,
        scalar_map: ScalarMapJSON | ScalarMap | None = None,
    ) -> None:
        if isinstance(data_type, str):
            zdtype = get_data_type_from_json(data_type, zarr_format=3)
        else:
            zdtype = data_type
        if zdtype.to_json(zarr_format=3) not in PERMITTED_DATA_TYPE_NAMES:
            raise ValueError(
                f"Invalid target data type {data_type!r}. "
                f"cast_value codec only supports integer and floating-point data types. "
                f"Got {zdtype}."
            )
        object.__setattr__(self, "dtype", zdtype)
        object.__setattr__(self, "rounding", rounding)
        object.__setattr__(self, "out_of_range", out_of_range)
        if scalar_map is not None:
            parsed = parse_scalar_map(scalar_map)
        else:
            parsed = None
        object.__setattr__(self, "scalar_map", parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "cast_value", require_configuration=True
        )
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        config: dict[str, JSON] = {"data_type": cast("JSON", self.dtype.to_json(zarr_format=3))}
        if self.rounding != "nearest-even":
            config["rounding"] = self.rounding
        if self.out_of_range is not None:
            config["out_of_range"] = self.out_of_range
        if self.scalar_map is not None:
            json_map: dict[str, list[tuple[object, object]]] = {}
            for direction in ("encode", "decode"):
                if direction in self.scalar_map:
                    json_map[direction] = [(k, v) for k, v in self.scalar_map[direction].items()]
            config["scalar_map"] = cast("JSON", json_map)
        return {"name": "cast_value", "configuration": config}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        # `dtype` is the source (the array's dtype); `self.dtype` is the
        # cast target. The spec requires both to be permitted, and rules
        # like `out_of_range="wrap"` apply to the target.
        source_name = dtype.to_json(zarr_format=3)
        target_name = self.dtype.to_json(zarr_format=3)
        for role, name in (("source", source_name), ("target", target_name)):
            if name not in PERMITTED_DATA_TYPE_NAMES:
                raise ValueError(
                    f"The cast_value codec only supports integer and floating-point data types. "
                    f"Got {role} dtype {name}."
                )
        if self.out_of_range == "wrap" and target_name not in CAST_VALUE_INT_DTYPES:
            raise ValueError(
                f"out_of_range='wrap' is only valid for integer target types. "
                f"Got target dtype {target_name}."
            )

        if self.scalar_map is not None:
            self._validate_scalar_map(dtype, self.dtype)

    def _validate_scalar_map(
        self,
        source_zdtype: ZDType[TBaseDType, TBaseScalar],
        target_zdtype: ZDType[TBaseDType, TBaseScalar],
    ) -> None:
        """Validate that scalar map entries are compatible with source/target dtypes."""
        assert self.scalar_map is not None
        # For encode: keys are source values, values are target values.
        # For decode: keys are target values, values are source values.
        direction_dtypes: dict[
            str, tuple[ZDType[TBaseDType, TBaseScalar], ZDType[TBaseDType, TBaseScalar]]
        ] = {
            "encode": (source_zdtype, target_zdtype),
            "decode": (target_zdtype, source_zdtype),
        }
        for direction, (key_zdtype, val_zdtype) in direction_dtypes.items():
            if direction not in self.scalar_map:
                continue
            sub_map = self.scalar_map[direction]  # type: ignore[literal-required]
            for k, v in sub_map.items():
                _check_representable(k, key_zdtype, f"scalar_map {direction} key")
                _check_representable(v, val_zdtype, f"scalar_map {direction} value")

    def _do_cast(
        self,
        arr: np.ndarray,  # type: ignore[type-arg]
        *,
        target_dtype: np.dtype,  # type: ignore[type-arg]
        scalar_map: Mapping[str | float | int, str | float | int] | None,
    ) -> np.ndarray:  # type: ignore[type-arg]
        if not _HAS_RUST_BACKEND:
            raise ImportError(
                "The cast_value codec requires the 'cast-value-rs' package. "
                "Install it with: pip install cast-value-rs"
            )
        scalar_map_entries: dict[float | int, float | int] | None = None
        if scalar_map is not None:
            src_dtype = arr.dtype
            to_src = int if np.issubdtype(src_dtype, np.integer) else float
            to_tgt = int if np.issubdtype(target_dtype, np.integer) else float
            scalar_map_entries = {to_src(k): to_tgt(v) for k, v in scalar_map.items()}
        return cast_array_rs(  # type: ignore[no-any-return]
            arr,
            target_dtype=target_dtype,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=scalar_map_entries,
        )

    def _get_scalar_map(
        self, direction: str
    ) -> Mapping[str | float | int, str | float | int] | None:
        """Extract the encode or decode mapping from scalar_map, or None."""
        if self.scalar_map is None:
            return None
        return self.scalar_map.get(direction)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        Update the fill value of the output spec by applying casting procedure.
        """
        target_zdtype = self.dtype
        target_native = target_zdtype.to_native_dtype()
        source_native = chunk_spec.dtype.to_native_dtype()

        fill = chunk_spec.fill_value
        fill_arr = np.array([fill], dtype=source_native)

        new_fill_arr = self._do_cast(
            fill_arr, target_dtype=target_native, scalar_map=self._get_scalar_map("encode")
        )
        new_fill = target_native.type(new_fill_arr[0])

        return replace(chunk_spec, dtype=target_zdtype, fill_value=new_fill)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = chunk_array.as_ndarray_like()
        target_native = self.dtype.to_native_dtype()

        result = self._do_cast(
            np.asarray(arr), target_dtype=target_native, scalar_map=self._get_scalar_map("encode")
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_data, chunk_spec)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = chunk_array.as_ndarray_like()
        target_native = chunk_spec.dtype.to_native_dtype()

        result = self._do_cast(
            np.asarray(arr), target_dtype=target_native, scalar_map=self._get_scalar_map("decode")
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_data, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        dtype_name = chunk_spec.dtype.to_json(zarr_format=3)
        if dtype_name not in PERMITTED_DATA_TYPE_NAMES:
            raise ValueError(
                "cast_value codec only supports fixed-size integer and floating-point data types. "
                f"Got source dtype: {chunk_spec.dtype}."
            )
        source_itemsize = chunk_spec.dtype.to_native_dtype().itemsize
        target_itemsize = self.dtype.to_native_dtype().itemsize
        if source_itemsize == 0 or target_itemsize == 0:
            raise ValueError(
                "cast_value codec requires fixed-size data types. "
                f"Got source itemsize={source_itemsize}, target itemsize={target_itemsize}."
            )
        num_elements = input_byte_length // source_itemsize
        return num_elements * target_itemsize
