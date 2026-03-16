from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict, cast

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype import get_data_type_from_json

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

NumericScalar: TypeAlias = np.integer[Any] | np.floating[Any]

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]


class ScalarMapJSON(TypedDict):
    encode: NotRequired[tuple[tuple[object, object]]]
    decode: NotRequired[tuple[tuple[object, object]]]


# Pre-parsed scalar map entry: (source_scalar, target_scalar)
_MapEntry = tuple[NumericScalar, NumericScalar]


def _parse_map_entries(
    mapping: dict[str, str],
    src_dtype: ZDType[TBaseDType, TBaseScalar],
    tgt_dtype: ZDType[TBaseDType, TBaseScalar],
) -> list[_MapEntry]:
    """Pre-parse a scalar map dict into a list of (src, tgt) tuples.

    Each entry's source value is deserialized using ``src_dtype`` and its target
    value using ``tgt_dtype``, preserving full precision for both data types.
    """
    entries: list[_MapEntry] = []
    for src_str, tgt_str in mapping.items():
        src = src_dtype.from_json_scalar(src_str, zarr_format=3)
        tgt = tgt_dtype.from_json_scalar(tgt_str, zarr_format=3)
        entries.append((src, tgt))  # type: ignore[arg-type]
    return entries


def _apply_scalar_map(work: np.ndarray[Any, np.dtype[Any]], entries: list[_MapEntry]) -> None:
    """Apply scalar map entries in-place. Single pass per entry."""
    for src, tgt in entries:
        if isinstance(src, (float, np.floating)) and np.isnan(src):
            mask = np.isnan(work)
        else:
            mask = work == src
        work[mask] = tgt


def _round_inplace(
    arr: np.ndarray[Any, np.dtype[Any]], mode: RoundingMode
) -> np.ndarray[Any, np.dtype[Any]]:
    """Round array, returning result (may or may not be a new array).

    For nearest-away, requires 3 numpy ops. All others are a single op.
    """
    match mode:
        case "nearest-even":
            return np.rint(arr)  # type: ignore [no-any-return]
        case "towards-zero":
            return np.trunc(arr)  # type: ignore [no-any-return]
        case "towards-positive":
            return np.ceil(arr)  # type: ignore [no-any-return]
        case "towards-negative":
            return np.floor(arr)  # type: ignore [no-any-return]
        case "nearest-away":
            return np.sign(arr) * np.floor(np.abs(arr) + 0.5)  # type: ignore [no-any-return]
    raise ValueError(f"Unknown rounding mode: {mode}")


def _cast_array(
    arr: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None,
    scalar_map_entries: list[_MapEntry] | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Cast an array to target_dtype with rounding, out-of-range, and scalar_map handling.

    Optimized to minimize allocations and passes over the data.
    For the simple case (no scalar_map, no rounding needed, no out-of-range),
    this is essentially just ``arr.astype(target_dtype)``.

    All casts are performed under ``np.errstate(over='raise', invalid='raise')``
    so that numpy overflow or invalid-value warnings become hard errors instead
    of being silently swallowed.
    """
    with np.errstate(over="raise", invalid="raise"):
        return _cast_array_impl(
            arr,
            target_dtype=target_dtype,
            rounding=rounding_mode,
            out_of_range=out_of_range_mode,
            scalar_map_entries=scalar_map_entries,
        )


def _check_int_range(
    work: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    out_of_range: OutOfRangeMode | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Check integer range and apply out-of-range handling, then cast."""
    info = np.iinfo(target_dtype)
    lo, hi = int(info.min), int(info.max)
    w_min, w_max = int(work.min()), int(work.max())
    if w_min >= lo and w_max <= hi:
        return work.astype(target_dtype)
    match out_of_range:
        case "clamp":
            return np.clip(work, lo, hi).astype(target_dtype)
        case "wrap":
            range_size = hi - lo + 1
            return ((work.astype(np.int64) - lo) % range_size + lo).astype(target_dtype)
        case None:
            oor_vals = work[(work < lo) | (work > hi)]
            raise ValueError(
                f"Values out of range for {target_dtype} (valid range: [{lo}, {hi}]), "
                f"got values in [{w_min}, {w_max}]. "
                f"Out-of-range values: {oor_vals.ravel()!r}. "
                f"Set out_of_range='clamp' or out_of_range='wrap' to handle this."
            )


def _cast_array_impl(
    arr: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    rounding: RoundingMode,
    out_of_range: OutOfRangeMode | None,
    scalar_map_entries: list[_MapEntry] | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    src_type: Literal["int", "float"] = "int" if np.issubdtype(arr.dtype, np.integer) else "float"
    tgt_type: Literal["int", "float"] = (
        "int" if np.issubdtype(target_dtype, np.integer) else "float"
    )
    has_map = bool(scalar_map_entries)

    match (src_type, tgt_type, has_map):
        # float→float or int→float without scalar_map — single astype
        case (_, "float", False):
            return arr.astype(target_dtype)

        # int→float with scalar_map — widen to float64, apply map, cast
        case ("int", "float", True):
            work = arr.astype(np.float64)
            _apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return work.astype(target_dtype)

        # float→float with scalar_map — copy, apply map, cast
        case ("float", "float", True):
            work = arr.copy()
            _apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return work.astype(target_dtype)

        # int→int without scalar_map — range check then astype
        case ("int", "int", False):
            if arr.dtype.itemsize > target_dtype.itemsize or arr.dtype != target_dtype:
                return _check_int_range(arr, target_dtype=target_dtype, out_of_range=out_of_range)
            return arr.astype(target_dtype)

        # int→int with scalar_map — widen to int64, apply map, range check
        case ("int", "int", True):
            work = arr.astype(np.int64)
            _apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return _check_int_range(work, target_dtype=target_dtype, out_of_range=out_of_range)

        # float→int (with or without scalar_map) — rounding + range check
        case ("float", "int", _):
            if arr.dtype != np.float64:
                work = arr.astype(np.float64)
            else:
                work = arr.copy()

            if scalar_map_entries:
                _apply_scalar_map(work, scalar_map_entries)

            bad = np.isnan(work) | np.isinf(work)
            if bad.any():
                raise ValueError("Cannot cast NaN or Infinity to integer type without scalar_map")

            work = _round_inplace(work, rounding)
            return _check_int_range(work, target_dtype=target_dtype, out_of_range=out_of_range)

    raise AssertionError(
        f"Unhandled type combination: src={src_type}, tgt={tgt_type}"
    )  # pragma: no cover


def _extract_raw_map(data: ScalarMapJSON | None, direction: str) -> dict[str, str] | None:
    """Extract raw string mapping from scalar_map JSON for 'encode' or 'decode'."""
    if data is None:
        return None
    raw: dict[str, str] = {}
    pairs = data.get(direction, [])
    for src, tgt in pairs:  # type: ignore[attr-defined]
        raw[str(src)] = str(tgt)
    return raw or None


@dataclass(frozen=True)
class CastValue(ArrayArrayCodec):
    """Cast-value array-to-array codec.

    Value-converts array elements to a new data type during encoding,
    and back to the original data type during decoding.

    Parameters
    ----------
    data_type : str
        Target zarr v3 data type name (e.g. "uint8", "float32").
    rounding : RoundingMode
        How to round when exact representation is impossible. Default is "nearest-even".
    out_of_range : OutOfRangeMode or None
        What to do when a value is outside the target's range.
        None means error. "clamp" clips to range. "wrap" uses modular arithmetic
        (only valid for integer types).
    scalar_map : dict or None
        Explicit value overrides as JSON: {"encode": [[src, tgt], ...], "decode": [[src, tgt], ...]}.
    """

    is_fixed_size = True

    dtype: ZDType[TBaseDType, TBaseScalar]
    rounding: RoundingMode
    out_of_range: OutOfRangeMode | None
    scalar_map: ScalarMapJSON | None

    def __init__(
        self,
        *,
        data_type: str | ZDType[TBaseDType, TBaseScalar],
        rounding: RoundingMode = "nearest-even",
        out_of_range: OutOfRangeMode | None = None,
        scalar_map: ScalarMapJSON | None = None,
    ) -> None:
        if isinstance(data_type, str):
            dtype = get_data_type_from_json(data_type, zarr_format=3)
        else:
            dtype = data_type
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "rounding", rounding)
        object.__setattr__(self, "out_of_range", out_of_range)
        object.__setattr__(self, "scalar_map", scalar_map)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "cast_value", require_configuration=True
        )
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        config: dict[str, JSON] = {"data_type": cast(JSON, self.dtype.to_json(zarr_format=3))}
        if self.rounding != "nearest-even":
            config["rounding"] = self.rounding
        if self.out_of_range is not None:
            config["out_of_range"] = self.out_of_range
        if self.scalar_map is not None:
            config["scalar_map"] = cast(JSON, self.scalar_map)
        return {"name": "cast_value", "configuration": config}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        source_native = dtype.to_native_dtype()
        target_native = self.dtype.to_native_dtype()
        for label, dt in [("source", source_native), ("target", target_native)]:
            if not np.issubdtype(dt, np.integer) and not np.issubdtype(dt, np.floating):
                raise ValueError(
                    f"cast_value codec only supports integer and floating-point data types. "
                    f"Got {label} dtype {dt}."
                )
        if self.out_of_range == "wrap" and not np.issubdtype(target_native, np.integer):
            raise ValueError("out_of_range='wrap' is only valid for integer target types.")
        # Check that int→float casts won't silently lose precision.
        # A float type with `m` mantissa bits can exactly represent all integers
        # in [-2**m, 2**m]. If the integer type's range exceeds that, the cast is lossy.
        if np.issubdtype(source_native, np.integer) and np.issubdtype(target_native, np.floating):
            int_info = np.iinfo(source_native)  # type: ignore[type-var]
            mantissa_bits = np.finfo(target_native).nmant  # type: ignore[arg-type]
            max_exact_int = 2**mantissa_bits
            if int_info.max > max_exact_int or int_info.min < -max_exact_int:
                raise ValueError(
                    f"Casting {source_native} to {target_native} may silently lose precision. "
                    f"{target_native} can only exactly represent integers up to 2**{mantissa_bits} "
                    f"({max_exact_int}), but {source_native} has range "
                    f"[{int_info.min}, {int_info.max}]."
                )
        # Same check for float→int decode direction
        if np.issubdtype(target_native, np.integer) and np.issubdtype(source_native, np.floating):
            int_info = np.iinfo(target_native)  # type: ignore[type-var]
            mantissa_bits = np.finfo(source_native).nmant  # type: ignore[arg-type]
            max_exact_int = 2**mantissa_bits
            if int_info.max > max_exact_int or int_info.min < -max_exact_int:
                raise ValueError(
                    f"Casting {source_native} to {target_native} may silently lose precision. "
                    f"{source_native} can only exactly represent integers up to 2**{mantissa_bits} "
                    f"({max_exact_int}), but {target_native} has range "
                    f"[{int_info.min}, {int_info.max}]."
                )

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        target_zdtype = self.dtype
        target_native = target_zdtype.to_native_dtype()
        source_native = chunk_spec.dtype.to_native_dtype()

        fill = chunk_spec.fill_value
        fill_arr = np.array([fill], dtype=source_native)

        encode_raw = _extract_raw_map(self.scalar_map, "encode")
        encode_entries = (
            _parse_map_entries(encode_raw, chunk_spec.dtype, self.dtype) if encode_raw else None
        )

        new_fill_arr = _cast_array(
            fill_arr,
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=encode_entries,
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

        encode_raw = _extract_raw_map(self.scalar_map, "encode")
        encode_entries = (
            _parse_map_entries(encode_raw, _chunk_spec.dtype, self.dtype) if encode_raw else None
        )

        result = _cast_array(
            np.asarray(arr),
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=encode_entries,
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = chunk_array.as_ndarray_like()
        target_native = chunk_spec.dtype.to_native_dtype()

        decode_raw = _extract_raw_map(self.scalar_map, "decode")
        decode_entries = (
            _parse_map_entries(decode_raw, self.dtype, chunk_spec.dtype) if decode_raw else None
        )

        result = _cast_array(
            np.asarray(arr),
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=decode_entries,
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        source_itemsize = chunk_spec.dtype.to_native_dtype().itemsize
        target_itemsize = self.dtype.to_native_dtype().itemsize
        if source_itemsize == 0:
            return 0
        num_elements = input_byte_length // source_itemsize
        return num_elements * target_itemsize
