from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype import get_data_type_from_json

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]

ScalarMapJSON = dict[str, list[list[JSON]]]

# Pre-parsed scalar map entry: (source_float, target_float, source_is_nan)
_MapEntry = tuple[float, float, bool]


def _special_float(s: str) -> float:
    """Convert special float string representations to float values."""
    if s == "NaN":
        return float("nan")
    if s in ("+Infinity", "Infinity"):
        return float("inf")
    if s == "-Infinity":
        return float("-inf")
    return float(s)


def _parse_map_entries(mapping: dict[str, str]) -> list[_MapEntry]:
    """Pre-parse a scalar map dict into a list of (src, tgt, src_is_nan) tuples."""
    entries: list[_MapEntry] = []
    for src_str, tgt_str in mapping.items():
        src = _special_float(src_str)
        tgt = _special_float(tgt_str)
        entries.append((src, tgt, np.isnan(src)))
    return entries


def _apply_scalar_map(work: np.ndarray, entries: list[_MapEntry]) -> None:
    """Apply scalar map entries in-place. Single pass per entry."""
    for src, tgt, src_is_nan in entries:
        if src_is_nan:
            mask = np.isnan(work)
        else:
            mask = work == src
        work[mask] = tgt


def _round_inplace(arr: np.ndarray, mode: RoundingMode) -> np.ndarray:
    """Round array, returning result (may or may not be a new array).

    For nearest-away, requires 3 numpy ops. All others are a single op.
    """
    if mode == "nearest-even":
        return np.rint(arr)
    elif mode == "towards-zero":
        return np.trunc(arr)
    elif mode == "towards-positive":
        return np.ceil(arr)
    elif mode == "towards-negative":
        return np.floor(arr)
    elif mode == "nearest-away":
        return np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    raise ValueError(f"Unknown rounding mode: {mode}")


def _cast_array(
    arr: np.ndarray,
    target_dtype: np.dtype,
    rounding: RoundingMode,
    out_of_range: OutOfRangeMode | None,
    scalar_map_entries: list[_MapEntry] | None,
) -> np.ndarray:
    """Cast an array to target_dtype with rounding, out-of-range, and scalar_map handling.

    Optimized to minimize allocations and passes over the data.
    For the simple case (no scalar_map, no rounding needed, no out-of-range),
    this is essentially just ``arr.astype(target_dtype)``.
    """
    src_is_int = np.issubdtype(arr.dtype, np.integer)
    src_is_float = np.issubdtype(arr.dtype, np.floating)
    tgt_is_int = np.issubdtype(target_dtype, np.integer)
    tgt_is_float = np.issubdtype(target_dtype, np.floating)

    # Fast path: float→float with no scalar_map — single astype
    if src_is_float and tgt_is_float and not scalar_map_entries:
        return arr.astype(target_dtype)

    # Fast path: int→float with no scalar_map — single astype
    if src_is_int and tgt_is_float and not scalar_map_entries:
        return arr.astype(target_dtype)

    # Fast path: int→int with no scalar_map — check range then astype
    if src_is_int and tgt_is_int and not scalar_map_entries:
        # Check if source range could exceed target range
        if arr.dtype.itemsize > target_dtype.itemsize or arr.dtype != target_dtype:
            info = np.iinfo(target_dtype)
            lo, hi = int(info.min), int(info.max)
            arr_min, arr_max = int(arr.min()), int(arr.max())
            if arr_min >= lo and arr_max <= hi:
                return arr.astype(target_dtype)
            if out_of_range == "clamp":
                return np.clip(arr, lo, hi).astype(target_dtype)
            elif out_of_range == "wrap":
                range_size = hi - lo + 1
                return ((arr.astype(np.int64) - lo) % range_size + lo).astype(target_dtype)
            else:
                raise ValueError(
                    f"Values out of range for {target_dtype} and no out_of_range policy set"
                )
        return arr.astype(target_dtype)

    # float→int: needs rounding, range check, possibly scalar_map
    if src_is_float and tgt_is_int:
        # Work in float64 for the arithmetic
        if arr.dtype != np.float64:
            work = arr.astype(np.float64)
        else:
            work = arr.copy()

        if scalar_map_entries:
            _apply_scalar_map(work, scalar_map_entries)

        # Check for unmapped NaN/Inf
        bad = np.isnan(work) | np.isinf(work)
        if bad.any():
            raise ValueError("Cannot cast NaN or Infinity to integer type without scalar_map")

        work = _round_inplace(work, rounding)

        info = np.iinfo(target_dtype)
        lo, hi = float(info.min), float(info.max)
        if out_of_range == "clamp":
            np.clip(work, lo, hi, out=work)
        elif out_of_range == "wrap":
            range_size = int(info.max) - int(info.min) + 1
            oor = (work < lo) | (work > hi)
            if oor.any():
                work[oor] = (work[oor].astype(np.int64) - int(info.min)) % range_size + int(
                    info.min
                )
        elif (work.min() < lo) or (work.max() > hi):
            raise ValueError(
                f"Values out of range for {target_dtype} and no out_of_range policy set"
            )

        return work.astype(target_dtype)

    # int→float with scalar_map
    if src_is_int and tgt_is_float and scalar_map_entries:
        work = arr.astype(np.float64)
        _apply_scalar_map(work, scalar_map_entries)
        return work.astype(target_dtype)

    # float→float with scalar_map
    if src_is_float and tgt_is_float and scalar_map_entries:
        work = arr.copy()
        _apply_scalar_map(work, scalar_map_entries)
        return work.astype(target_dtype)

    # int→int with scalar_map
    if src_is_int and tgt_is_int and scalar_map_entries:
        work = arr.astype(np.int64)
        _apply_scalar_map(work, scalar_map_entries)
        info = np.iinfo(target_dtype)
        lo, hi = int(info.min), int(info.max)
        w_min, w_max = int(work.min()), int(work.max())
        if w_min < lo or w_max > hi:
            if out_of_range == "clamp":
                np.clip(work, lo, hi, out=work)
            elif out_of_range == "wrap":
                range_size = hi - lo + 1
                oor = (work < lo) | (work > hi)
                work[oor] = (work[oor] - lo) % range_size + lo
            else:
                raise ValueError(
                    f"Values out of range for {target_dtype} and no out_of_range policy set"
                )
        return work.astype(target_dtype)

    # Fallback
    return arr.astype(target_dtype)


def _parse_scalar_map(
    data: ScalarMapJSON | None,
) -> tuple[list[_MapEntry] | None, list[_MapEntry] | None]:
    """Parse scalar_map JSON into pre-parsed encode and decode entry lists.

    Returns (encode_entries, decode_entries). Either may be None.
    """
    if data is None:
        return None, None
    encode_raw: dict[str, str] = {}
    decode_raw: dict[str, str] = {}
    for src, tgt in data.get("encode", []):
        encode_raw[str(src)] = str(tgt)
    for src, tgt in data.get("decode", []):
        decode_raw[str(src)] = str(tgt)
    return (
        _parse_map_entries(encode_raw) if encode_raw else None,
        _parse_map_entries(decode_raw) if decode_raw else None,
    )


@dataclass(frozen=True)
class CastValueCodec(ArrayArrayCodec):
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

    data_type: str
    rounding: RoundingMode
    out_of_range: OutOfRangeMode | None
    scalar_map: ScalarMapJSON | None

    def __init__(
        self,
        *,
        data_type: str,
        rounding: RoundingMode = "nearest-even",
        out_of_range: OutOfRangeMode | None = None,
        scalar_map: ScalarMapJSON | None = None,
    ) -> None:
        object.__setattr__(self, "data_type", data_type)
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
        config: dict[str, JSON] = {"data_type": self.data_type}
        if self.rounding != "nearest-even":
            config["rounding"] = self.rounding
        if self.out_of_range is not None:
            config["out_of_range"] = self.out_of_range
        if self.scalar_map is not None:
            config["scalar_map"] = self.scalar_map
        return {"name": "cast_value", "configuration": config}

    def _target_zdtype(self) -> ZDType[TBaseDType, TBaseScalar]:
        return get_data_type_from_json(self.data_type, zarr_format=3)

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        source_native = dtype.to_native_dtype()
        target_native = self._target_zdtype().to_native_dtype()
        for label, dt in [("source", source_native), ("target", target_native)]:
            if not np.issubdtype(dt, np.integer) and not np.issubdtype(dt, np.floating):
                raise ValueError(
                    f"cast_value codec only supports integer and floating-point data types. "
                    f"Got {label} dtype {dt}."
                )
        if self.out_of_range == "wrap":
            if not np.issubdtype(target_native, np.integer):
                raise ValueError("out_of_range='wrap' is only valid for integer target types.")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        target_zdtype = self._target_zdtype()
        target_native = target_zdtype.to_native_dtype()
        source_native = chunk_spec.dtype.to_native_dtype()

        fill = chunk_spec.fill_value
        fill_arr = np.array([fill], dtype=source_native)

        encode_entries, _ = _parse_scalar_map(self.scalar_map)

        new_fill_arr = _cast_array(
            fill_arr, target_native, self.rounding, self.out_of_range, encode_entries
        )
        new_fill = target_native.type(new_fill_arr[0])

        return replace(chunk_spec, dtype=target_zdtype, fill_value=new_fill)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = chunk_array.as_ndarray_like()
        target_native = self._target_zdtype().to_native_dtype()

        encode_entries, _ = _parse_scalar_map(self.scalar_map)

        result = _cast_array(
            np.asarray(arr), target_native, self.rounding, self.out_of_range, encode_entries
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

        _, decode_entries = _parse_scalar_map(self.scalar_map)

        result = _cast_array(
            np.asarray(arr), target_native, self.rounding, self.out_of_range, decode_entries
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
        target_itemsize = self._target_zdtype().to_native_dtype().itemsize
        if source_itemsize == 0:
            return 0
        num_elements = input_byte_length // source_itemsize
        return num_elements * target_itemsize
