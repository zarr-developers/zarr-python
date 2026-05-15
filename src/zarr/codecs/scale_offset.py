from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata


_WIDE_INT = np.dtype(np.int64)


def _encode_fits_natively(dtype: np.dtype[Any], offset: int, scale: int) -> bool:
    """Static range proof: is ``(x - offset) * scale`` always in range for every ``x`` in dtype?

    Uses Python ints (unbounded) to avoid overflow in the proof itself.
    """
    info = np.iinfo(dtype)
    d_lo = int(info.min) - offset
    d_hi = int(info.max) - offset
    # Taking min/max of both products handles negative scale without a sign branch.
    products = (d_lo * scale, d_hi * scale)
    lo, hi = min(products), max(products)
    return info.min <= lo and hi <= info.max


def _decode_fits_natively(dtype: np.dtype[Any], offset: int, scale: int) -> bool:
    """Static range proof for decode: is ``x // scale + offset`` always in range?"""
    info = np.iinfo(dtype)
    # x // scale is bounded by the extremes of x / scale (integer division stays within that range)
    if scale > 0:
        q_lo, q_hi = int(info.min) // scale, int(info.max) // scale
    else:
        q_lo, q_hi = int(info.max) // scale, int(info.min) // scale
    lo, hi = q_lo + offset, q_hi + offset
    return info.min <= lo and hi <= info.max


def _check_int_range(
    values: npt.NDArray[np.integer[Any]], target: np.dtype[np.integer[Any]]
) -> None:
    """Raise if any value is outside the representable range of ``target``.

    Uses a single min/max pass instead of two ``np.any`` passes.
    """
    info = np.iinfo(target)
    lo, hi = values.min(), values.max()
    if lo < info.min or hi > info.max:
        raise ValueError(
            f"scale_offset produced a value outside the range of dtype {target} "
            f"[{info.min}, {info.max}]."
        )


def _check_exact_division(
    arr: npt.NDArray[np.integer[Any]], scale: np.integer[Any], scale_repr: object
) -> None:
    """Raise ValueError if ``arr`` has any element not exactly divisible by ``scale``."""
    if np.any(arr % scale):
        raise ValueError(
            f"scale_offset decode produced a non-zero remainder when dividing by "
            f"scale={scale_repr!r}; result is not exactly representable in dtype {arr.dtype}."
        )


def _encode_int_native(
    arr: npt.NDArray[np.integer[Any]], offset: np.integer[Any], scale: np.integer[Any]
) -> npt.NDArray[np.integer[Any]]:
    """Compute ``(arr - offset) * scale`` directly in ``arr.dtype``.

    This is the fast path; it exists only as a separate function to make the contract with
    ``_encode_fits_natively`` explicit: the caller must have already proved that no ``x`` in
    ``arr.dtype``'s range can overflow, so we can skip widening and range-checking entirely.
    Using it without that proof would silently wrap on overflow.
    """
    return cast("npt.NDArray[np.integer[Any]]", (arr - offset) * scale)


def _encode_int_widened(
    arr: npt.NDArray[np.integer[Any]], offset: np.integer[Any], scale: np.integer[Any]
) -> npt.NDArray[np.integer[Any]]:
    """Overflow-checked integer encode for int8..int64 and uint8..uint32.

    Exists because numpy integer arithmetic silently wraps on overflow, which the spec
    forbids. We widen to int64, perform the arithmetic there (int64 holds the product of any
    two values from these dtypes), range-check against the target dtype, then cast back.
    uint64 cannot use this path because its range exceeds int64 — see ``_encode_uint64``.
    """
    wide_arr = arr.astype(_WIDE_INT, copy=False)
    result = (wide_arr - _WIDE_INT.type(offset)) * _WIDE_INT.type(scale)
    _check_int_range(result, arr.dtype)
    return result.astype(arr.dtype, copy=False)


def _encode_float(
    arr: npt.NDArray[np.floating[Any]], offset: np.floating[Any], scale: np.floating[Any]
) -> npt.NDArray[np.floating[Any]]:
    """Encode float arrays in-dtype, guarding only against silent promotion.

    Float arithmetic doesn't need widening — float64 is already the widest supported dtype,
    and ``inf``/``nan`` from overflow are representable IEEE 754 values, so no range check is
    required by the spec. The one thing that can still go wrong is numpy promoting the
    result to a wider float dtype (e.g. float32 * float64 scalar -> float64), which would
    violate the spec's "arithmetic semantics of the input array's data type" clause.
    """
    result = cast("npt.NDArray[np.floating[Any]]", (arr - offset) * scale)
    if result.dtype != arr.dtype:
        raise ValueError(
            f"scale_offset changed dtype from {arr.dtype} to {result.dtype}. "
            f"Arithmetic must preserve the data type."
        )
    return result


def _check_py_int_range(
    result: np.ndarray[tuple[Any, ...], np.dtype[Any]],
    target: np.dtype[np.unsignedinteger[Any]],
) -> None:
    """Range-check an ``object``-dtype ndarray holding Python ints against ``target``'s iinfo.

    Exists as a uint64-specific counterpart to ``_check_int_range``. That one compares numpy
    integers against ``iinfo``; here the values are unbounded Python ints produced by
    ``_encode_uint64`` / ``_decode_uint64``, so we rely on Python's arbitrary-precision
    comparison to detect values outside the target dtype's range.
    """
    info = np.iinfo(target)
    # np.min/np.max on an object array returns a Python int (which compares correctly with iinfo).
    # Works uniformly for 0-d arrays where .flat iteration is awkward.
    lo = np.min(result)
    hi = np.max(result)
    if lo < int(info.min) or hi > int(info.max):
        raise ValueError(
            f"scale_offset produced a value outside the range of dtype {target} "
            f"[{info.min}, {info.max}]."
        )


def _encode_uint64(
    arr: npt.NDArray[np.unsignedinteger[Any]], offset: int, scale: int
) -> npt.NDArray[np.unsignedinteger[Any]]:
    """Encode uint64 via Python-int arithmetic in an ``object``-dtype array.

    Exists because uint64's range [0, 2**64) exceeds int64, so the int64 widening used by
    ``_encode_int_widened`` would itself overflow. Python ints are unbounded, so computing
    via ``object`` dtype is correct by construction. The trade-off is speed: object-dtype
    arithmetic is interpreted per element and is roughly 10x slower than ufunc paths.
    """
    obj = arr.astype(object, copy=False)
    # np.asarray restores ndarray-ness in the 0-d/scalar edge case.
    result = np.asarray((obj - offset) * scale, dtype=object)
    _check_py_int_range(result, arr.dtype)
    return cast("npt.NDArray[np.unsignedinteger[Any]]", result.astype(arr.dtype, copy=False))


def _decode_uint64(
    arr: npt.NDArray[np.unsignedinteger[Any]], offset: int, scale: int
) -> npt.NDArray[np.unsignedinteger[Any]]:
    """Decode uint64 via Python-int arithmetic. See ``_encode_uint64`` for why."""
    obj = arr.astype(object, copy=False)
    result = np.asarray((obj // scale) + offset, dtype=object)
    _check_py_int_range(result, arr.dtype)
    return cast("npt.NDArray[np.unsignedinteger[Any]]", result.astype(arr.dtype, copy=False))


def _decode_int_native(
    arr: npt.NDArray[np.integer[Any]], offset: np.integer[Any], scale: np.integer[Any]
) -> npt.NDArray[np.integer[Any]]:
    """Compute ``arr // scale + offset`` directly in ``arr.dtype``.

    Fast-path counterpart to ``_encode_int_native``; same contract. Caller must have proved
    via ``_decode_fits_natively`` that the result can't overflow. Divisibility is checked
    upstream in ``_decode`` before this is called, so ``//`` is exact here.
    """
    return cast("npt.NDArray[np.integer[Any]]", (arr // scale) + offset)


def _decode_int_widened(
    arr: npt.NDArray[np.integer[Any]], offset: np.integer[Any], scale: np.integer[Any]
) -> npt.NDArray[np.integer[Any]]:
    """Overflow-checked integer decode for int8..int64 and uint8..uint32.

    Counterpart to ``_encode_int_widened``. Widens to int64 so the addition of ``offset``
    after division can't silently wrap, then range-checks against the target dtype.
    """
    wide_arr = arr.astype(_WIDE_INT, copy=False)
    result = (wide_arr // _WIDE_INT.type(scale)) + _WIDE_INT.type(offset)
    _check_int_range(result, arr.dtype)
    return result.astype(arr.dtype, copy=False)


def _decode_float(
    arr: npt.NDArray[np.floating[Any]], offset: np.floating[Any], scale: np.floating[Any]
) -> npt.NDArray[np.floating[Any]]:
    """Decode float arrays in-dtype, guarding only against silent promotion.

    Counterpart to ``_encode_float``; same reasoning. ``arr / scale`` is true division and
    always well-defined for floats (including ``0/0 = nan`` and ``x/0 = ±inf``), so no range
    or exactness check is needed.
    """
    result = cast("npt.NDArray[np.floating[Any]]", (arr / scale) + offset)
    if result.dtype != arr.dtype:
        raise ValueError(
            f"scale_offset changed dtype from {arr.dtype} to {result.dtype}. "
            f"Arithmetic must preserve the data type."
        )
    return result


def _encode(
    arr: np.ndarray[tuple[Any, ...], np.dtype[Any]],
    offset: np.generic,
    scale: np.generic,
) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
    """Compute ``(arr - offset) * scale`` without silent overflow, returning ``arr.dtype``."""
    # uint64 is split out first because its full range (up to 2**64-1) doesn't fit in int64,
    # so the widening strategy used for every other integer dtype would itself overflow.
    if arr.dtype == np.uint64:
        u_arr = cast("npt.NDArray[np.unsignedinteger[Any]]", arr)
        return _encode_uint64(u_arr, int(offset), int(scale))
    if np.issubdtype(arr.dtype, np.integer):
        i_arr = cast("npt.NDArray[np.integer[Any]]", arr)
        i_offset = cast("np.integer[Any]", offset)
        i_scale = cast("np.integer[Any]", scale)
        # Fast path: if a static proof shows no ``x`` in the dtype's range can overflow,
        # skip the int64 widening and run the arithmetic directly in the input dtype.
        if _encode_fits_natively(arr.dtype, int(offset), int(scale)):
            return _encode_int_native(i_arr, i_offset, i_scale)
        return _encode_int_widened(i_arr, i_offset, i_scale)
    # Float path: arithmetic stays in-dtype (no widening); only guard against numpy
    # silently promoting a narrower float to a wider one via scalar type mismatch.
    f_arr = cast("npt.NDArray[np.floating[Any]]", arr)
    f_offset = cast("np.floating[Any]", offset)
    f_scale = cast("np.floating[Any]", scale)
    return _encode_float(f_arr, f_offset, f_scale)


def _decode(
    arr: np.ndarray[tuple[Any, ...], np.dtype[Any]],
    offset: np.generic,
    scale: np.generic,
    *,
    scale_repr: object,
) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
    """Compute ``arr / scale + offset`` without silent overflow, returning ``arr.dtype``."""
    # uint64: same reasoning as _encode — its range exceeds int64, so the Python-int path is the
    # only correct option. Exactness check runs first so non-divisible inputs fail before the
    # slower object-dtype arithmetic.
    if arr.dtype == np.uint64:
        u_arr = cast("npt.NDArray[np.unsignedinteger[Any]]", arr)
        _check_exact_division(u_arr, cast("np.integer[Any]", scale), scale_repr)
        return _decode_uint64(u_arr, int(offset), int(scale))
    if np.issubdtype(arr.dtype, np.integer):
        i_arr = cast("npt.NDArray[np.integer[Any]]", arr)
        i_offset = cast("np.integer[Any]", offset)
        i_scale = cast("np.integer[Any]", scale)
        # The spec requires decode to use true division and error if the result isn't
        # representable. For integers that means the remainder must be zero; if any element
        # isn't exactly divisible we fail here rather than silently truncating via //.
        _check_exact_division(i_arr, i_scale, scale_repr)
        # Fast path mirrors _encode: static proof that ``x // scale + offset`` stays in dtype.
        if _decode_fits_natively(arr.dtype, int(offset), int(scale)):
            return _decode_int_native(i_arr, i_offset, i_scale)
        return _decode_int_widened(i_arr, i_offset, i_scale)
    # Float path: division is well-defined; only guard against dtype promotion.
    f_arr = cast("npt.NDArray[np.floating[Any]]", arr)
    f_offset = cast("np.floating[Any]", offset)
    f_scale = cast("np.floating[Any]", scale)
    return _decode_float(f_arr, f_offset, f_scale)


@dataclass(frozen=True)
class ScaleOffset(ArrayArrayCodec):
    """Scale-offset array-to-array codec.

    Encodes values with `out = (in - offset) * scale` and decodes with
    `out = (in / scale) + offset`, using the input array's data type semantics.
    Intermediate or final values that are not representable in that dtype are reported
    as errors (integer overflow, unsigned underflow, non-exact integer division).

    Parameters
    ----------
    offset : int, float, or str
        Value subtracted during encoding. Strings preserve the exact JSON
        representation when round-tripping metadata. Default is 0.
    scale : int, float, or str
        Value multiplied during encoding (after offset subtraction). Strings
        preserve the exact JSON representation when round-tripping metadata.
        Default is 1.

    Attributes
    ----------
    offset : int, float, or str
        The offset value, as supplied to the constructor.
    scale : int, float, or str
        The scale value, as supplied to the constructor.

    References
    ----------

    - The `scale_offset` codec spec: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/scale_offset
    """

    is_fixed_size = True

    offset: int | float | str
    scale: int | float | str

    def __init__(self, *, offset: object = 0, scale: object = 1) -> None:
        if not isinstance(offset, int | float | str):
            raise TypeError(f"offset must be a number or string, got {type(offset).__name__}")
        if not isinstance(scale, int | float | str):
            raise TypeError(f"scale must be a number or string, got {type(scale).__name__}")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "scale_offset", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        if self.offset == 0 and self.scale == 1:
            return {"name": "scale_offset"}
        config: dict[str, JSON] = {}
        if self.offset != 0:
            config["offset"] = self.offset
        if self.scale != 1:
            config["scale"] = self.scale
        return {"name": "scale_offset", "configuration": config}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.integer) and not np.issubdtype(native, np.floating):
            raise ValueError(
                f"scale_offset codec only supports integer and floating-point data types. "
                f"Got {dtype}."
            )
        if self.scale == 0:
            raise ValueError("scale_offset scale must be non-zero.")
        for name, value in [("offset", self.offset), ("scale", self.scale)]:
            try:
                dtype.from_json_scalar(value, zarr_format=3)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(
                    f"scale_offset {name} value {value!r} is not representable in dtype {native}."
                ) from e

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        zdtype = chunk_spec.dtype
        fill = np.asarray(zdtype.cast_scalar(chunk_spec.fill_value))
        offset = cast("np.generic", zdtype.from_json_scalar(self.offset, zarr_format=3))
        scale = cast("np.generic", zdtype.from_json_scalar(self.scale, zarr_format=3))
        new_fill = _encode(fill, offset, scale)
        return replace(chunk_spec, fill_value=new_fill.reshape(()).item())

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = cast("np.ndarray[tuple[Any, ...], np.dtype[Any]]", chunk_array.as_ndarray_like())
        zdtype = chunk_spec.dtype
        offset = cast("np.generic", zdtype.from_json_scalar(self.offset, zarr_format=3))
        scale = cast("np.generic", zdtype.from_json_scalar(self.scale, zarr_format=3))
        result = _decode(arr, offset, scale, scale_repr=self.scale)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = cast("np.ndarray[tuple[Any, ...], np.dtype[Any]]", chunk_array.as_ndarray_like())
        zdtype = chunk_spec.dtype
        offset = cast("np.generic", zdtype.from_json_scalar(self.offset, zarr_format=3))
        scale = cast("np.generic", zdtype.from_json_scalar(self.scale, zarr_format=3))
        result = _encode(arr, offset, scale)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, _chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
