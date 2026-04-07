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
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype import get_data_type_from_json

if TYPE_CHECKING:
    from typing import NotRequired, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

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


class ScalarMap(TypedDict):
    """
    The normalized, in-memory form of a scalar map.
    """

    encode: NotRequired[Mapping[str | float | int, str | float | int]]
    decode: NotRequired[Mapping[str | float | int, str | float | int]]


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


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CastValue(ArrayArrayCodec):
    """Cast-value array-to-array codec.

    Value-converts array elements to a new data type during encoding,
    and back to the original data type during decoding.

    Requires the ``cast-value-rs`` package for the actual casting logic.

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
        Explicit mapping from input scalars to output scalars.

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
        chunk_grid: ChunkGrid,
    ) -> None:
        source_native = dtype.to_native_dtype()
        target_native = self.dtype.to_native_dtype()
        for label, dt in [("source", source_native), ("target", target_native)]:
            if not np.issubdtype(dt, np.integer) and not np.issubdtype(dt, np.floating):
                raise ValueError(
                    f"The cast_value codec only supports integer and floating-point data types. "
                    f"Got {label} dtype {dt}."
                )
        if self.out_of_range == "wrap" and not np.issubdtype(target_native, np.integer):
            raise ValueError("out_of_range='wrap' is only valid for integer target types.")

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
        scalar_map_entries: dict[float, float] | None = None
        if scalar_map is not None:
            scalar_map_entries = {float(k): float(v) for k, v in scalar_map.items()}
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
        source_itemsize = chunk_spec.dtype.to_native_dtype().itemsize
        target_itemsize = self.dtype.to_native_dtype().itemsize
        if source_itemsize == 0 or target_itemsize == 0:
            raise ValueError(
                "cast_value codec requires fixed-size data types. "
                f"Got source itemsize={source_itemsize}, target itemsize={target_itemsize}."
            )
        num_elements = input_byte_length // source_itemsize
        return num_elements * target_itemsize
