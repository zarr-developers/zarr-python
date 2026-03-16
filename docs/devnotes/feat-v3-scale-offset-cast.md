# scale_offset and cast_value codecs

Source: https://github.com/zarr-developers/zarr-extensions/pull/43

## Overview

Two array-to-array codecs for zarr v3, designed to work together for the
common pattern of storing floating-point data as compressed integers.

---

## scale_offset

**Type:** array -> array (does NOT change dtype)

**Encode:** `out = (in - offset) * scale`
**Decode:** `out = (in / scale) + offset`

### Parameters
- `offset` (optional, float): scalar subtracted during encoding. Default: 0.
- `scale` (optional, float): scalar multiplied during encoding (after offset subtraction). Default: 1.

### Key rules
- Arithmetic uses the input array's own data type semantics (no implicit promotion).
- If neither scale nor offset is given, `configuration` may be omitted (codec is a no-op).
- Fill value is transformed through the codec (encode direction).
- Only valid for real-number data types (int/uint/float families). Complex dtypes are rejected at validation time.

### JSON
```json
{"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}}
```
When both offset and scale are defaults: `{"name": "scale_offset"}` (no configuration key).

---

## cast_value

**Type:** array -> array (CHANGES dtype)

**Purpose:** Value-convert (not binary-reinterpret) array elements to a new data type.

### Parameters
- `data_type` (required): target zarr v3 data type name (e.g. `"uint8"`, `"float32"`).
  Internally stored as a `ZDType` instance, resolved via `get_data_type_from_json`.
- `rounding` (optional): how to round when casting float to int.
  Values: `"nearest-even"` (default), `"towards-zero"`, `"towards-positive"`,
  `"towards-negative"`, `"nearest-away"`.
- `out_of_range` (optional): what to do when a value is outside the target's range.
  Values: `"clamp"`, `"wrap"`. If absent, out-of-range values raise an error.
  `"wrap"` is only valid for integer target types.
- `scalar_map` (optional): explicit value overrides.
  `{"encode": [[input, output], ...], "decode": [[input, output], ...]}`.
  Applied BEFORE rounding/out_of_range. Each entry's source is deserialized using the
  source dtype and target using the target dtype (via `ZDType.from_json_scalar`),
  preserving full precision for both sides.

### Cast procedure (`_cast_array_impl`)

Dispatches on `(src_type, tgt_type, has_map)` where src/tgt are `"int"` or `"float"`:

| Source | Target | scalar_map | Procedure |
|--------|--------|------------|-----------|
| any    | float  | no         | `arr.astype(target_dtype)` |
| int    | float  | yes        | widen to float64, apply map, cast |
| float  | float  | yes        | copy, apply map, cast |
| int    | int    | no         | range check, then astype |
| int    | int    | yes        | widen to int64, apply map, range check |
| float  | int    | any        | widen to float64, apply map (if any), reject NaN/Inf, round, range check |

All casts are wrapped in `np.errstate(over='raise', invalid='raise')` to convert
numpy overflow/invalid warnings to hard errors.

### Validation checks
- Only integer and floating-point dtypes are allowed (both source and target).
- `out_of_range='wrap'` is rejected for non-integer target types.
- Int-to-float casts are rejected if the float type's mantissa cannot exactly represent
  the full integer range (e.g. int64 -> float64 is rejected because float64 has only
  52 mantissa bits, but int64 has values up to 2^63-1). Same check applies for the
  float-to-int decode direction.

### Special values
- NaN: detected dynamically via `isinstance(src, (float, np.floating)) and np.isnan(src)`.
  NaN-to-integer casts error unless `scalar_map` provides a mapping.
  Hex-encoded NaN strings (e.g. `"0x7fc00001"`) preserve NaN payloads per the zarr v3 spec.
- `_check_int_range` handles out-of-range integer values with clamp (via `np.clip`) or
  wrap (via modular arithmetic).

### Fill value
- Cast using the same `_cast_array` path as array elements, including scalar_map and rounding.
- Done in `resolve_metadata`, which also changes the chunk spec's dtype to the target.

### JSON
```json
{
  "name": "cast_value",
  "configuration": {
    "data_type": "uint8",
    "rounding": "nearest-even",
    "out_of_range": "clamp",
    "scalar_map": {
      "encode": [["NaN", 0], ["+Infinity", 0], ["-Infinity", 0]],
      "decode": [[0, "NaN"]]
    }
  }
}
```
Only non-default fields are serialized (rounding and out_of_range are omitted when default).

---

## Typical combined usage

```json
{
  "data_type": "float64",
  "fill_value": "NaN",
  "codecs": [
    {"name": "scale_offset", "configuration": {"offset": -10, "scale": 0.1}},
    {"name": "cast_value", "configuration": {
      "data_type": "uint8",
      "rounding": "nearest-even",
      "scalar_map": {"encode": [["NaN", 0]], "decode": [[0, "NaN"]]}
    }},
    "bytes"
  ]
}
```

---

## Implementation notes

### Module structure
- `src/zarr/codecs/scale_offset.py` — `ScaleOffset` class
- `src/zarr/codecs/cast_value.py` — `CastValue` class and casting helpers
- `tests/test_codecs/test_scale_offset.py` — ScaleOffset tests
- `tests/test_codecs/test_cast_value.py` — CastValue tests + combined pipeline tests

### scale_offset
- `@dataclass(kw_only=True, frozen=True)`, subclasses `ArrayArrayCodec`.
- Uses `ScaleOffsetJSON` (a `NamedConfig` TypedDict) for typed serialization.
- `from_dict` uses `parse_named_configuration(data, "scale_offset", require_configuration=False)`.
- `to_dict` omits the `configuration` key entirely when both offset=0 and scale=1.
- `resolve_metadata`: transforms fill_value via `(fill - offset) * scale`, dtype unchanged.
- `_encode_sync`: `(arr - offset) * scale` using the array's own dtype.
- `_decode_sync`: `(arr / scale) + offset` using the array's own dtype.
- `is_fixed_size = True`, `compute_encoded_size` returns input size unchanged.

### cast_value
- `@dataclass(frozen=True)` with custom `__init__` (accepts `data_type: str | ZDType`).
- Stores `dtype: ZDType` (not a string). String data_type is resolved via `get_data_type_from_json`.
- `from_dict` uses `parse_named_configuration(data, "cast_value", require_configuration=True)`.
- `to_dict` serializes dtype via `self.dtype.to_json(zarr_format=3)`, only includes
  non-default rounding/out_of_range/scalar_map.
- `resolve_metadata`: casts fill value, changes chunk spec dtype to target.
- `_encode_sync` / `_decode_sync`: delegate to `_cast_array`, threading the appropriate
  scalar_map direction ("encode" or "decode") and the correct src/tgt ZDType pair for
  scalar map deserialization.
- `compute_encoded_size`: scales by `target_itemsize / source_itemsize`.

### Key helpers (cast_value.py)
- `_cast_array` — public entry point, wraps `_cast_array_impl` with `np.errstate`.
- `_cast_array_impl` — match-based dispatch on `(src_type, tgt_type, has_map)`.
- `_check_int_range` — integer range check with clamp/wrap/error.
- `_round_inplace` — rounding dispatch (rint, trunc, ceil, floor, nearest-away).
- `_apply_scalar_map` — in-place value remapping with NaN-aware matching.
- `_parse_map_entries` — deserializes scalar_map JSON using separate src/tgt ZDType instances.
- `_extract_raw_map` — extracts "encode" or "decode" direction from ScalarMapJSON.

### Key design decisions
1. Encode = `(in - offset) * scale` (subtract, not add) — matches HDF5 and numcodecs.
2. No implicit precision promotion — arithmetic stays in the input dtype.
3. `out_of_range` defaults to error (not clamp).
4. `scalar_map` entries are typed: each side is deserialized with its own ZDType,
   so int64 scalars don't lose precision through float64 intermediaries.
5. Fill value is cast through the same `_cast_array` path as data elements.
6. Int-to-float precision loss is caught at validate time (mantissa bit check).
7. Runtime overflow/invalid is caught via `np.errstate(over='raise', invalid='raise')`.
