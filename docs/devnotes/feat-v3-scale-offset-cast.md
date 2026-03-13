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
- `offset` (optional): scalar subtracted during encoding. Default: 0 (additive identity).
  Serialized in JSON using the zarr v3 fill-value encoding for the array's dtype.
- `scale` (optional): scalar multiplied during encoding (after offset subtraction). Default: 1
  (multiplicative identity). Same JSON encoding as offset.

### Key rules
- Arithmetic MUST use the input array's own data type semantics (no implicit promotion).
- If any intermediate or final value is unrepresentable in that dtype, error.
- If neither scale nor offset is given, `configuration` may be omitted (codec is a no-op).
- Fill value MUST be transformed through the codec (encode direction).
- Only valid for real-number data types (int/uint/float families).

### JSON
```json
{"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}}
```

---

## cast_value

**Type:** array -> array (CHANGES dtype)

**Purpose:** Value-convert (not binary-reinterpret) array elements to a new data type.

### Parameters
- `data_type` (required): target zarr v3 data type.
- `rounding` (optional): how to round when exact representation is impossible.
  Values: `"nearest-even"` (default), `"towards-zero"`, `"towards-positive"`,
  `"towards-negative"`, `"nearest-away"`.
- `out_of_range` (optional): what to do when a value is outside the target's range.
  Values: `"clamp"`, `"wrap"`. If absent, out-of-range MUST error.
  `"wrap"` only valid for integral two's-complement types.
- `scalar_map` (optional): explicit value overrides.
  `{"encode": [[input, output], ...], "decode": [[input, output], ...]}`.
  Evaluated BEFORE rounding/out_of_range.

### Casting procedure (same for encode and decode, swapping source/target)
1. Check scalar_map — if input matches a key, use mapped value.
2. Check exact representability — if yes, use directly.
3. Apply rounding and out_of_range rules.
4. If none apply, MUST error.

### Special values
- NaN propagates between IEEE 754 types unless scalar_map overrides.
- Signed zero preserved between IEEE 754 types.
- If target doesn't support NaN/infinity and input has them, MUST error
  unless scalar_map provides a mapping.

### Fill value
- MUST be cast using same semantics as elements.
- Implementations SHOULD validate fill value survives round-trip at metadata
  construction time.

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

## Implementation notes for zarr-python

### scale_offset
- Subclass `ArrayArrayCodec`.
- `resolve_metadata`: transform fill_value via `(fill - offset) * scale`, keep dtype.
- `_encode_single`: `(array - offset) * scale` using numpy with same dtype.
- `_decode_single`: `(array / scale) + offset` using numpy with same dtype.
- `is_fixed_size = True`.

### cast_value
- Subclass `ArrayArrayCodec`.
- `resolve_metadata`: change dtype to target dtype, cast fill_value.
- `_encode_single`: cast array from input dtype to target dtype.
- `_decode_single`: cast array from target dtype back to input dtype.
- Needs the input dtype stored (from `evolve_from_array_spec` or `resolve_metadata`).
- `is_fixed_size = True` (for fixed-size types).
- Initial implementation: support `rounding` and `out_of_range` for common cases.
  `scalar_map` adds complexity but is needed for NaN handling.

### Key design decisions from PR review
1. Encode = `(in - offset) * scale` (subtract, not add) — matches HDF5 and numcodecs.
2. No implicit precision promotion — arithmetic stays in the input dtype.
3. `out_of_range` defaults to error (not clamp).
4. `scalar_map` was added specifically to handle NaN-to-integer mappings.
5. Fill value must round-trip exactly through the codec chain.
6. Name uses underscore: `scale_offset`, `cast_value`.
