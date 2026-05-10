> **Canonical source:** https://gist.github.com/d-v-b/9fd3fe92f82a24c929129f42a6f11f60
> This file is a local cache. Edit the gist; sync the file from the gist (or push local edits with `gh gist edit 9fd3fe92f82a24c929129f42a6f11f60 -f 2026-05-10-deprecate-blosc-enums-design.md <local-path>`).

# Deprecate `BloscShuffle` and `BloscCname` enums

## Goal

Steer users from `BloscShuffle` / `BloscCname` enum members toward the equivalent literal strings (`"shuffle"`, `"zstd"`, etc.) when constructing a `BloscCodec`. The enum classes remain importable, but accessing a member emits a `DeprecationWarning`. Internal storage of the codec, and the public `cname` / `shuffle` attributes, become literal strings.

## Out of scope

- Removing the `BloscShuffle` / `BloscCname` classes. Deletion is deferred to a future major release.
- Other codecs that accept enum-style parameters (Zstd, etc.).

## User-facing surface after the change

- `BloscCodec(cname="zstd", shuffle="bitshuffle")` — preferred form, no warning.
- `BloscCodec(cname=BloscCname.zstd)` — works, but emits `DeprecationWarning` from the enum-member access *and* from the codec init (the access warning fires first; the init normalization treats the resolved string the same as a direct string).
- `from zarr.codecs import BloscShuffle, BloscCname` — silent (no warning on import).
- `BloscShuffle.shuffle` (member access) — `DeprecationWarning`, returns the string `"shuffle"`.
- `codec.cname`, `codec.shuffle` — return `str` (typed as `CName` / `Shuffle` literals), not enum members.

## Design

### Replace the enums with deprecation shims

In [src/zarr/codecs/blosc.py](../../../src/zarr/codecs/blosc.py), replace `BloscShuffle(Enum)` and `BloscCname(Enum)` with classes whose metaclass intercepts attribute access:

```python
class _DeprecatedStrEnumMeta(type):
    _members: dict[str, str]

    def __getattr__(cls, name: str) -> str:
        members = type.__getattribute__(cls, "_members")
        if name in members:
            warnings.warn(
                f"{cls.__name__}.{name} is deprecated; pass the string "
                f"{members[name]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return members[name]
        raise AttributeError(name)


class BloscShuffle(metaclass=_DeprecatedStrEnumMeta):
    _members = {"noshuffle": "noshuffle", "shuffle": "shuffle", "bitshuffle": "bitshuffle"}

    @staticmethod
    def from_int(num: int) -> Shuffle:
        mapping = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}
        if num not in mapping:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return mapping[num]


class BloscCname(metaclass=_DeprecatedStrEnumMeta):
    _members = {
        "lz4": "lz4", "lz4hc": "lz4hc", "blosclz": "blosclz",
        "zstd": "zstd", "snappy": "snappy", "zlib": "zlib",
    }
```

Notes:
- `BloscShuffle.from_int` is now a `@staticmethod` returning a `str` (was a `@classmethod` returning an enum member). The only internal caller is `migrate_v3._convert_compressor`, which passes the result to `BloscCodec(shuffle=...)`. That call site needs no change because the codec already accepts strings.
- `_members` is read via `type.__getattribute__` to avoid recursing through `__getattr__`.
- No warning fires on `from zarr.codecs import BloscShuffle` — the class object is imported by name, not by member access.

### Detect enum-shaped values in `BloscCodec.__init__`

The init signature stays type-compatible (still accepts `BloscCname | CName` and `BloscShuffle | Shuffle | None`). The body changes to:

1. If `cname` or `shuffle` is an `enum.Enum` instance, emit `DeprecationWarning("Passing a {BloscCname,BloscShuffle} enum to BloscCodec is deprecated; pass a literal string instead.")` and convert via `value.value`. (The new `BloscShuffle` / `BloscCname` shims never *produce* an enum instance — member access already returns a string — so this branch primarily catches code that pickled or otherwise materialized a real `enum.Enum` from the old definition.)
2. Replace `parse_enum(cname, BloscCname)` and `parse_enum(shuffle, BloscShuffle)` with explicit membership checks against the `SHUFFLE` / `CNAME` tuples; raise `ValueError` listing valid options on miss.
3. Internal stores: `typesize: int`, `cname: CName`, `clevel: int`, `shuffle: Shuffle`, `blocksize: int`. Update class-level annotations and the dataclass field declarations.

### Downstream cleanup inside the file

- `to_dict`: drop `.value`; emit `self.cname` / `self.shuffle` directly.
- `evolve_from_array_spec`: use string literals `"bitshuffle"` / `"shuffle"` instead of `BloscShuffle.bitshuffle`.
- `_blosc_codec`: rebuild the `map_shuffle_str_to_int` mapping with string keys; rebuild `cname` directly from `self.cname` (no `.name` needed).
- Update the docstring: `cname` / `shuffle` typed as literal strings; remove the `<BloscCname.zstd: 'zstd'>` example output; drop or rephrase the `See Also` section.

### Internal call sites

- [src/zarr/metadata/migrate_v3.py](../../../src/zarr/metadata/migrate_v3.py): no change needed beyond confirming `BloscShuffle.from_int(...)` still resolves. With `from_int` as a staticmethod on the new class body, attribute access on the class itself goes through `_DeprecatedStrEnumMeta.__getattr__`, which only intercepts `_members` keys — `from_int` is on the class normally and is not deprecated.
- Tests / docs that read `codec.cname.value` or compare against enum members must switch to string comparison.

## Tests

In [tests/test_codecs/test_blosc.py](../../../tests/test_codecs/test_blosc.py):

1. Update `test_tunable_attrs_param`:
   - The parametrized `BloscShuffle.shuffle` case must wrap codec construction in `pytest.warns(DeprecationWarning)`.
   - Assertions like `codec.shuffle == BloscShuffle.bitshuffle` become `codec.shuffle == "bitshuffle"`.
2. New test: `BloscShuffle.shuffle` access raises `DeprecationWarning` and returns `"shuffle"`. Same for `BloscCname.zstd`.
3. New test: `BloscCodec(cname=BloscCname.zstd)` triggers `DeprecationWarning` and produces a codec with `codec.cname == "zstd"`.
4. New test: importing the names is silent (`with warnings.catch_warnings(record=True)` confirms no warning).

## Documentation

- [docs/quick-start.md](../../../docs/quick-start.md), [docs/user-guide/arrays.md](../../../docs/user-guide/arrays.md): replace each `zarr.codecs.BloscShuffle.<name>` / `BloscCname.<name>` with the literal string.
- `BloscCodec` docstring (in [src/zarr/codecs/blosc.py](../../../src/zarr/codecs/blosc.py)): drop enum-typed parameter docs in favor of literal string docs; remove the `<BloscShuffle.bitshuffle: 'bitshuffle'>` example output line.

## Changelog

Add `changes/<PR#>.removal.md` containing one paragraph: "`BloscShuffle` and `BloscCname` enums are now deprecated. Pass the equivalent literal string (e.g. `'zstd'`, `'bitshuffle'`) when constructing `BloscCodec`. The enum classes remain importable but emit `DeprecationWarning` on member access; they will be removed in a future release." Use a placeholder filename like `0000.removal.md` and let the PR-creator rename it.

## Risks

- **Type-check fallout for downstream code** that annotates a variable as `BloscShuffle` and assigns it to `codec.shuffle`. After this change `codec.shuffle: str` so static checkers will flag the mismatch. This is the desired migration nudge, but worth calling out in the changelog.
- **`isinstance(codec.shuffle, BloscShuffle)` checks** in downstream code will silently start returning `False`. Same intent — those callers need to migrate — but harder to discover than a type error.
