from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from zarr.core.config import config
from zarr.errors import ZarrFutureWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.common import JSON

AttributeJSONPolicy = Literal["allow", "warn", "raise"]
ATTRIBUTE_JSON_POLICIES: Final = ("allow", "warn", "raise")

# Key types that `json.dumps` silently converts to strings. Other non-string
# key types make `json.dumps` raise, so they need no check here.
_COERCED_KEY_TYPES: Final = (int, float, bool, type(None))

# The root of the zarr package, used to attribute warnings to the caller outside zarr.
_ZARR_PACKAGE_ROOT: Final = str(Path(__file__).parents[2])


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return dict(data)


def _iter_non_json(
    value: Any, path: tuple[object, ...] = ()
) -> Iterator[tuple[Literal["non_string_key", "non_finite_float"], tuple[object, ...]]]:
    """Yield `(kind, path)` for every part of `value` that `json.dumps` accepts
    but that is not round-trippable JSON.

    `non_string_key` marks a dict key that `json.dumps` converts to a string;
    `non_finite_float` marks a NaN or infinite float, which `json.dumps` writes
    as a non-standard `NaN` / `Infinity` literal.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, _COERCED_KEY_TYPES):
                yield "non_string_key", (*path, key)
            yield from _iter_non_json(item, (*path, key))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_non_json(item, (*path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        yield "non_finite_float", path


def _format_path(path: tuple[object, ...]) -> str:
    return "attributes" + "".join(f"[{part!r}]" for part in path)


def _get_policy(key: str) -> AttributeJSONPolicy:
    policy = config.get(key)
    if policy not in ATTRIBUTE_JSON_POLICIES:
        raise ValueError(
            f"Invalid value for config option {key!r}: {policy!r}. "
            f"Expected one of {ATTRIBUTE_JSON_POLICIES}."
        )
    return cast("AttributeJSONPolicy", policy)


def check_attributes_json(attributes: dict[str, JSON]) -> None:
    """Apply the configured policy to attributes that are not valid JSON.

    Called when attributes are serialized for writing. Two kinds of value are
    checked, each under its own config option:

    - `attributes.non_string_keys`: dict keys that are not strings. JSON object
      keys are strings, so these are written as strings and read back as
      different keys. Defaults to `"warn"`; this will become an error in a
      future version of zarr.
    - `attributes.non_finite_floats`: NaN and infinite floats. JSON cannot
      represent them, so they are written as non-standard `NaN` / `Infinity`
      literals that other JSON parsers may reject. Defaults to `"allow"`.

    Each option is `"allow"` (write without comment), `"warn"` (write, and emit
    a `ZarrFutureWarning`) or `"raise"` (refuse to write).
    """
    problems: dict[str, list[tuple[object, ...]]] = {
        "non_string_key": [],
        "non_finite_float": [],
    }
    for kind, path in _iter_non_json(attributes):
        problems[kind].append(path)

    if problems["non_string_key"]:
        policy = _get_policy("attributes.non_string_keys")
        if policy != "allow":
            paths = ", ".join(_format_path(p) for p in problems["non_string_key"])
            detail = (
                f"Attribute keys must be strings, but these keys are not: {paths}. "
                "JSON object keys are strings, so these keys are written to the store as "
                "strings and read back as different keys."
            )
            if policy == "raise":
                raise TypeError(detail)
            warnings.warn(
                f"{detail} This will be an error in a future version of zarr. Convert the "
                "keys to strings, or set the config option 'attributes.non_string_keys' to "
                "'raise' to opt in to the error now, or to 'allow' to keep the current "
                "behavior without this warning.",
                ZarrFutureWarning,
                skip_file_prefixes=(_ZARR_PACKAGE_ROOT,),
            )

    if problems["non_finite_float"]:
        policy = _get_policy("attributes.non_finite_floats")
        if policy != "allow":
            paths = ", ".join(_format_path(p) for p in problems["non_finite_float"])
            detail = (
                f"Attribute values must be valid JSON, but these values are NaN or infinite: "
                f"{paths}. JSON cannot represent them, so they are written as non-standard "
                "NaN / Infinity literals that other JSON parsers may reject."
            )
            if policy == "raise":
                raise ValueError(detail)
            warnings.warn(
                f"{detail} Set the config option 'attributes.non_finite_floats' to 'raise' "
                "to refuse to write them, or to 'allow' to write them without this warning.",
                ZarrFutureWarning,
                skip_file_prefixes=(_ZARR_PACKAGE_ROOT,),
            )
