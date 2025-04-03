from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast

import numpy as np

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype._numpy import DateUnit, TimeUnit

Endianness = Literal["little", "big"]
JSONFloat = float | Literal["NaN", "Infinity", "-Infinity"]


class DataTypeValidationError(ValueError): ...


def check_json_bool(data: JSON) -> TypeGuard[bool]:
    """
    Check if a JSON value is a boolean.

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a boolean, False otherwise.
    """
    return isinstance(data, bool)


def check_json_str(data: JSON) -> TypeGuard[str]:
    """
    Check if a JSON value is a string.

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a string, False otherwise.
    """
    return bool(isinstance(data, str))


def check_json_int(data: JSON) -> TypeGuard[int]:
    """
    Check if a JSON value is an integer.

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is an integer, False otherwise.
    """
    return bool(isinstance(data, int))


def check_json_float_v2(data: JSON) -> TypeGuard[JSONFloat]:
    """
    Check if a JSON value represents a float (v2).

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a float, False otherwise.
    """
    if data == "NaN" or data == "Infinity" or data == "-Infinity":
        return True
    return isinstance(data, float | int)


def check_json_float_v3(data: JSON) -> TypeGuard[JSONFloat]:
    """
    Check if a JSON value represents a float (v3).

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a float, False otherwise.
    """
    # TODO: handle the special JSON serialization of different NaN values
    return check_json_float_v2(data)


def check_json_float(data: JSON, zarr_format: ZarrFormat) -> TypeGuard[float]:
    """
    Check if a JSON value represents a float based on zarr format.

    Parameters
    ----------
    data : JSON
        The JSON value to check.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    Bool
        True if the data is a float, False otherwise.
    """
    if zarr_format == 2:
        return check_json_float_v2(data)
    else:
        return check_json_float_v3(data)


def check_json_complex_float_v3(data: JSON) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    """
    Check if a JSON value represents a complex float, as per the zarr v3 spec

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a complex float, False otherwise.
    """
    return (
        not isinstance(data, str)
        and isinstance(data, Sequence)
        and len(data) == 2
        and check_json_float_v3(data[0])
        and check_json_float_v3(data[1])
    )


def check_json_complex_float_v2(data: JSON) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    """
    Check if a JSON value represents a complex float, as per the behavior of zarr-python 2.x

    Parameters
    ----------
    data : JSON
        The JSON value to check.

    Returns
    -------
    Bool
        True if the data is a complex float, False otherwise.
    """
    return (
        not isinstance(data, str)
        and isinstance(data, Sequence)
        and len(data) == 2
        and check_json_float_v2(data[0])
        and check_json_float_v2(data[1])
    )


def check_json_complex_float(
    data: JSON, zarr_format: ZarrFormat
) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    """
    Check if a JSON value represents a complex float based on zarr format.

    Parameters
    ----------
    data : JSON
        The JSON value to check.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    Bool
        True if the data represents a complex float, False otherwise.
    """
    if zarr_format == 2:
        return check_json_complex_float_v2(data)
    return check_json_complex_float_v3(data)


def float_to_json_v2(data: float | np.floating[Any]) -> JSONFloat:
    """
    Convert a float to JSON (v2).

    Parameters
    ----------
    data : float or np.floating
        The float value to convert.

    Returns
    -------
    JSONFloat
        The JSON representation of the float.
    """
    if np.isnan(data):
        return "NaN"
    elif np.isinf(data):
        return "Infinity" if data > 0 else "-Infinity"
    return float(data)


def float_to_json_v3(data: float | np.floating[Any]) -> JSONFloat:
    """
    Convert a float to JSON (v3).

    Parameters
    ----------
    data : float or np.floating
        The float value to convert.

    Returns
    -------
    JSONFloat
        The JSON representation of the float.
    """
    # v3 can in principle handle distinct NaN values, but numpy does not represent these explicitly
    # so we just reuse the v2 routine here
    return float_to_json_v2(data)


def float_to_json(data: float | np.floating[Any], zarr_format: ZarrFormat) -> JSONFloat:
    """
    Convert a float to JSON, parametrized by the zarr format version.

    Parameters
    ----------
    data : float or np.floating
        The float value to convert.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    JSONFloat
        The JSON representation of the float.
    """
    if zarr_format == 2:
        return float_to_json_v2(data)
    else:
        return float_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def complex_to_json_v2(data: complex | np.complexfloating[Any, Any]) -> tuple[JSONFloat, JSONFloat]:
    """
    Convert a complex number to JSON (v2).

    Parameters
    ----------
    data : complex or np.complexfloating
        The complex value to convert.

    Returns
    -------
    tuple[JSONFloat, JSONFloat]
        The JSON representation of the complex number.
    """
    return float_to_json_v2(data.real), float_to_json_v2(data.imag)


def complex_to_json_v3(data: complex | np.complexfloating[Any, Any]) -> tuple[JSONFloat, JSONFloat]:
    """
    Convert a complex number to JSON (v3).

    Parameters
    ----------
    data : complex or np.complexfloating
        The complex value to convert.

    Returns
    -------
    tuple[JSONFloat, JSONFloat]
        The JSON representation of the complex number.
    """
    return float_to_json_v3(data.real), float_to_json_v3(data.imag)


def complex_float_to_json(
    data: complex | np.complexfloating[Any, Any], zarr_format: ZarrFormat
) -> tuple[JSONFloat, JSONFloat]:
    """
    Convert a complex number to JSON, parametrized by the zarr format version.

    Parameters
    ----------
    data : complex or np.complexfloating
        The complex value to convert.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    tuple[JSONFloat, JSONFloat] or JSONFloat
        The JSON representation of the complex number.
    """
    if zarr_format == 2:
        return complex_to_json_v2(data)
    else:
        return complex_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def bytes_to_json(data: bytes, zarr_format: ZarrFormat) -> str:
    """
    Convert bytes to JSON.

    Parameters
    ----------
    data : bytes
        The bytes to store.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    str
        The bytes encoded as ascii using the base64 alphabet.
    """
    # TODO: decide if we are going to make this implementation zarr format-specific
    return base64.b64encode(data).decode("ascii")


def bytes_from_json(data: str, zarr_format: ZarrFormat) -> bytes:
    """
    Convert a JSON string to bytes

    Parameters
    ----------
    data : str
        The JSON string to convert.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    bytes
        The bytes.
    """
    if zarr_format == 2:
        return base64.b64decode(data.encode("ascii"))
    # TODO: differentiate these as needed. This is a spec question.
    if zarr_format == 3:
        return base64.b64decode(data.encode("ascii"))
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def float_from_json_v2(data: JSONFloat) -> float:
    """
    Convert a JSON float to a float (Zarr v2).

    Parameters
    ----------
    data : JSONFloat
        The JSON float to convert.

    Returns
    -------
    float
        The float value.
    """
    match data:
        case "NaN":
            return float("nan")
        case "Infinity":
            return float("inf")
        case "-Infinity":
            return float("-inf")
        case _:
            return float(data)


def float_from_json_v3(data: JSONFloat) -> float:
    """
    Convert a JSON float to a float (v3).

    Parameters
    ----------
    data : JSONFloat
        The JSON float to convert.

    Returns
    -------
    float
        The float value.
    """
    # todo: support the v3-specific NaN handling
    return float_from_json_v2(data)


def float_from_json(data: JSONFloat, zarr_format: ZarrFormat) -> float:
    """
    Convert a JSON float to a float based on zarr format.

    Parameters
    ----------
    data : JSONFloat
        The JSON float to convert.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    float
        The float value.
    """
    if zarr_format == 2:
        return float_from_json_v2(data)
    else:
        return float_from_json_v3(data)


def complex_float_from_json_v2(data: tuple[JSONFloat, JSONFloat]) -> complex:
    """
    Convert a JSON complex float to a complex number (v2).

    Parameters
    ----------
    data : tuple[JSONFloat, JSONFloat]
        The JSON complex float to convert.

    Returns
    -------
    np.complexfloating
        The complex number.
    """
    return complex(float_from_json_v2(data[0]), float_from_json_v2(data[1]))


def complex_float_from_json_v3(data: tuple[JSONFloat, JSONFloat]) -> complex:
    """
    Convert a JSON complex float to a complex number (v3).

    Parameters
    ----------
    data : tuple[JSONFloat, JSONFloat]
        The JSON complex float to convert.

    Returns
    -------
    np.complexfloating
        The complex number.
    """
    return complex(float_from_json_v3(data[0]), float_from_json_v3(data[1]))


def complex_float_from_json(data: tuple[JSONFloat, JSONFloat], zarr_format: ZarrFormat) -> complex:
    """
    Convert a JSON complex float to a complex number based on zarr format.

    Parameters
    ----------
    data : tuple[JSONFloat, JSONFloat]
        The JSON complex float to convert.
    zarr_format : ZarrFormat
        The zarr format version.

    Returns
    -------
    np.complexfloating
        The complex number.
    """
    if zarr_format == 2:
        return complex_float_from_json_v2(data)
    else:
        return complex_float_from_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def datetime_to_json(data: np.datetime64) -> int:
    """
    Convert a datetime64 to a JSON integer.

    Parameters
    ----------
    data : np.datetime64
        The datetime64 value to convert.

    Returns
    -------
    int
        The JSON representation of the datetime64.
    """
    return data.view(np.int64).item()


def datetime_from_json(data: int, unit: DateUnit | TimeUnit) -> np.datetime64:
    """
    Convert a JSON integer to a datetime64.

    Parameters
    ----------
    data : int
        The JSON integer to convert.
    unit : DateUnit or TimeUnit
        The unit of the datetime64.

    Returns
    -------
    np.datetime64
        The datetime64 value.
    """
    return cast("np.datetime64", np.int64(data).view(f"datetime64[{unit}]"))
