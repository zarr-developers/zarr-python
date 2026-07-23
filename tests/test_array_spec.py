from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import BufferPrototype, default_buffer_prototype
from zarr.core.buffer.cpu import NDBuffer
from zarr.core.dtype import get_data_type_from_native_dtype

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.core.common import MemoryOrder


def _make_spec(
    *,
    shape: tuple[int, ...] = (4, 4),
    native_dtype: Any = "int16",
    fill_value: Any = 0,
    order: MemoryOrder = "C",
    write_empty_chunks: bool = False,
    prototype: BufferPrototype | None = None,
) -> ArraySpec:
    """Creates an ArraySpec with common defaults"""
    zdtype = get_data_type_from_native_dtype(np.dtype(native_dtype))
    fill_value = zdtype.cast_scalar(fill_value)  # mirrors ArrayV3Metadata's fill_value
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=fill_value,
        config=ArrayConfig(order=order, write_empty_chunks=write_empty_chunks),
        prototype=prototype if prototype is not None else default_buffer_prototype(),
    )


class _AltNDBuffer(NDBuffer):
    """A distinct NDBuffer subclass"""


_ALT_PROTOTYPE = BufferPrototype(
    buffer=default_buffer_prototype().buffer,
    nd_buffer=_AltNDBuffer,
)  # a distinct BufferPrototype with a different nd_buffer subclass


# Difficult / important cases:
# issue #3054: np.void is unhashable when writeable
# nan/NaT aren't self-equal yet must compare equal for a ArraySpec
SPECS = [
    pytest.param({"native_dtype": "int16", "fill_value": 7}, id="int16"),
    pytest.param({"native_dtype": "float64", "fill_value": 1.5}, id="float64"),
    pytest.param({"native_dtype": "float64", "fill_value": float("nan")}, id="float64-nan"),
    pytest.param({"native_dtype": "float64", "fill_value": -0.0}, id="float64-negzero"),
    pytest.param({"native_dtype": "complex128", "fill_value": 1 + 2j}, id="complex128"),
    pytest.param(
        {"native_dtype": "complex128", "fill_value": complex(-0.0, -0.0)},
        id="complex128-negzero",
    ),
    pytest.param({"native_dtype": "bool", "fill_value": True}, id="bool"),
    pytest.param(
        {"native_dtype": "datetime64[s]", "fill_value": np.datetime64("2020-01-01")},
        id="datetime64",
    ),
    pytest.param(
        {"native_dtype": "datetime64[s]", "fill_value": np.datetime64("NaT", "s")},
        id="datetime64-NaT",
    ),
    pytest.param(
        {"native_dtype": [("a", "f8"), ("b", "i8")], "fill_value": (1.0, 2)},
        id="structured-void",
    ),
    pytest.param({"native_dtype": "U5", "fill_value": "hello"}, id="fixed-string"),
    pytest.param({"shape": ()}, id="scalar-shape"),
    pytest.param({"shape": (0,)}, id="zero-size"),
    pytest.param({"order": "F"}, id="order-F"),
]


# Mutations: each mutate kwargs to an uneqal version
def _grow_shape(kw: dict[str, Any]) -> dict[str, Any]:
    return {"shape": (*kw.get("shape", (4, 4)), 1)}


def _flip_order(kw: dict[str, Any]) -> dict[str, Any]:
    return {"order": "F" if kw.get("order", "C") == "C" else "C"}


def _swap_prototype(_kw: dict[str, Any]) -> dict[str, Any]:
    return {"prototype": _ALT_PROTOTYPE}


MUTATIONS = [
    pytest.param(_grow_shape, id="shape"),
    pytest.param(_flip_order, id="order"),
    pytest.param(_swap_prototype, id="prototype"),
]


class TestArraySpecHashEq:
    @pytest.mark.parametrize("kwargs", SPECS)
    def test_hashable(self, kwargs: dict[str, Any]) -> None:
        """Every ArraySpec is hashable, including structured (np.void) fill values."""
        assert isinstance(hash(_make_spec(**kwargs)), int)

    @pytest.mark.parametrize("kwargs", SPECS)
    def test_equal_specs_hash_equal(self, kwargs: dict[str, Any]) -> None:
        """Independently built specs with identical fields are equal and hash equal."""
        a = _make_spec(**kwargs)
        b = _make_spec(**kwargs)
        assert a == b
        assert hash(a) == hash(b)

    @pytest.mark.parametrize("kwargs", SPECS)
    @pytest.mark.parametrize("mutate", MUTATIONS)
    def test_distinct_specs_unequal(
        self,
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> None:
        """Changing one dtype-independent field makes a spec unequal to its base."""
        base = _make_spec(**kwargs)
        variant = _make_spec(**{**kwargs, **mutate(kwargs)})
        assert base != variant
        assert hash(base) != hash(variant)

    @pytest.mark.parametrize(
        ("base", "variant"),
        [
            pytest.param({"fill_value": 0}, {"fill_value": 1}, id="fill_value"),
            pytest.param({"native_dtype": "int16"}, {"native_dtype": "int32"}, id="dtype"),
            pytest.param(
                {"native_dtype": "float32", "fill_value": 1.0},
                {"native_dtype": "float64", "fill_value": 1.0},
                id="dtype-float-promote",
            ),
        ],
    )
    def test_dtype_and_fill_value_matter(
        self, base: dict[str, Any], variant: dict[str, Any]
    ) -> None:
        """dtype and fill_value participate in equality; they can't join the cross
        product because fill_value is coupled to dtype."""
        assert _make_spec(**base) != _make_spec(**variant)

    @pytest.mark.parametrize(
        ("native_dtype", "neg_fill", "pos_fill"),
        [
            pytest.param("float16", -0.0, 0.0, id="float16"),
            pytest.param("float32", -0.0, 0.0, id="float32"),
            pytest.param("float64", -0.0, 0.0, id="float64"),
            pytest.param("complex128", complex(-0.0, -0.0), 0j, id="complex128-both"),
            pytest.param("complex128", complex(0.0, -0.0), 0j, id="complex128-imag"),
            pytest.param("complex128", complex(-0.0, 0.0), 0j, id="complex128-real"),
            pytest.param([("a", "f8")], (-0.0,), (0.0,), id="structured"),
        ],
    )
    def test_signed_zero_fills_are_distinct(
        self, native_dtype: Any, neg_fill: Any, pos_fill: Any
    ) -> None:
        """A -0.0 fill writes different bytes than +0.0, so the specs are not equal."""
        neg = _make_spec(native_dtype=native_dtype, fill_value=neg_fill)
        pos = _make_spec(native_dtype=native_dtype, fill_value=pos_fill)
        assert neg != pos
        assert hash(neg) != hash(pos)
