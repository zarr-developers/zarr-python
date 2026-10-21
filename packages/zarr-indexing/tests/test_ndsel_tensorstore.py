"""Cross-check canonical ndsel bodies against a real TensorStore.

Canonical ndsel bodies use TensorStore's `IndexTransform` field vocabulary,
but the consumers have different validation constraints. This test loads a
handful of finite-bound canonical bodies supported by both implementations into
`tensorstore.IndexTransform(json=...)` and confirms that TensorStore's own
`to_json()` re-loads, through our engine layer, into an equivalent transform.

Skipped when tensorstore is not installed. Run it explicitly with:

    hatch run test.py3.12-optional:pytest \
        packages/zarr-indexing/tests/test_ndsel_tensorstore.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_indexing.transform import IndexTransform

ts = pytest.importorskip("tensorstore")


def _canonical_transforms() -> list[IndexTransform]:
    base = IndexTransform.from_shape((10, 20))
    return [
        base,  # identity
        base[2:8:2, :],  # strided DimensionMap + identity
        base[3, :],  # integer index -> ConstantMap + DimensionMap
        base.oindex[np.array([1, 5, 9]), :],  # orthogonal index_array
        IndexTransform.from_shape((10, 20, 30)).vindex[
            np.array([1, 3]), np.array([2, 4]), :
        ],  # correlated index_arrays + residual slice
    ]


@pytest.mark.parametrize("transform", _canonical_transforms())
def test_body_loads_in_tensorstore_and_round_trips(transform: IndexTransform) -> None:
    body = transform.to_json()

    # (1) The canonical body loads directly as a TensorStore IndexTransform.
    ts_transform = ts.IndexTransform(json=body)

    # (2) TensorStore's own JSON re-loads, through our engine, to an equivalent
    #     transform. Comparing via our canonical form normalizes away
    #     representational choices (index_array_bounds, default omissions) that
    #     both sides make differently but that denote the same selection.
    ts_json = ts_transform.to_json()
    reloaded = IndexTransform.from_json(ts_json)
    assert reloaded.to_json() == transform.to_json()
