"""Cross-check canonical ndsel bodies against a real TensorStore.

A normalized ndsel `transform` body is, field-for-field, a TensorStore
`IndexTransform` (minus the `kind` discriminator, which the canonical body never
carries). This test loads a handful of finite-bound canonical bodies into
`tensorstore.IndexTransform(json=...)` and confirms that TensorStore's own
`to_json()` re-loads, through our engine layer, into an equivalent transform.

Skipped when tensorstore is not installed. Run it explicitly with:

    uv run --with tensorstore pytest \
        packages/zarr-transforms/tests/test_ndsel_tensorstore.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_transforms.json import transform_from_canonical, transform_to_canonical
from zarr_transforms.transform import IndexTransform

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
    body = transform_to_canonical(transform)

    # (1) The canonical body loads directly as a TensorStore IndexTransform.
    ts_transform = ts.IndexTransform(json=body)

    # (2) TensorStore's own JSON re-loads, through our engine, to an equivalent
    #     transform. Comparing via our canonical form normalizes away
    #     representational choices (index_array_bounds, default omissions) that
    #     both sides make differently but that denote the same selection.
    ts_json = ts_transform.to_json()
    reloaded = transform_from_canonical(ts_json)
    assert transform_to_canonical(reloaded) == transform_to_canonical(transform)
