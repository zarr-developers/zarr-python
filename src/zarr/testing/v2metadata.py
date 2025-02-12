from hypothesis import strategies as st
import hypothesis.extra.numpy as npst

from zarr.testing.strategies import v2_dtypes

@st.composite
def array_metadata_v2_inputs(draw):
    dims = draw(st.integers(min_value=1, max_value=10))
    shape = draw(st.lists(st.integers(min_value=1, max_value=100), min_size=dims, max_size=dims))
    chunks = draw(st.lists(st.integers(min_value=1, max_value=100), min_size=dims, max_size=dims))
    dtype = draw(v2_dtypes())
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(dtype)]))
    order = draw(st.sampled_from(["C", "F"]))
    dimension_separator = draw(st.sampled_from([".", "/"]))
    return {
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
        "fill_value": fill_value,
        "order": order,
        "dimension_separator": dimension_separator,
    }