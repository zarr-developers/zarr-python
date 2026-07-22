Added a fast path to `CoordinateIndexer` for sorted, in-bounds, one-dimensional integer coordinate
selections over regular chunk grids (e.g. `arr.get_coordinate_selection(sorted_idx)`,
`arr.vindex[sorted_idx]`, and the gather behind sparse/CSR row selections). These now build their
per-chunk projections with a single `searchsorted` over the touched chunk boundaries instead of
several passes over every selected element, making index construction ~15x faster for large
gathers. Unsorted, negative, multi-dimensional, and irregular-grid selections are unaffected and
continue to use the existing path.
