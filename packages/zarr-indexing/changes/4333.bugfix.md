Direct `IndexTransform.oindex` and `IndexTransform.vindex` selections now reject
integer array coordinates outside the `np.intp` range before conversion.
Previously, oversized `uint64` values could wrap to negative coordinates and
silently select a different location in a domain containing negative coordinates.
