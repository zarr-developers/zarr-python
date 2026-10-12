"""The type of a key decides whether it is read relative to the view or as absolute coordinates."""

import numpy as np

from zarr_indexing import IndexDomain, IndexTransform, LazyArray

source = np.arange(30)

# --8<-- [start:key-types]
view = LazyArray(source)[10:20]
assert view.transform.domain == IndexDomain((10,), (20,))

# NumPy keys are positions relative to the view: 0 is its first element.
relative = view[2:5]
assert relative.result().tolist() == [12, 13, 14]
assert view[-1].result() == 19

# An IndexDomain key names absolute coordinates of the view's domain.
absolute = view[IndexDomain((12,), (15,))]
assert absolute.result().tolist() == [12, 13, 14]

# An IndexTransform key composes onto the view; index the identity over the
# view's domain to spell any selection in absolute coordinates.
literal = IndexTransform.identity(view.transform.domain)
assert view[literal[12:15]].result().tolist() == [12, 13, 14]
assert view[literal[::-1]].result().tolist()[:2] == [19, 18]

# Whichever key produced a view, its domain is absolute.
assert relative.transform.domain == absolute.transform.domain == IndexDomain((12,), (15,))
# --8<-- [end:key-types]
