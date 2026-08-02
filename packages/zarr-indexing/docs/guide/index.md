# Visual guide

Start with a familiar NumPy selection and follow it through the three ideas that
make indexing lazy: coordinates, transforms, and composition. The same 6-by-8
image stays with us, so each chapter adds one idea without changing the data.

1. [An index selects coordinates](01-indexing.md) begins with the observable
   shape, values, and order of `image[1:5, 2]`.
2. [Coordinates are addresses](02-transforms.md) explains literal coordinate
   domains, NumPy positions, and the request-to-source mapping between them.
3. [Lazy views compose](03-composition.md) shows how repeated indexing becomes
   one description and identifies exactly when data is read.

You can finish the tour with a practical mental model: indexing through
`LazyArray.lazy` builds a view, while `result()` materializes that view.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../">zarr-indexing</a>
  ·
  <strong>Next:</strong> <a href="01-indexing/">An index selects coordinates</a>
</nav>
