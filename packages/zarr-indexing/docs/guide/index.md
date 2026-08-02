# Visual guide

Start with a familiar NumPy selection and follow it through the five ideas that
make indexing lazy and partitionable: coordinates, transforms, composition,
chunk planning, and paired projections. The same 6-by-8 image stays with us,
so each chapter adds one idea without changing the data.

1. [An index selects coordinates](01-indexing.md) begins with the observable
   shape, values, and order of `image[1:5, 2]`.
2. [Coordinates are addresses](02-transforms.md) explains literal coordinate
   domains, NumPy positions, and the request-to-source mapping between them.
3. [Lazy views compose](03-composition.md) shows how repeated indexing becomes
   one description and identifies exactly when data is read.
4. [A request becomes a chunk plan](04-chunks.md) divides that description over
   a 3-by-4 grid without choosing a source or scheduler.
5. [One cell domain, two projections](05-projections.md) pairs chunk-local reads
   with their exact positions in the requested result.

You can finish the tour with a practical mental model: indexing through
`LazyArray.lazy` builds a view, chunk planning partitions its coordinates, and
`result()` materializes that view.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../">zarr-indexing</a>
  ·
  <strong>Next:</strong> <a href="01-indexing/">An index selects coordinates</a>
</nav>
