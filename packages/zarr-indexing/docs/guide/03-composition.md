# Lazy views compose

A lazy view can be indexed again. Each step changes the request-to-source
description, but it does not read an intermediate array. The chain is reduced
to one direct transform from the newest request to the original source.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable composed views diagram" tabindex="0">
--8<-- "_static/diagrams/compose-views.svg"
</div>
<figcaption>The two view steps collapse into one direct request-to-source map, so composition adds no intermediate read.</figcaption>
</figure>

The executable example first selects the familiar row, then reverses that
view and trims its first element:

```python
--8<-- "examples/lazy_composition.py:lazy-composition"
```

Immediately after `composed` is created—and before the final `result()` call—its
metadata is ready to inspect:

| Available without reading | Value in this example |
| --- | --- |
| `composed.shape` | `(3,)` |
| `composed.transform` | One transform mapping request `i` to source `(1, 2 - i)` |

Neither property needs source values. Composition works only on the coordinate
description; the assertion's call to `result()` is the first operation in the
example that materializes the selected data.

!!! warning "Stop here: the materialization boundary"
    Indexing through `.lazy[...]` is lazy. Calling `result()`, indexing the
    wrapper eagerly with `view[...]`, converting it with `numpy.asarray`, or
    passing it to a NumPy function materializes data. Python arithmetic such as
    `view + 1` is outside this wrapper's deferred scope and raises `TypeError`;
    NumPy arithmetic such as `numpy.add(view, 1)` converts and materializes the
    view. This wrapper defers indexing, not a general compute graph.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../02-transforms/">Coordinates are addresses</a>
  ·
  <strong>Next:</strong> <a href="../04-chunks/">A request becomes a chunk plan</a>
</nav>
