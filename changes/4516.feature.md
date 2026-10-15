Added the ``offset`` chunk key encoding (the ``default`` keys of ``chunk_coords + offset``) and
``Array.prepend``, which adds data before index 0 of an axis by whole chunks without changing any
existing chunk's key.
