Added the ``offset`` chunk key encoding (the ``default`` keys of ``chunk_coords + offset``) and
``Array.prepend_chunks`` / ``Array.prepend``, which grow an array before index 0 by whole chunks
as a metadata-only change: every existing chunk keeps its key.
