**Breaking:** a v3 array's `chunk_grid` is judged against its `shape`, by
the grid's definition. A regular grid whose `chunk_shape` does not have
one length per dimension of the shape, or has a length of 0 for a
dimension that is not empty, and a rectilinear grid whose `chunk_shapes`
does not have one entry per dimension, or whose chunk lengths fall short
of their dimension, each have a problem in `chunk_grid.configuration`,
where the package accepted them before.
