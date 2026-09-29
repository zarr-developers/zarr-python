**Breaking:** a v3 array's `codecs` are read as a pipeline: in order --
array -> array codecs, then one array -> bytes codec, then bytes -> bytes
codecs -- and each codec judged against the chunk it is handed, which is
the grid's chunks of the array's data type, as the array -> array codecs
before it hand them on. A codec out of order, a second array -> bytes
codec or none at all; a `bytes` codec without an `endian` handed numbers
of several bytes, or handed values that vary in size; a `transpose` whose
`order` has another number of axes than its chunk; a `cast_value` to or
from a data type that models no real numbers, wrapping to one that is
not integral, or mapping a scalar that is not a fill value of the data
type on its side; a `scale_offset` handed values that are no numbers, or
with an `offset` or `scale` that is not a value of its data type; and a
struct field whose values vary in size are each a problem where they
sit, where the package accepted them before. `read_pipeline(codecs,
chunk)` gives each codec with the chunk it is handed,
`chunk_grid_lengths(grid, shape)` the lengths a grid's chunks take, and
`storage_of(data_type)` how a data type's values are stored. A
definition's `rules` are handed the fields its configuration holds as
the scope read them, and the kinds gain `chunk_lengths` (grids),
`storage` (data types), and `chunk_rules` and `transition` (codecs).
