# Metadata compatibility

This page lists every kind of stored metadata that zarr-python reads or writes
although it does not conform to the
[Zarr specifications](https://zarr-specs.readthedocs.io/) (non-conformant), or
uses a name that the
[zarr-extensions registry](https://github.com/zarr-developers/zarr-extensions)
does not define (unregistered). Other Zarr implementations may refuse to read
such metadata, or read it differently.

For each kind of metadata, the list says which software wrote it, how the
current release of `zarr` reads it, and whether other implementations read it
(the end of the page says which versions were tested). Some are read by a repair: zarr reads the
metadata as the valid metadata it was meant to be, and stores that valid
metadata the next time it writes the array's metadata. Where a repair needs you
to act, opening the array warns and says what to do.

The list is generated from `zarr.core.metadata.ledger` when the documentation
is built. A test checks that ledger against the code that reads and writes
these kinds of metadata, so the list stays complete.

<!-- metadata-ledger -->
