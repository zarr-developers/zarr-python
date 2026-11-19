"""The ledger of stored metadata that does not conform to the Zarr specifications.

Each entry is one kind of stored metadata that is not conformant (the specification
forbids it) or not registered (it uses a name the zarr-extensions registry does not
define), which zarr-python writes or reads. An entry records which software wrote it,
whether zarr still writes it, and how zarr reads it now. The Metadata compatibility page
of the documentation lists this ledger.

Every place in `src/zarr` that reads or writes such metadata carries a comment naming
its entry, `# NON-CONFORMANT: <key>` or `# UNREGISTERED: <key>`, so a reader of the code
finds the entry and the entry can be checked against the code: `tests/test_ledger.py`
fails if a marker names no entry, if an entry has no marker, or if a repair in
`zarr.core.metadata.repair.ARRAY_REPAIRS` is in no entry.

The ledger records zarr releases; the warnings zarr gives describe the metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from zarr.core.metadata import repair

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.metadata.repair import Repair

type Conformance = Literal["non-conformant", "unregistered"]


@dataclass(frozen=True, kw_only=True)
class Entry:
    """One kind of non-conformant or unregistered stored metadata. Text is Markdown."""

    title: str
    """What the stored metadata holds."""
    conformance: Conformance
    """`non-conformant` if a specification forbids it, `unregistered` if it uses a name the
    zarr-extensions registry does not define."""
    written_by: str
    """Which software wrote it, and which releases."""
    still_written: bool
    """Whether the current zarr writes it."""
    reading: str
    """How the current zarr reads it, and whether it warns."""
    other_readers: str
    """Whether the implementations in `READERS_TESTED` read it."""
    repairs: tuple[Repair, ...] = ()
    """The repairs in `zarr.core.metadata.repair` that read it."""


MARKERS: Final[Mapping[Conformance, str]] = {
    "non-conformant": "NON-CONFORMANT",
    "unregistered": "UNREGISTERED",
}
"""The comment tag that marks the code of an entry of each conformance."""

READERS_TESTED: Final = (
    "zarrs 0.23.14 (Rust crate, `Array::open` then reading every element, else `Group::open`), "
    "tensorstore 0.1.85 (`zarr` and `zarr3` drivers, which open arrays only), and zarr 2.18.7 "
    "(Zarr format 2 only), on 2026-10-10, each against a sample store of every entry"
)
"""Which other implementations `Entry.other_readers` was tested with, and how."""


LEDGER: Final[Mapping[str, Entry]] = {
    "chunk-size-zero": Entry(
        title=(
            "A regular chunk size of `0` or `false` (Zarr format 2 `chunks`, Zarr format 3 "
            "`regular` `chunk_shape`)"
        ),
        conformance="non-conformant",
        written_by=(
            "zarr 3.0 to 3.3 for a Zarr format 2 array created with a zero-length axis, and "
            "zarr 3.0 to 3.1 for a Zarr format 3 one; zarr 2.18.7 to 3.2.1 for an explicit chunk "
            "size of `0` or `False` on an axis of any length"
        ),
        still_written=False,
        reading=(
            "Repaired: read as chunk size 1 (the inner chunk size of a sharded array). "
            "Silent on a zero-length axis; on an axis of positive length it warns that the "
            "array holds only its fill value. Writing chunks first stores the repaired "
            "metadata."
        ),
        other_readers=(
            "zarrs and tensorstore reject it. zarr 2.18.7 fails to open it on a zero-length axis."
        ),
        repairs=(repair._invalid_chunk_sizes_v2, repair._invalid_chunk_sizes_v3),
    ),
    "chunk-size-true": Entry(
        title=(
            "A chunk size of JSON `true`, in a regular chunk shape, in the inner chunk shape "
            "of a sharding codec, or as a rectilinear chunk edge length"
        ),
        conformance="non-conformant",
        written_by="zarr 3.0 and 3.2 for a chunk size of `True`",
        still_written=False,
        reading="Repaired silently: read as 1.",
        other_readers=(
            "zarrs and tensorstore reject it, in every place it was tested (regular chunk shape, "
            "sharding inner chunk shape, rectilinear edges; tensorstore reads no rectilinear "
            "grid). zarr 2.18.7 reads the Zarr format 2 form."
        ),
        repairs=(
            repair._invalid_chunk_sizes_v2,
            repair._invalid_chunk_sizes_v3,
            repair._invalid_inner_chunk_sizes_v3,
            repair._invalid_edge_lengths_v3,
        ),
    ),
    "regular-grid-edge-lists": Entry(
        title=(
            "A `regular` chunk grid whose `chunk_shape` mixes chunk sizes with lists of chunk "
            "edge lengths, such as `[2, [5, 10, 5]]`"
        ),
        conformance="non-conformant",
        written_by="zarr 3.2.0 and 3.2.1 for `chunks=(2, (5, 10, 5))`",
        still_written=False,
        reading=(
            "Repaired with a warning: read as the rectilinear chunk grid it describes, "
            "without the `array.rectilinear_chunks` flag. Re-saving the metadata, or "
            "writing chunks, stores that rectilinear grid, which requires the flag."
        ),
        other_readers="zarrs and tensorstore reject it.",
        repairs=(repair._invalid_chunk_sizes_v3,),
    ),
    "rectilinear-float-edges": Entry(
        title="A rectilinear chunk edge length that is an integral JSON float, such as `4.0`",
        conformance="non-conformant",
        written_by="zarr 3.2 for float chunk edges",
        still_written=False,
        reading=(
            "Repaired silently: read as the integer it equals. A float anywhere else in a "
            "chunk grid is rejected."
        ),
        other_readers=(
            "zarrs rejects it (it reads a rectilinear grid with integer edges). tensorstore reads "
            "no rectilinear grid."
        ),
        repairs=(repair._invalid_edge_lengths_v3,),
    ),
    "v2-empty-filters": Entry(
        title="Zarr format 2 `filters: []`",
        conformance="non-conformant",
        written_by="zarr 3.0.0 to 3.0.3 for `filters=[]`",
        still_written=False,
        reading="Read as `filters: null`, with a warning.",
        other_readers="zarrs, tensorstore and zarr 2.18.7 read it.",
    ),
    "group-consolidated-metadata-null": Entry(
        title='A Zarr format 3 group with `"consolidated_metadata": null`',
        conformance="non-conformant",
        written_by="zarr 3.0.0 to 3.1.3, for every Zarr format 3 group",
        still_written=False,
        reading="Read silently as a group with no consolidated metadata.",
        other_readers="zarrs opens the group.",
    ),
    "struct-bytes-codec-without-endian": Entry(
        title=(
            'A `bytes` codec with no `endian` (`{"name": "bytes"}`) on an array whose '
            "structured data type has multi-byte fields"
        ),
        conformance="non-conformant",
        written_by="zarr 3.1.x",
        still_written=False,
        reading="Read as little-endian, with a warning.",
        other_readers=(
            "tensorstore reads it, field by field, as little-endian. zarrs reads no `structured` "
            "data type."
        ),
    ),
    "structured-data-type": Entry(
        title=(
            'The `structured` data type: `"name": "structured"`, fields as `[name, data_type]` '
            "pairs, and a base64 fill value"
        ),
        conformance="non-conformant",
        written_by=(
            "zarr 3.1.x for every structured data type; the current zarr writes it only for "
            "a `zarr.dtype.Structured` constructed explicitly"
        ),
        still_written=True,
        reading=(
            "Read silently as the `struct` data type, which is how the zarr-extensions "
            "registry defines it: a legacy alias that implementations may read but must not "
            "write. Written with an `UnstableSpecificationWarning`."
        ),
        other_readers="tensorstore reads it, field by field. zarrs rejects the `structured` name.",
    ),
    "variable-length-bytes": Entry(
        title='The `"variable_length_bytes"` data type',
        conformance="unregistered",
        written_by=(
            "zarr 3.1.0 and later, for variable-length bytes, where zarr 3.0 wrote the "
            "registered name `bytes`"
        ),
        still_written=True,
        reading=(
            "Read, and so is the registered name `bytes` with a base64 fill value; the "
            "fill value `[]` that zarr 3.0 wrote is rejected. Written with an "
            "`UnstableSpecificationWarning`; re-saving metadata that names `bytes` stores "
            "`variable_length_bytes`."
        ),
        other_readers="zarrs reads it. tensorstore rejects the name.",
    ),
    "null-terminated-bytes": Entry(
        title='The `"null_terminated_bytes"` data type, for NumPy `S` data types',
        conformance="unregistered",
        written_by="zarr 3.1.0 and later",
        still_written=True,
        reading="Read. Written with an `UnstableSpecificationWarning`.",
        other_readers="zarrs and tensorstore reject the name.",
    ),
    "raw-bytes": Entry(
        title='The `"raw_bytes"` data type, for NumPy `V` data types',
        conformance="unregistered",
        written_by="zarr 3.1.0 and later",
        still_written=True,
        reading="Read. Written with an `UnstableSpecificationWarning`.",
        other_readers=(
            "tensorstore opens it as an array of bytes with an extra trailing dimension. zarrs "
            "rejects the name."
        ),
    ),
    "numcodecs-codecs": Entry(
        title=(
            "Codecs named `numcodecs.<id>` (`numcodecs.delta`, `numcodecs.bz2`, ...), with "
            "the numcodecs configuration of the codec"
        ),
        conformance="unregistered",
        written_by=(
            "zarr 3.1.3 and later, through `zarr.codecs.numcodecs`, which the Zarr format 3 "
            "conversion of `zarr migrate` also uses; before that, numcodecs' `numcodecs.zarr3` "
            "module wrote the same names"
        ),
        still_written=True,
        reading=(
            "Read when numcodecs has a codec of that id; the `configuration` member is "
            "required, and is passed to numcodecs as it is. Written without a warning."
        ),
        other_readers=(
            "zarrs reads some of these codecs (`numcodecs.zlib`, `numcodecs.bz2`, "
            "`numcodecs.fixedscaleoffset`) and rejects others (`numcodecs.delta`, `numcodecs.lz4`, "
            "`numcodecs.crc32`). tensorstore rejects every one tested."
        ),
    ),
    "v3-consolidated-metadata": Entry(
        title=(
            "Consolidated metadata in a Zarr format 3 group, as a `consolidated_metadata` "
            'member with `"must_understand": false`'
        ),
        conformance="unregistered",
        written_by="zarr 3.0.0 and later, by `zarr.consolidate_metadata`",
        still_written=True,
        reading=(
            "Read. Readers that do not know it ignore it, because of `must_understand`. "
            "`zarr.consolidate_metadata` warns when it writes it."
        ),
        other_readers="zarrs opens the group.",
    ),
    "v2-consolidated-group-entries": Entry(
        title=(
            "Zarr format 2 consolidated metadata (`.zmetadata`) whose group entries carry a "
            "`consolidated_metadata` member"
        ),
        conformance="non-conformant",
        written_by="zarr 3.0.0 and later",
        still_written=True,
        reading=(
            "Read: the member marks a group with no members, as the Zarr format 3 "
            "consolidated metadata does."
        ),
        other_readers="zarrs opens the group; zarr 2.18.7 opens the consolidated group.",
    ),
    "json-nan-tokens": Entry(
        title=(
            "The tokens `NaN`, `Infinity` and `-Infinity` in a metadata document, which are not "
            "JSON, for a non-finite float in user attributes"
        ),
        conformance="non-conformant",
        written_by=(
            "zarr 2, zarr 3.0.0 to 3.0.6, and zarr 3.1.1 and later (zarr 3.0.7 to 3.0.10 wrote the "
            'strings `"NaN"`, `"Infinity"` and `"-Infinity"`; zarr 3.1.0 refused to write them)'
        ),
        still_written=True,
        reading="Read silently in any metadata document. Written without a warning.",
        other_readers=(
            "zarrs rejects the document (`zarr.json`, or `.zattrs` with Zarr format 2), and "
            "tensorstore rejects such a `zarr.json`. tensorstore reads a Zarr format 2 array, "
            "whose `.zattrs` it does not read; zarr 2.18.7 reads it."
        ),
    ),
    "numeric-fill-value-spellings": Entry(
        title=(
            "An integer fill value written as an integral float (`0.0`) or as a string "
            '(`"42"`), or a float fill value written as a decimal string (`"3.14"`)'
        ),
        conformance="non-conformant",
        written_by="Other software; found in public data. No zarr release wrote it.",
        still_written=False,
        reading="Read silently as the number it spells; re-saving stores that number.",
        other_readers=(
            "zarrs rejects every form. tensorstore reads an integer fill value of `0.0` and "
            "rejects the string forms. zarr 2.18.7 reads the Zarr format 2 string form."
        ),
    ),
    "string-fill-value-number": Entry(
        title="A number as the fill value of a variable-length string array",
        conformance="non-conformant",
        written_by="zarr 2, which gave such arrays a fill value of `0` by default",
        still_written=False,
        reading=(
            'Read silently as the string the number prints as (`0` as `"0"`), in both Zarr '
            "formats; re-saving stores that string."
        ),
        other_readers=(
            "zarrs and zarr 2.18.7 read the Zarr format 2 form; zarrs rejects the Zarr format 3 "
            "form. tensorstore reads no variable-length string array."
        ),
    ),
}
"""Every kind of non-conformant or unregistered stored metadata zarr reads or writes."""
