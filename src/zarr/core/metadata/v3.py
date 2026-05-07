from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypeGuard, cast

from typing_extensions import TypedDict

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.abc.metadata import Metadata
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.chunk_grids import is_regular_nd
from zarr.core.chunk_key_encodings import (
    ChunkKeyEncoding,
    ChunkKeyEncodingLike,
    parse_chunk_key_encoding,
)
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    DimensionNamesLike,
    NamedConfig,
    NamedRequiredConfig,
    compress_rle,
    expand_rle,
    parse_named_configuration,
    parse_shapelike,
    validate_rectilinear_edges,
    validate_rectilinear_kind,
)
from zarr.core.config import config
from zarr.core.dtype import VariableLengthUTF8, ZDType, get_data_type_from_json
from zarr.core.dtype.common import check_dtype_spec_v3
from zarr.core.metadata.common import parse_attributes
from zarr.errors import MetadataValidationError, NodeTypeValidationError, UnknownCodecError
from zarr.registry import get_codec_class

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_grids import ChunksTuple
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar


def parse_zarr_format(data: object) -> Literal[3]:
    if data == 3:
        return 3
    msg = f"Invalid value for 'zarr_format'. Expected '3'. Got '{data}'."
    raise MetadataValidationError(msg)


def parse_node_type_array(data: object) -> Literal["array"]:
    if data == "array":
        return "array"
    msg = f"Invalid value for 'node_type'. Expected 'array'. Got '{data}'."
    raise NodeTypeValidationError(msg)


def parse_codecs(data: object) -> tuple[Codec, ...]:
    out: tuple[Codec, ...] = ()

    if not isinstance(data, Iterable):
        raise TypeError(f"Expected iterable, got {type(data)}")

    for c in data:
        if isinstance(
            c, ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
        ):  # Can't use Codec here because of mypy limitation
            out += (c,)
        else:
            name_parsed, _ = parse_named_configuration(c, require_configuration=False)

            try:
                out += (get_codec_class(name_parsed).from_dict(c),)
            except KeyError as e:
                raise UnknownCodecError(f"Unknown codec: {e.args[0]!r}") from e

    return out


def validate_array_bytes_codec(codecs: tuple[Codec, ...]) -> ArrayBytesCodec:
    # ensure that we have at least one ArrayBytesCodec
    abcs: list[ArrayBytesCodec] = [codec for codec in codecs if isinstance(codec, ArrayBytesCodec)]
    if len(abcs) == 0:
        raise ValueError("At least one ArrayBytesCodec is required.")
    elif len(abcs) > 1:
        raise ValueError("Only one ArrayBytesCodec is allowed.")

    return abcs[0]


def validate_codecs(codecs: tuple[Codec, ...], dtype: ZDType[TBaseDType, TBaseScalar]) -> None:
    """Check that the codecs are valid for the given dtype"""
    from zarr.codecs.sharding import ShardingCodec

    abc = validate_array_bytes_codec(codecs)

    # Recursively resolve array-bytes codecs within sharding codecs
    while isinstance(abc, ShardingCodec):
        abc = validate_array_bytes_codec(abc.codecs)

    # we need to have special codecs if we are decoding vlen strings or bytestrings
    # TODO: use codec ID instead of class name
    codec_class_name = abc.__class__.__name__
    # TODO: Fix typing here
    if isinstance(dtype, VariableLengthUTF8) and codec_class_name not in (  # type: ignore[unreachable]
        "VLenUTF8Codec",
        "ArrowIPCCodec",
    ):
        raise ValueError(
            f"For string dtype, ArrayBytesCodec must be `VLenUTF8Codec`, got `{codec_class_name}`."
        )


def parse_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if data is None:
        return data
    elif isinstance(data, Iterable) and all(isinstance(x, type(None) | str) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or an iterable of str, got {type(data)}"
        raise TypeError(msg)


def parse_storage_transformers(data: object) -> tuple[dict[str, JSON], ...]:
    """
    Parse storage_transformers. Zarr python cannot use storage transformers
    at this time, so this function doesn't attempt to validate them.
    """
    if data is None:
        return ()
    if isinstance(data, Iterable):
        if len(tuple(data)) >= 1:
            return data  # type: ignore[return-value]
        else:
            return ()
    raise TypeError(
        f"Invalid storage_transformers. Expected an iterable of dicts. Got {type(data)} instead."
    )


class AllowedExtraField(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    This class models allowed extra fields in array metadata.
    They must have ``must_understand`` set to ``False``, and may contain
    arbitrary additional JSON data.
    """

    must_understand: Literal[False]


def check_allowed_extra_field(data: object) -> TypeGuard[AllowedExtraField]:
    """
    Check if the extra field is allowed according to the Zarr v3 spec. The object
    must be a mapping with a "must_understand" key set to `False`.
    """
    return isinstance(data, Mapping) and data.get("must_understand") is False


def parse_extra_fields(
    data: Mapping[str, AllowedExtraField] | None,
) -> dict[str, AllowedExtraField]:
    if data is None:
        return {}
    else:
        conflict_keys = ARRAY_METADATA_KEYS & set(data.keys())
        if len(conflict_keys) > 0:
            msg = (
                "Invalid extra fields. "
                "The following keys: "
                f"{sorted(conflict_keys)} "
                "are invalid because they collide with keys reserved for use by the "
                "array metadata document."
            )
            raise ValueError(msg)
        return dict(data)


# JSON type for a single dimension's rectilinear spec:
# bare int (uniform shorthand), or list of ints / [value, count] RLE pairs.
RectilinearDimSpecJSON = int | list[int | list[int]]


class RegularChunkGridMetadataConfig(TypedDict):
    chunk_shape: Sequence[int]


class RectilinearChunkGridMetadataConfig(TypedDict):
    kind: Literal["inline"]
    chunk_shapes: Sequence[RectilinearDimSpecJSON]


RegularChunkGridMetadataJSON = NamedRequiredConfig[
    Literal["regular"], RegularChunkGridMetadataConfig
]
RectilinearChunkGridMetadataJSON = NamedRequiredConfig[
    Literal["rectilinear"], RectilinearChunkGridMetadataConfig
]


def _parse_chunk_shape(chunk_shape: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize a regular chunk shape.

    Delegates to ``_validate_chunk_shapes`` — a regular chunk shape is just
    a sequence of bare ints (one per dimension), each of which must be >= 1.
    """
    result = _validate_chunk_shapes(tuple(chunk_shape))
    # Regular grids only have bare ints — cast is safe after validation
    return cast(tuple[int, ...], result)


def _validate_chunk_shapes(
    chunk_shapes: Sequence[int | Sequence[int]],
) -> tuple[int | tuple[int, ...], ...]:
    """Validate per-dimension chunk specifications.

    Each element is either a bare ``int`` (regular step size, must be >= 1)
    or a sequence of explicit edge lengths (all must be >= 1, non-empty).
    """
    result: list[int | tuple[int, ...]] = []
    for dim_idx, dim_spec in enumerate(chunk_shapes):
        if isinstance(dim_spec, int):
            if dim_spec < 1:
                raise ValueError(
                    f"Dimension {dim_idx}: integer chunk edge length must be >= 1, got {dim_spec}"
                )
            result.append(dim_spec)
        else:
            edges = tuple(dim_spec)
            if not edges:
                raise ValueError(f"Dimension {dim_idx} has no chunk edges.")
            bad = [i for i, e in enumerate(edges) if e < 1]
            if bad:
                raise ValueError(
                    f"Dimension {dim_idx} has invalid edge lengths at indices {bad}: "
                    f"{[edges[i] for i in bad]}"
                )
            result.append(edges)
    return tuple(result)


@dataclass(frozen=True, kw_only=True)
class RegularChunkGridMetadata(Metadata):
    """Metadata-only description of a regular chunk grid.

    Stores just the chunk shape — no array extent, no runtime logic.
    This is what lives on ``ArrayV3Metadata.chunk_grid``.
    """

    chunk_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        chunk_shape_parsed = _parse_chunk_shape(self.chunk_shape)
        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @property
    def ndim(self) -> int:
        return len(self.chunk_shape)

    def to_dict(self) -> RegularChunkGridMetadataJSON:  # type: ignore[override]
        return {
            "name": "regular",
            "configuration": {"chunk_shape": self.chunk_shape},
        }

    @classmethod
    def from_dict(cls, data: RegularChunkGridMetadataJSON) -> Self:  # type: ignore[override]
        parse_named_configuration(data, "regular")  # validate name
        configuration = data["configuration"]
        return cls(chunk_shape=_parse_chunk_shape(configuration["chunk_shape"]))


@dataclass(frozen=True, kw_only=True)
class RectilinearChunkGridMetadata(Metadata):
    """Metadata-only description of a rectilinear chunk grid.

    Each element of ``chunk_shapes`` is either:

    - A bare ``int`` — a regular step size that repeats to cover the axis
      (the spec's single-integer shorthand).
    - A ``tuple[int, ...]`` — explicit per-chunk edge lengths (already
      expanded from any RLE encoding).

    This distinction matters for faithful round-tripping: a bare int
    serializes back as a bare int, while a single-element tuple serializes
    as a list.
    """

    chunk_shapes: tuple[int | tuple[int, ...], ...]

    def __post_init__(self) -> None:
        from zarr.core.config import config

        if not config.get("array.rectilinear_chunks"):
            raise ValueError(
                "Rectilinear chunk grids are experimental and disabled by default. "
                "Enable them with: zarr.config.set({'array.rectilinear_chunks': True}) "
                "or set the environment variable ZARR_ARRAY__RECTILINEAR_CHUNKS=True"
            )
        object.__setattr__(self, "chunk_shapes", _validate_chunk_shapes(self.chunk_shapes))

    @property
    def ndim(self) -> int:
        return len(self.chunk_shapes)

    def to_dict(self) -> RectilinearChunkGridMetadataJSON:  # type: ignore[override]
        serialized_dims: list[RectilinearDimSpecJSON] = []
        for dim_spec in self.chunk_shapes:
            if isinstance(dim_spec, int):
                # Bare int shorthand — serialize as-is
                serialized_dims.append(dim_spec)
            else:
                rle = compress_rle(dim_spec)
                # Use RLE only if it's actually shorter
                if len(rle) < len(dim_spec):
                    serialized_dims.append(rle)
                else:
                    serialized_dims.append(list(dim_spec))
        return {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": tuple(serialized_dims),
            },
        }

    def update_shape(
        self, old_shape: tuple[int, ...], new_shape: tuple[int, ...]
    ) -> RectilinearChunkGridMetadata:
        """Return a new RectilinearChunkGridMetadata with edges adjusted for *new_shape*.

        - Bare-int dimensions stay as bare ints (they cover any extent).
        - Explicit-edge dimensions: if the new extent exceeds the sum of
          edges, a new chunk is appended to cover the additional extent.
          Otherwise edges are kept as-is (the spec allows trailing edges
          beyond the array extent).
        """
        new_chunk_shapes: list[int | tuple[int, ...]] = []
        for dim_spec, new_ext in zip(self.chunk_shapes, new_shape, strict=True):
            if isinstance(dim_spec, int):
                # Bare int covers any extent — no change needed
                new_chunk_shapes.append(dim_spec)
            else:
                edge_sum = sum(dim_spec)
                if new_ext > edge_sum:
                    new_chunk_shapes.append((*dim_spec, new_ext - edge_sum))
                else:
                    new_chunk_shapes.append(dim_spec)
        return RectilinearChunkGridMetadata(chunk_shapes=tuple(new_chunk_shapes))

    @classmethod
    def from_dict(cls, data: RectilinearChunkGridMetadataJSON) -> Self:  # type: ignore[override]
        parse_named_configuration(data, "rectilinear")  # validate name
        configuration = data["configuration"]
        validate_rectilinear_kind(configuration.get("kind"))
        raw_shapes = configuration["chunk_shapes"]
        parsed: list[int | tuple[int, ...]] = []
        for dim_spec in raw_shapes:
            if isinstance(dim_spec, int):
                if dim_spec < 1:
                    raise ValueError(f"Integer chunk edge length must be >= 1, got {dim_spec}")
                parsed.append(dim_spec)
            elif isinstance(dim_spec, list):
                parsed.append(tuple(expand_rle(dim_spec)))
            else:
                raise TypeError(
                    f"Invalid chunk_shapes entry: expected int or list, got {type(dim_spec)}"
                )
        return cls(chunk_shapes=tuple(parsed))


ChunkGridMetadata = RegularChunkGridMetadata | RectilinearChunkGridMetadata


def create_chunk_grid_metadata(
    chunks: ChunksTuple,
) -> ChunkGridMetadata:
    """Construct a chunk grid metadata object from a normalized `ChunksTuple`.

    Regular chunks produce a `RegularChunkGridMetadata`.
    Rectilinear chunks produce a `RectilinearChunkGridMetadata`.

    Parameters
    ----------
    chunks : ChunksTuple
        Normalized chunk specification, as returned by
        `normalize_chunks_nd` or `guess_chunks`.

    See Also
    --------
    parse_chunk_grid : Deserialize a chunk grid from stored JSON metadata.
    """
    if is_regular_nd(chunks):
        # If we know the chunks specification is regular, then we can take the first
        # chunk size for each dimension as the chunk shape.
        chunk_shape = tuple(int(dim_chunks[0]) for dim_chunks in chunks)
        return RegularChunkGridMetadata(chunk_shape=chunk_shape)
    else:
        return RectilinearChunkGridMetadata(
            chunk_shapes=tuple(tuple(int(x) for x in d) for d in chunks)
        )


def parse_chunk_grid(
    data: dict[str, JSON] | ChunkGridMetadata | NamedConfig[str, Any],
) -> ChunkGridMetadata:
    """Deserialize a chunk grid from stored JSON metadata or pass through an existing instance.

    See Also
    --------
    create_chunk_grid_metadata : Construct a chunk grid from user-facing input.
    """
    if isinstance(data, (RegularChunkGridMetadata, RectilinearChunkGridMetadata)):
        return data

    name, _ = parse_named_configuration(data)
    if name == "regular":
        return RegularChunkGridMetadata.from_dict(data)  # type: ignore[arg-type]
    if name == "rectilinear":
        return RectilinearChunkGridMetadata.from_dict(data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown chunk grid name: {name!r}")


class ArrayMetadataJSON_V3(TypedDict, extra_items=AllowedExtraField):  # type: ignore[call-arg]
    """
    A typed dictionary model for zarr v3 array metadata.

    Extra keys are permitted if they conform to ``AllowedExtraField``
    (i.e. they are mappings with ``must_understand: false``).
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: str | NamedConfig[str, Mapping[str, JSON]]
    shape: tuple[int, ...]
    chunk_grid: str | NamedConfig[str, Mapping[str, JSON]]
    chunk_key_encoding: str | NamedConfig[str, Mapping[str, JSON]]
    fill_value: JSON
    codecs: tuple[str | NamedConfig[str, Mapping[str, JSON]], ...]
    attributes: NotRequired[Mapping[str, JSON]]
    storage_transformers: NotRequired[tuple[str | NamedConfig[str, Mapping[str, JSON]], ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


"""
The names of the fields of the array metadata document defined in the zarr V3 spec.
"""
ARRAY_METADATA_KEYS: Final[set[str]] = {
    "zarr_format",
    "node_type",
    "data_type",
    "shape",
    "chunk_grid",
    "chunk_key_encoding",
    "fill_value",
    "codecs",
    "attributes",
    "storage_transformers",
    "dimension_names",
}


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(Metadata):
    shape: tuple[int, ...]
    data_type: ZDType[TBaseDType, TBaseScalar]
    chunk_grid: ChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str | None, ...] | None = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    storage_transformers: tuple[dict[str, JSON], ...]
    extra_fields: dict[str, AllowedExtraField]

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: dict[str, JSON] | ChunkGridMetadata | NamedConfig[str, Any],
        chunk_key_encoding: ChunkKeyEncodingLike,
        fill_value: object,
        codecs: Iterable[Codec | dict[str, JSON] | NamedConfig[str, Any] | str],
        attributes: dict[str, JSON] | None,
        dimension_names: DimensionNamesLike,
        storage_transformers: Iterable[dict[str, JSON]] | None = None,
        extra_fields: Mapping[str, AllowedExtraField] | None = None,
    ) -> None:
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """

        shape_parsed = parse_shapelike(shape)
        chunk_grid_parsed = parse_chunk_grid(chunk_grid)
        chunk_key_encoding_parsed = parse_chunk_key_encoding(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        # Note: relying on a type method is numpy-specific
        fill_value_parsed = data_type.cast_scalar(fill_value)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)
        storage_transformers_parsed = parse_storage_transformers(storage_transformers)
        extra_fields_parsed = parse_extra_fields(extra_fields)
        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type,
            fill_value=fill_value_parsed,
            config=ArrayConfig.from_dict({}),  # TODO: config is not needed here.
            prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
        )
        # Thread the spec through evolution: each codec must be evolved against
        # the spec it will actually see at run-time, not the original array spec.
        # Earlier array->array codecs may transform the dtype (e.g. cast_value),
        # so the spec passed to later codecs must reflect those transformations.
        # Per-codec validate() must run before resolve_metadata(), since the
        # latter may rely on invariants the former checks (e.g. cast_value
        # rejects complex source dtypes that would otherwise crash _do_cast).
        evolved: list[Codec] = []
        spec = array_spec
        for c in codecs_parsed_partial:
            evolved_codec = c.evolve_from_array_spec(spec)
            evolved_codec.validate(shape=spec.shape, dtype=spec.dtype, chunk_grid=chunk_grid_parsed)
            evolved.append(evolved_codec)
            spec = evolved_codec.resolve_metadata(spec)
        codecs_parsed = tuple(evolved)
        validate_codecs(codecs_parsed_partial, data_type)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)
        object.__setattr__(self, "storage_transformers", storage_transformers_parsed)
        object.__setattr__(self, "extra_fields", extra_fields_parsed)

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if len(self.shape) != self.chunk_grid.ndim:
            raise ValueError("`chunk_grid` and `shape` need to have the same number of dimensions.")
        if isinstance(self.chunk_grid, RectilinearChunkGridMetadata):
            validate_rectilinear_edges(self.chunk_grid.chunk_shapes, self.shape)
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            raise ValueError(
                "`dimension_names` and `shape` need to have the same number of dimensions."
            )
        if self.fill_value is None:
            raise ValueError("`fill_value` is required.")
        for codec in self.codecs:
            codec.validate(shape=self.shape, dtype=self.data_type, chunk_grid=self.chunk_grid)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> ZDType[TBaseDType, TBaseScalar]:
        return self.data_type

    # TODO: move these properties to the Array class.
    # They require knowledge of codecs (ShardingCodec) and don't belong on a metadata DTO.

    @property
    def chunks(self) -> tuple[int, ...]:
        if not isinstance(self.chunk_grid, RegularChunkGridMetadata):
            msg = (
                "The `chunks` attribute is only defined for arrays using regular chunk grids. "
                "This array has a rectilinear chunk grid. Use `read_chunk_sizes` for general access."
            )
            raise NotImplementedError(msg)

        from zarr.codecs.sharding import ShardingCodec

        if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
            return self.codecs[0].chunk_shape
        return self.chunk_grid.chunk_shape

    @property
    def shards(self) -> tuple[int, ...] | None:
        from zarr.codecs.sharding import ShardingCodec

        if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
            if not isinstance(self.chunk_grid, RegularChunkGridMetadata):
                msg = (
                    "The `shards` attribute is only defined for arrays using regular chunk grids. "
                    "This array has a rectilinear chunk grid. Use `write_chunk_sizes` for general access."
                )
                raise NotImplementedError(msg)
            return self.chunk_grid.chunk_shape
        return None

    @property
    def inner_codecs(self) -> tuple[Codec, ...]:
        from zarr.codecs.sharding import ShardingCodec

        if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
            return self.codecs[0].codecs
        return self.codecs

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        return self.chunk_key_encoding.encode_chunk_key(chunk_coords)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        json_indent = config.get("json_indent")
        d = self.to_dict()
        return {
            ZARR_JSON: prototype.buffer.from_bytes(
                json.dumps(d, allow_nan=True, indent=json_indent).encode()
            )
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        # make a copy because we are modifying the dict
        _data = data.copy()

        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(_data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(_data.pop("node_type"))

        data_type_json = _data.pop("data_type")
        if not check_dtype_spec_v3(data_type_json):
            raise ValueError(f"Invalid data_type: {data_type_json!r}")
        data_type = get_data_type_from_json(data_type_json, zarr_format=3)

        # check that the fill value is consistent with the data type
        try:
            fill = _data.pop("fill_value")
            fill_value_parsed = data_type.from_json_scalar(fill, zarr_format=3)
        except ValueError as e:
            raise TypeError(f"Invalid fill_value: {fill!r}") from e

        # check if there are extra keys
        extra_keys = set(_data.keys()) - ARRAY_METADATA_KEYS
        allowed_extra_fields: dict[str, AllowedExtraField] = {}
        invalid_extra_fields = {}
        for key in extra_keys:
            val = _data[key]
            if check_allowed_extra_field(val):
                allowed_extra_fields[key] = val
            else:
                invalid_extra_fields[key] = val
        if len(invalid_extra_fields) > 0:
            msg = (
                "Got a Zarr V3 metadata document with the following disallowed extra fields:"
                f"{sorted(invalid_extra_fields.keys())}."
                'Extra fields are not allowed unless they are a dict with a "must_understand" key'
                "which is assigned the value `False`."
            )
            raise MetadataValidationError(msg)
        # TODO: replace this with a real type check!
        _data_typed = cast(ArrayMetadataJSON_V3, _data)

        return cls(
            shape=_data_typed["shape"],
            chunk_grid=_data_typed["chunk_grid"],  # type: ignore[arg-type]
            chunk_key_encoding=_data_typed["chunk_key_encoding"],  # type: ignore[arg-type]
            codecs=_data_typed["codecs"],
            attributes=_data_typed.get("attributes", {}),  # type: ignore[arg-type]
            dimension_names=_data_typed.get("dimension_names", None),
            fill_value=fill_value_parsed,
            data_type=data_type,
            extra_fields=allowed_extra_fields,
            storage_transformers=_data_typed.get("storage_transformers", ()),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, JSON]:
        out_dict = super().to_dict()
        extra_fields = out_dict.pop("extra_fields")
        out_dict = out_dict | extra_fields  # type: ignore[operator]

        out_dict["chunk_grid"] = self.chunk_grid.to_dict()

        out_dict["fill_value"] = self.data_type.to_json_scalar(
            self.fill_value, zarr_format=self.zarr_format
        )
        if not isinstance(out_dict, dict):
            raise TypeError(f"Expected dict. Got {type(out_dict)}.")

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")

        # TODO: replace the `to_dict` / `from_dict` on the `Metadata`` class with
        # to_json, from_json, and have ZDType inherit from `Metadata`
        # until then, we have this hack here, which relies on the fact that to_dict will pass through
        # any non-`Metadata` fields as-is.
        dtype_meta = out_dict["data_type"]
        if isinstance(dtype_meta, ZDType):
            out_dict["data_type"] = dtype_meta.to_json(zarr_format=3)  # type: ignore[unreachable]
        return out_dict

    def update_shape(self, shape: tuple[int, ...]) -> Self:
        chunk_grid = self.chunk_grid
        if isinstance(chunk_grid, RectilinearChunkGridMetadata):
            chunk_grid = chunk_grid.update_shape(self.shape, shape)
        return replace(self, shape=shape, chunk_grid=chunk_grid)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)
