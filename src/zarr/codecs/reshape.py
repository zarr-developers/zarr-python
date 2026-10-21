from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self

    from zarr.core.buffer import NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata


# Type for shape specification: int, list of ints, or -1
type ShapeElement = int | tuple[int, ...]


def prod(values: Iterable[int]) -> int:
    p = 1
    for v in values:
        p *= v
    return p


def _compute_output_shape(
    shape_spec: Sequence[ShapeElement],
    input_shape: Sequence[int],
) -> tuple[int, ...]:
    """
    Compute the output shape from the shape specification and input shape.

    Parameters
    ----------
    shape_spec : tuple[ShapeElement, ...]
        The shape specification from the codec configuration.
    input_shape : tuple[int, ...]
        The shape of the input array.

    Returns
    -------
    tuple[int, ...]
        The computed output shape.

    Raises
    ------
    ValueError
        If the shape specification is invalid or the invariant
        prod(output_shape) == prod(input_shape) cannot be satisfied.
    """
    _validate_shape_spec(shape_spec, len(input_shape))
    output_shape: list[int] = []
    infer_idx: int | None = None
    input_total = int(prod(input_shape))

    # we can skip a lot of validity checks here because we have already used _validate_shape_spec
    for i, elem in enumerate(shape_spec):
        if isinstance(elem, int):
            if elem == -1:
                if infer_idx is not None:
                    raise ValueError(
                        "The special value -1 may occur at most once in the shape specification."
                    )
                infer_idx = i
                output_shape.append(-1)  # placeholder
            else:
                output_shape.append(elem)
        else:
            # Product of input dimensions specified
            output_shape.append(prod(input_shape[d] for d in elem))

    # Handle inference of -1 dimension
    if infer_idx is not None:
        known_product = prod(s for s in output_shape if s != -1)
        if known_product == 0 or input_total % known_product != 0:
            raise ValueError(
                f"Cannot infer dimension size: input total {input_total} is not divisible "
                f"by known product {known_product}."
            )
        output_shape[infer_idx] = input_total // known_product

    output_total = prod(output_shape)
    if output_total != input_total:
        raise ValueError(
            f"The invariant prod(output_shape) == prod(input_shape) is not satisfied. "
            f"Got prod({output_shape}) = {output_total} != prod({input_shape}) = {input_total}."
        )

    return tuple(output_shape)


def _validate_shape_spec(shape_spec: Sequence[ShapeElement], input_ndim: int | None = None) -> None:
    """
    Validate the shape specification according to the constraints.

    Validates:
    1. Input dimensions are in monotonically increasing order.
    2. For each input_dims list with k > 0 dimensions, the constraints about
       raveled indices are satisfiable.

    Parameters
    ----------
    shape_spec : tuple[ShapeElement, ...]
        The shape specification from the codec configuration.
    input_ndim : int or None
        Number of dimensions in the input array, if available.

    Raises
    ------
    ValueError
        If any constraint is violated.
    """
    if input_ndim is None:
        dim_idxs = None
    else:
        dim_idxs = set(range(input_ndim))

    last_dim_idx = -1
    found_minus_one = False
    msg = "Arrays in reshape shape spec must contain unique indices into the input shape, in strictly monotonic increasing order"

    for elem in shape_spec:
        if isinstance(elem, int):
            pass
        else:
            for d in elem:
                if d <= last_dim_idx:
                    raise ValueError(msg)

                last_dim_idx = d

                if d == -1:
                    if found_minus_one:
                        raise ValueError("-1 should appear at most once in reshape spec")
                    found_minus_one = True
                elif d < 0:
                    raise ValueError("Negative indices not allowed in reshape spec")

                if dim_idxs is not None:
                    try:
                        dim_idxs.remove(d)
                    except KeyError as e:
                        raise ValueError(msg) from e


@dataclass(frozen=True)
class ReshapeCodec(ArrayArrayCodec):
    """Reshape codec.

    An array-to-array codec that performs a reshape operation, similar to numpy's
    reshape function. The reshape operation preserves the lexicographical (C-order)
    traversal of elements.

    Parameters
    ----------
    shape : Iterable[int | list[int]]
        An array specifying the size of each dimension of the output array as a
        function of the input array shape. Each element must be one of:
        - A positive integer specifying an explicit size.
        - A list of integers specifying input dimensions whose product determines
          the output dimension size.
        - The special value -1 (at most once), which is inferred to satisfy
          prod(output_shape) == prod(input_shape).

    Notes
    -----
    This codec does NOT alter the lexicographical order of elements. The contents
    of the output array B is related to the input array A by:
    ``ravel(B) == ravel(A)``.

    When possible, implementations should construct a virtual view rather than
    copy the array.

    References
    ----------

    - The `reshape` codec spec: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/scale_offset
    """

    is_fixed_size = True

    shape: tuple[ShapeElement, ...]

    def __init__(self, *, shape: tuple[ShapeElement, ...]) -> None:
        _validate_shape_spec(shape)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "reshape")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "reshape", "configuration": {"shape": self.shape}}

    def validate(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        """Validate the codec configuration against the array metadata."""
        _compute_output_shape(self.shape, shape)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Evolve the codec from an array specification."""
        _compute_output_shape(self.shape, array_spec.shape)
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """Resolve the output metadata after applying this codec."""
        output_shape = _compute_output_shape(self.shape, chunk_spec.shape)
        return ArraySpec(
            shape=output_shape,
            dtype=chunk_spec.dtype,
            fill_value=chunk_spec.fill_value,
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        """Decode: reshape back to the original shape."""
        # chunk_spec contains the *decoded* (original) shape
        # The input chunk_array has the encoded (reshaped) shape
        # We need to reshape it back to the original shape
        inferred_shape = _compute_output_shape(self.shape, chunk_spec.shape)
        if inferred_shape != chunk_array.shape:
            raise RuntimeError(
                "Shape of chunk to decode %s does not match shape of requested array %s encoded with reshape codec shape=%s",
                chunk_array.shape,
                chunk_spec.shape,
                self.shape,
            )
        return chunk_array.reshape(chunk_spec.shape)

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Encode: reshape to the output shape."""
        output_shape = _compute_output_shape(self.shape, chunk_spec.shape)
        return chunk_array.reshape(output_shape)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Compute the encoded size (same as input since reshape doesn't change data)."""
        return input_byte_length


__all__ = ["ReshapeCodec"]
