from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Iterable
import numpy as np
from dataclasses import dataclass

from zarr.config import config
from zarr.abc.codec import (
    Codec,
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
)
from zarr.abc.store import ByteGetter, ByteSetter, set_or_delete
from zarr.codecs.pipeline.core import CodecPipeline
from zarr.common import concurrent_map
from zarr.indexing import is_total_slice

if TYPE_CHECKING:
    from typing import List, Optional, Tuple
    from zarr.common import ArraySpec, BytesLike, SliceSelection

T = TypeVar("T")
U = TypeVar("U")


def _unzip2(iterable: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    out0: list[T] = []
    out1: list[U] = []
    for item0, item1 in iterable:
        out0.append(item0)
        out1.append(item1)
    return (out0, out1)


def resolve_batched(codec: Codec, chunk_specs: Iterable[ArraySpec]) -> Iterable[ArraySpec]:
    return [codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]


@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    def _codecs_with_resolved_metadata_batched(
        self, chunk_specs: Iterable[ArraySpec]
    ) -> Tuple[
        List[Tuple[ArrayArrayCodec, List[ArraySpec]]],
        Tuple[ArrayBytesCodec, List[ArraySpec]],
        List[Tuple[BytesBytesCodec, List[ArraySpec]]],
    ]:
        aa_codecs_with_spec: List[Tuple[ArrayArrayCodec, List[ArraySpec]]] = []
        chunk_specs = list(chunk_specs)
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, chunk_specs))
            chunk_specs = [aa_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        ab_codec_with_spec = (self.array_bytes_codec, chunk_specs)
        chunk_specs = [
            self.array_bytes_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs
        ]

        bb_codecs_with_spec: List[Tuple[BytesBytesCodec, List[ArraySpec]]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, chunk_specs))
            chunk_specs = [bb_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[Optional[BytesLike], ArraySpec]],
    ) -> Iterable[Optional[np.ndarray]]:
        chunk_bytes_batch: Iterable[Optional[BytesLike]]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata_batched(chunk_specs)

        for bb_codec, chunk_spec_batch in bb_codecs_with_spec[::-1]:
            chunk_bytes_batch = await bb_codec.decode(zip(chunk_bytes_batch, chunk_spec_batch))

        ab_codec, chunk_spec_batch = ab_codec_with_spec
        chunk_array_batch = await ab_codec.decode(zip(chunk_bytes_batch, chunk_spec_batch))

        for aa_codec, chunk_spec_batch in aa_codecs_with_spec[::-1]:
            chunk_array_batch = await aa_codec.decode(zip(chunk_array_batch, chunk_spec_batch))

        return chunk_array_batch

    async def decode_partial(
        self,
        batch_info: Iterable[Tuple[ByteGetter, SliceSelection, ArraySpec]],
    ) -> Iterable[Optional[np.ndarray]]:
        assert self.supports_partial_decode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
        return await self.array_bytes_codec.decode_partial(batch_info)

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
    ) -> Iterable[Optional[BytesLike]]:
        chunk_array_batch: Iterable[Optional[np.ndarray]]
        chunk_specs: Iterable[ArraySpec]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = await aa_codec.encode(zip(chunk_array_batch, chunk_specs))
            chunk_specs = resolve_batched(aa_codec, chunk_specs)

        chunk_bytes_batch = await self.array_bytes_codec.encode(zip(chunk_array_batch, chunk_specs))
        chunk_specs = resolve_batched(self.array_bytes_codec, chunk_specs)

        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = await bb_codec.encode(zip(chunk_bytes_batch, chunk_specs))
            chunk_specs = resolve_batched(bb_codec, chunk_specs)

        return chunk_bytes_batch

    async def encode_partial(
        self,
        batch_info: Iterable[Tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
    ) -> None:
        assert self.supports_partial_encode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
        await self.array_bytes_codec.encode_partial(batch_info)

    async def read_batch(
        self,
        batch_info: Iterable[Tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: np.ndarray,
    ) -> None:
        if self.supports_partial_decode:
            chunk_array_batch = await self.decode_partial(
                [
                    (byte_getter, chunk_selection, chunk_spec)
                    for byte_getter, chunk_spec, chunk_selection, _ in batch_info
                ]
            )
            for chunk_array, (_, chunk_spec, _, out_selection) in zip(
                chunk_array_batch, batch_info
            ):
                if chunk_array is not None:
                    out[out_selection] = chunk_array
                else:
                    out[out_selection] = chunk_spec.fill_value
        else:
            chunk_bytes_batch = await concurrent_map(
                [(byte_getter,) for byte_getter, _, _, _ in batch_info],
                lambda byte_getter: byte_getter.get(),
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, _, _) in zip(chunk_bytes_batch, batch_info)
                ],
            )
            for chunk_array, (_, chunk_spec, chunk_selection, out_selection) in zip(
                chunk_array_batch, batch_info
            ):
                if chunk_array is not None:
                    tmp = chunk_array[chunk_selection]
                    out[out_selection] = tmp
                else:
                    out[out_selection] = chunk_spec.fill_value

    async def write_batch(
        self,
        batch_info: Iterable[Tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: np.ndarray,
    ) -> None:
        if self.supports_partial_encode:
            await self.encode_partial(
                [
                    (byte_setter, value[out_selection], chunk_selection, chunk_spec)
                    for byte_setter, chunk_spec, chunk_selection, out_selection in batch_info
                ],
            )

        else:
            # Read existing bytes if not total slice
            async def _read_key(byte_setter: Optional[ByteSetter]) -> Optional[BytesLike]:
                if byte_setter is None:
                    return None
                return await byte_setter.get()

            chunk_bytes_batch: Iterable[Optional[BytesLike]]
            chunk_bytes_batch = await concurrent_map(
                [
                    (None if is_total_slice(chunk_selection, chunk_spec.shape) else byte_setter,)
                    for byte_setter, chunk_spec, chunk_selection, _ in batch_info
                ],
                _read_key,
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, _, _) in zip(chunk_bytes_batch, batch_info)
                ],
            )

            def _merge_chunk_array(
                existing_chunk_array: Optional[np.ndarray],
                new_chunk_array_slice: np.ndarray,
                chunk_spec: ArraySpec,
                chunk_selection: SliceSelection,
            ) -> np.ndarray:
                if is_total_slice(chunk_selection, chunk_spec.shape):
                    return new_chunk_array_slice
                if existing_chunk_array is None:
                    chunk_array = np.empty(
                        chunk_spec.shape,
                        dtype=chunk_spec.dtype,
                    )
                    chunk_array.fill(chunk_spec.fill_value)
                else:
                    chunk_array = existing_chunk_array.copy()  # make a writable copy
                chunk_array[chunk_selection] = new_chunk_array_slice
                return chunk_array

            chunk_array_batch = [
                _merge_chunk_array(chunk_array, value[out_selection], chunk_spec, chunk_selection)
                for chunk_array, (_, chunk_spec, chunk_selection, out_selection) in zip(
                    chunk_array_batch, batch_info
                )
            ]

            chunk_array_batch = [
                None if np.all(chunk_array == chunk_spec.fill_value) else chunk_array
                for chunk_array, (_, chunk_spec, _, _) in zip(chunk_array_batch, batch_info)
            ]

            chunk_bytes_batch = await self.encode(
                [
                    (chunk_array, chunk_spec)
                    for chunk_array, (_, chunk_spec, _, _) in zip(chunk_array_batch, batch_info)
                ],
            )

            await concurrent_map(
                [
                    (byte_setter, chunk_bytes)
                    for chunk_bytes, (byte_setter, _, _, _) in zip(chunk_bytes_batch, batch_info)
                ],
                set_or_delete,
                config.get("async.concurrency"),
            )
