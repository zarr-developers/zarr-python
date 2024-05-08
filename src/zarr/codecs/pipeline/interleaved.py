from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

from zarr.abc.codec import (
    ByteGetter,
    ByteSetter,
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
)
from zarr.codecs.pipeline.core import CodecPipeline
from zarr.common import concurrent_map
from zarr.indexing import is_total_slice

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Iterable
    from zarr.config import RuntimeConfiguration
    from zarr.common import ArraySpec, BytesLike, SliceSelection


@dataclass(frozen=True)
class InterleavedCodecPipeline(CodecPipeline):
    def _codecs_with_resolved_metadata(
        self, chunk_spec: ArraySpec
    ) -> Tuple[
        List[Tuple[ArrayArrayCodec, ArraySpec]],
        Tuple[ArrayBytesCodec, ArraySpec],
        List[Tuple[BytesBytesCodec, ArraySpec]],
    ]:
        aa_codecs_with_spec: List[Tuple[ArrayArrayCodec, ArraySpec]] = []
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, chunk_spec))
            chunk_spec = aa_codec.resolve_metadata(chunk_spec)

        ab_codec_with_spec = (self.array_bytes_codec, chunk_spec)
        chunk_spec = self.array_bytes_codec.resolve_metadata(chunk_spec)

        bb_codecs_with_spec: List[Tuple[BytesBytesCodec, ArraySpec]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, chunk_spec))
            chunk_spec = bb_codec.resolve_metadata(chunk_spec)

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[Optional[BytesLike], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        return await concurrent_map(
            [
                (chunk_bytes, chunk_spec, runtime_configuration)
                for chunk_bytes, chunk_spec in chunk_bytes_and_specs
            ],
            self.decode_single,
            runtime_configuration.concurrency,
        )

    async def decode_single(
        self,
        chunk_bytes: Optional[BytesLike],
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        if chunk_bytes is None:
            return None

        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata(chunk_spec)

        for bb_codec, chunk_spec in bb_codecs_with_spec[::-1]:
            chunk_bytes = await bb_codec.decode(chunk_bytes, chunk_spec, runtime_configuration)

        ab_codec, chunk_spec = ab_codec_with_spec
        chunk_array = await ab_codec.decode(chunk_bytes, chunk_spec, runtime_configuration)

        for aa_codec, chunk_spec in aa_codecs_with_spec[::-1]:
            chunk_array = await aa_codec.decode(chunk_array, chunk_spec, runtime_configuration)

        return chunk_array

    async def decode_partial(
        self,
        batch_info: Iterable[Tuple[ByteGetter, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        return await concurrent_map(
            [
                (byte_getter, selection, chunk_spec, runtime_configuration)
                for byte_getter, selection, chunk_spec in batch_info
            ],
            self.decode_partial_single,
            runtime_configuration.concurrency,
        )

    async def decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        assert self.supports_partial_decode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
        return await self.array_bytes_codec.decode_partial(
            byte_getter, selection, chunk_spec, runtime_configuration
        )

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[BytesLike]]:
        return await concurrent_map(
            [
                (chunk_array, chunk_spec, runtime_configuration)
                for chunk_array, chunk_spec in chunk_arrays_and_specs
            ],
            self.encode_single,
            runtime_configuration.concurrency,
        )

    async def encode_single(
        self,
        chunk_array: Optional[np.ndarray],
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        if chunk_array is None:
            return None

        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata(chunk_spec)

        for aa_codec, chunk_spec in aa_codecs_with_spec:
            chunk_array_maybe = await aa_codec.encode(
                chunk_array, chunk_spec, runtime_configuration
            )
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        ab_codec, array_spec = ab_codec_with_spec
        chunk_bytes_maybe = await ab_codec.encode(chunk_array, array_spec, runtime_configuration)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec, array_spec in bb_codecs_with_spec:
            chunk_bytes_maybe = await bb_codec.encode(
                chunk_bytes, array_spec, runtime_configuration
            )
            if chunk_bytes_maybe is None:
                return None
            chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    async def encode_partial(
        self,
        batch_info: Iterable[Tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (byte_setter, chunk_array, selection, chunk_spec, runtime_configuration)
                for byte_setter, chunk_array, selection, chunk_spec in batch_info
            ],
            self.encode_partial_single,
            runtime_configuration.concurrency,
        )

    async def encode_partial_single(
        self,
        byte_setter: ByteSetter,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        assert self.supports_partial_encode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
        await self.array_bytes_codec.encode_partial(
            byte_setter, chunk_array, selection, chunk_spec, runtime_configuration
        )

    async def read_single(
        self,
        byte_getter: ByteGetter,
        chunk_spec: ArraySpec,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        out: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        if self.supports_partial_decode:
            chunk_array = await self.decode_partial_single(
                byte_getter, chunk_selection, chunk_spec, runtime_configuration
            )
            if chunk_array is not None:
                out[out_selection] = chunk_array
            else:
                out[out_selection] = chunk_spec.fill_value
        else:
            chunk_bytes = await byte_getter.get()
            chunk_array = await self.decode_single(chunk_bytes, chunk_spec, runtime_configuration)
            if chunk_array is not None:
                tmp = chunk_array[chunk_selection]
                out[out_selection] = tmp
            else:
                out[out_selection] = chunk_spec.fill_value

    async def read_batch(
        self,
        batch_info: Iterable[Tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (
                    byte_getter,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    out,
                    runtime_configuration,
                )
                for byte_getter, chunk_spec, chunk_selection, out_selection in batch_info
            ],
            self.read_single,
            runtime_configuration.concurrency,
        )

    async def write_single(
        self,
        byte_setter: ByteSetter,
        chunk_spec: ArraySpec,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        value: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        async def _write_chunk_to_store(chunk_array: np.ndarray) -> None:
            if np.all(chunk_array == chunk_spec.fill_value):
                # chunks that only contain fill_value will be removed
                await byte_setter.delete()
            else:
                chunk_bytes = await self.encode_single(
                    chunk_array, chunk_spec, runtime_configuration
                )
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

        if is_total_slice(chunk_selection, chunk_spec.shape):
            # write entire chunks
            if np.isscalar(value):
                chunk_array = np.empty(
                    chunk_spec.shape,
                    dtype=chunk_spec.dtype,
                )
                chunk_array.fill(value)
            else:
                chunk_array = value[out_selection]
            await _write_chunk_to_store(chunk_array)

        elif self.supports_partial_encode:
            await self.encode_partial_single(
                byte_setter,
                value[out_selection],
                chunk_selection,
                chunk_spec,
                runtime_configuration,
            )
        else:
            # writing partial chunks
            # read chunk first
            chunk_bytes = await byte_setter.get()

            # merge new value
            chunk_array_maybe = await self.decode_single(
                chunk_bytes, chunk_spec, runtime_configuration
            )
            if chunk_array_maybe is None:
                chunk_array = np.empty(
                    chunk_spec.shape,
                    dtype=chunk_spec.dtype,
                )
                chunk_array.fill(chunk_spec.fill_value)
            else:
                chunk_array = chunk_array_maybe.copy()  # make a writable copy
            chunk_array[chunk_selection] = value[out_selection]

            await _write_chunk_to_store(chunk_array)

    async def write_batch(
        self,
        batch_info: Iterable[Tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (
                    byte_setter,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    value,
                    runtime_configuration,
                )
                for byte_setter, chunk_spec, chunk_selection, out_selection in batch_info
            ],
            self.write_single,
            runtime_configuration.concurrency,
        )
