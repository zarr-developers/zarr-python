"""
Copyright © 2023 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from typing_extensions import Self

from zarr.abc.codec import CodecPipeline
from zarr.array import Array
from zarr.buffer import NDBuffer
from zarr.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.chunk_key_encodings import ChunkKeyEncoding, DefaultChunkKeyEncoding
from zarr.codecs.bytes import BytesCodec
from zarr.group import Group, GroupMetadata
from zarr.metadata import ArrayV3Metadata
from zarr.store.core import StorePath
from zarr.v2.util import guess_chunks


def auto_data_type(data: Any) -> Any:
    if hasattr(data, "dtype"):
        if hasattr(data, "data_type"):
            msg = (
                f"Could not infer the data_type attribute from {data}, because "
                "it has both `dtype` and `data_type` attributes. "
                "This method requires input with one, or the other, of these attributes."
            )
            raise ValueError(msg)
        return data.dtype
    elif hasattr(data, "data_type") and not hasattr(data, "dtype"):
        return data.data_type
    else:
        msg = (
            f"Could not infer the data_type attribute from {data}. "
            "Expected either an object with a `dtype` attribute, "
            "or an object with a `data_type` attribute."
        )
        raise ValueError(msg)


def auto_attributes(data: Any) -> Any:
    """
    Guess attributes from:
        input with an `attrs` attribute, or
        input with an `attributes` attribute,
        or anything (returning {})
    """
    if hasattr(data, "attrs"):
        return data.attrs
    if hasattr(data, "attributes"):
        return data.attributes
    return {}


def auto_chunk_key_encoding(data: Any) -> Any:
    if hasattr(data, "chunk_key_encoding"):
        return data.chunk_key_encoding
    return DefaultChunkKeyEncoding()


def auto_fill_value(data: Any) -> Any:
    """
    Guess fill value from an input with a `fill_value` attribute, returning 0 otherwise.
    """
    if hasattr(data, "fill_value"):
        return data.fill_value
    return 0


def auto_codecs(data: Any) -> Any:
    """
    Guess compressor from an input with a `compressor` attribute, returning `None` otherwise.
    """
    if hasattr(data, "codecs"):
        return data.codecs
    return (BytesCodec(),)


def auto_dimension_names(data: Any) -> Any:
    """
    If the input has a `dimension_names` attribute, return it, otherwise
    return None.
    """

    if hasattr(data, "dimension_names"):
        return data.dimension_names
    return None


def auto_chunk_grid(data: Any) -> Any:
    """
    Guess a chunk grid from:
      input with a `chunk_grid` attribute,
      input with a `chunksize` attribute, or
      input with a `chunks` attribute, or,
      input with `shape` and `dtype` attributes
    """
    if hasattr(data, "chunk_grid"):
        # more a statement of intent than anything else
        return data.chunk_grid
    if hasattr(data, "chunksize"):
        chunks = data.chunksize
    elif hasattr(data, "chunks"):
        chunks = data.chunks
    else:
        chunks = guess_chunks(data.shape, np.dtype(data.dtype).itemsize)
    return RegularChunkGrid(chunk_shape=chunks)


class ArrayModel(ArrayV3Metadata):
    """
    A model of a Zarr v3 array.
    """

    @classmethod
    def from_stored(cls: type[Self], node: Array) -> Self:
        """
        Create an array model from a stored array.
        """
        return cls.from_dict(node.metadata.to_dict())

    def to_stored(self, store_path: StorePath, exists_ok: bool = False) -> Array:
        """
        Create a stored version of this array.
        """
        # exists_ok kwarg is unhandled until we wire it up to the
        # array creation routines

        return Array.from_dict(store_path=store_path, data=self.to_dict())

    @classmethod
    def from_array(
        cls: type[Self],
        data: NDBuffer,
        *,
        chunk_grid: ChunkGrid | Literal["auto"] = "auto",
        chunk_key_encoding: ChunkKeyEncoding | Literal["auto"] = "auto",
        fill_value: Any | Literal["auto"] = "auto",
        codecs: CodecPipeline | Literal["auto"] = "auto",
        attributes: dict[str, Any] | Literal["auto"] = "auto",
        dimension_names: tuple[str, ...] | Literal["auto"] = "auto",
    ) -> Self:
        """
        Create an ArrayModel from an array-like object, e.g. a numpy array.

        The returned ArrayModel will use the shape and dtype attributes of the input.
        The remaining ArrayModel attributes are exposed by this method as keyword arguments,
        which can either be the string "auto", which instructs this method to infer or guess
        a value, or a concrete value to use.
        """
        shape_out = data.shape
        data_type_out = auto_data_type(data)

        if chunk_grid == "auto":
            chunk_grid_out = auto_chunk_grid(data)
        else:
            chunk_grid_out = chunk_grid

        if chunk_key_encoding == "auto":
            chunk_key_encoding_out = auto_chunk_key_encoding(data)
        else:
            chunk_key_encoding_out = chunk_key_encoding

        if fill_value == "auto":
            fill_value_out = auto_fill_value(data)
        else:
            fill_value_out = fill_value

        if codecs == "auto":
            codecs_out = auto_codecs(data)
        else:
            codecs_out = codecs

        if attributes == "auto":
            attributes_out = auto_attributes(data)
        else:
            attributes_out = attributes

        if dimension_names == "auto":
            dimension_names_out = auto_dimension_names(data)
        else:
            dimension_names_out = dimension_names

        return cls(
            shape=shape_out,
            data_type=data_type_out,
            chunk_grid=chunk_grid_out,
            chunk_key_encoding=chunk_key_encoding_out,
            fill_value=fill_value_out,
            codecs=codecs_out,
            attributes=attributes_out,
            dimension_names=dimension_names_out,
        )


@dataclass(frozen=True)
class GroupModel(GroupMetadata):
    """
    A model of a Zarr v3 group.
    """

    members: dict[str, GroupModel | ArrayModel] | None = field(default_factory=dict)

    @classmethod
    def from_stored(cls: type[Self], node: Group, *, depth: int | None = None) -> Self:
        """
        Create a GroupModel from a Group. This function is recursive. The depth of recursion is
        controlled by the `depth` argument, which is either None (no depth limit) or a finite natural number
        specifying how deep into the hierarchy to parse.
        """
        members: dict[str, GroupModel | ArrayModel] = {}

        if depth is None:
            new_depth = depth
        else:
            new_depth = depth - 1

        if depth == 0:
            return cls(**node.metadata.to_dict(), members=None)

        else:
            for name, member in node.members:
                item_out: ArrayModel | GroupModel
                if isinstance(member, Array):
                    item_out = ArrayModel.from_stored(member)
                else:
                    item_out = GroupModel.from_stored(member, depth=new_depth)

                members[name] = item_out

        return cls(attributes=node.metadata.attributes, members=members)

    # todo: make this async
    def to_stored(self, store_path: StorePath, *, exists_ok: bool = False) -> Group:
        """
        Serialize this GroupModel to storage.
        """

        result = Group.create(store_path, attributes=self.attributes, exists_ok=exists_ok)
        if self.members is not None:
            for name, member in self.members.items():
                substore = store_path / name
                member.to_stored(substore, exists_ok=exists_ok)
        return result


def to_flat(
    node: ArrayModel | GroupModel, root_path: str = ""
) -> dict[str, ArrayModel | GroupModel]:
    """
    Generate a dict representation of an ArrayModel or GroupModel, where the hierarchy structure
    is represented by the keys of the dict.
    """
    result = {}
    model_copy: ArrayModel | GroupModel
    if isinstance(node, ArrayModel):
        # we can remove this if we add a model_copy method
        model_copy = ArrayModel(
            shape=node.shape,
            data_type=node.data_type,
            chunk_grid=node.chunk_grid,
            chunk_key_encoding=node.chunk_key_encoding,
            fill_value=node.fill_value,
            codecs=node.codecs,
            attributes=node.attributes,
            dimension_names=node.dimension_names,
        )
    else:
        model_copy = GroupModel(attributes=node.attributes, members=None)
        if node.members is not None:
            for name, value in node.members.items():
                result.update(to_flat(value, "/".join([root_path, name])))

    result[root_path] = model_copy
    # sort by increasing key length
    result_sorted_keys = dict(sorted(result.items(), key=lambda v: len(v[0])))
    return result_sorted_keys


def from_flat(data: dict[str, ArrayModel | GroupModel]) -> ArrayModel | GroupModel:
    """
    Create a GroupModel or ArrayModel from a dict representation.
    """
    # minimal check that the keys are valid
    invalid_keys = []
    for key in data.keys():
        if key.endswith("/"):
            invalid_keys.append(key)
    if len(invalid_keys) > 0:
        msg = f'Invalid keys {invalid_keys} found in data. Keys may not end with the "/"" character'
        raise ValueError(msg)

    if tuple(data.keys()) == ("",) and isinstance(tuple(data.values())[0], ArrayModel):
        return tuple(data.values())[0]
    else:
        return from_flat_group(data)


def from_flat_group(data: dict[str, ArrayModel | GroupModel]) -> GroupModel:
    """
    Create a GroupModel from a hierarchy represented as a dict with string keys and ArrayModel
    or GroupModel values.
    """
    root_name = ""
    sep = "/"
    # arrays that will be members of the returned GroupModel
    member_arrays: dict[str, ArrayModel] = {}
    # groups, and their members, that will be members of the returned GroupModel.
    # this dict is populated by recursively applying `from_flat_group` function.
    member_groups: dict[str, GroupModel] = {}
    # this dict collects the arrayspecs and groupspecs that belong to one of the members of the
    # groupspecs we are constructing. They will later be aggregated in a recursive step that
    # populates member_groups
    submember_by_parent_name: dict[str, dict[str, ArrayModel | GroupModel]] = {}
    # copy the input to ensure that mutations are contained inside this function
    data_copy = data.copy()
    # Get the root node
    try:
        # The root node is a GroupModel with the key ""
        root_node = data_copy.pop(root_name)
        if isinstance(root_node, ArrayModel):
            raise ValueError("Got an ArrayModel as the root node. This is invalid.")
    except KeyError:
        # If a root node was not found, create a default one
        root_node = GroupModel(attributes={}, members=None)

    # partition the tree (sans root node) into 2 categories: (arrays, groups + their members).
    for key, value in data_copy.items():
        key_parts = key.split(sep)
        if key_parts[0] != root_name:
            raise ValueError(f'Invalid path: {key} does not start with "{root_name}{sep}".')

        subparent_name = key_parts[1]
        if len(key_parts) == 2:
            # this is an array or group that belongs to the group we are ultimately returning
            if isinstance(value, ArrayModel):
                member_arrays[subparent_name] = value
            else:
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                submember_by_parent_name[subparent_name][root_name] = value
        else:
            # these are groups or arrays that belong to one of the member groups
            # not great that we repeat this conditional dict initialization
            if subparent_name not in submember_by_parent_name:
                submember_by_parent_name[subparent_name] = {}
            submember_by_parent_name[subparent_name][sep.join([root_name, *key_parts[2:]])] = value

    # recurse
    for subparent_name, submemb in submember_by_parent_name.items():
        member_groups[subparent_name] = from_flat_group(submemb)

    return GroupModel(members={**member_groups, **member_arrays}, attributes=root_node.attributes)
