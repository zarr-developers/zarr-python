"""
Copyright © 2023 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from zarr.array import Array
from zarr.group import Group, GroupMetadata
from zarr.metadata import ArrayV3Metadata
from zarr.store.core import StorePath


class ArrayModel(ArrayV3Metadata):
    """
    A model of a Zarr v3 array.
    """

    @classmethod
    def from_stored(cls: type[Self], node: Array):
        return cls.from_dict(node.metadata.to_dict())

    def to_stored(self, store_path: StorePath) -> Array:
        return Array.from_dict(store_path=store_path, data=self.to_dict())


class GroupModel(GroupMetadata):
    """
    A model of a Zarr v3 group.
    """

    members: dict[str, GroupModel | ArrayModel] | None

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]):
        return cls(**data)

    @classmethod
    def from_stored(cls: type[Self], node: Group, *, depth: int | None = None) -> Self:
        """
        Create a GroupModel from a Group. This function is recursive. The depth of recursion is
        controlled by the `depth` argument, which is either None (no depth limit) or a finite natural number
        specifying how deep into the hierarchy to parse.
        """
        members: dict[str, GroupModel | ArrayModel]

        if depth is None:
            new_depth = depth
        else:
            new_depth = depth - 1

        if depth == 0:
            return cls(**node.metadata.to_dict(), members=None)

        else:
            for name, member in node.members():
                if isinstance(member, Array):
                    item_out = ArrayModel.from_stored(member)
                else:
                    item_out = GroupModel.from_stored(member, depth=new_depth)

                members[name] = item_out

        return cls(**node.metadata.to_dict(), members=members)

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
    result = {}
    model_copy: ArrayModel | GroupModel
    node_dict = node.to_dict()
    if isinstance(node, ArrayModel):
        model_copy = ArrayModel(**node_dict)
    else:
        members = node_dict.pop("members")
        model_copy = GroupModel(node_dict)
        if members is not None:
            for name, value in node.members.items():
                result.update(to_flat(value, "/".join([root_path, name])))

    result[root_path] = model_copy
    # sort by increasing key length
    result_sorted_keys = dict(sorted(result.items(), key=lambda v: len(v[0])))
    return result_sorted_keys


def from_flat(data: dict[str, ArrayModel | GroupModel]) -> ArrayModel | GroupModel:
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
