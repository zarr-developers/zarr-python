# /// script
# requires-python = ">=3.11"
# dependencies = [
#  "zarr==2.18",
#  "numcodecs==0.15"
# ]
# ///

import argparse

import zarr
from zarr._storage.store import BaseStore


def copy_group(
    *, node: zarr.hierarchy.Group, store: zarr.storage.BaseStore, path: str, overwrite: bool
) -> zarr.hierarchy.Group:
    result = zarr.group(store=store, path=path, overwrite=overwrite)
    result.attrs.put(node.attrs.asdict())
    for key, child in node.items():
        child_path = f"{path}/{key}"
        if isinstance(child, zarr.hierarchy.Group):
            copy_group(node=child, store=store, path=child_path, overwrite=overwrite)
        elif isinstance(child, zarr.core.Array):
            copy_array(node=child, store=store, overwrite=overwrite, path=child_path)
    return result


def copy_array(
    *, node: zarr.core.Array, store: BaseStore, path: str, overwrite: bool
) -> zarr.core.Array:
    result = zarr.create(
        shape=node.shape,
        dtype=node.dtype,
        fill_value=node.fill_value,
        chunks=node.chunks,
        compressor=node.compressor,
        filters=node.filters,
        order=node.order,
        dimension_separator=node._dimension_separator,
        store=store,
        path=path,
        overwrite=overwrite,
    )
    result.attrs.put(node.attrs.asdict())
    result[:] = node[:]
    return result


def copy_node(
    node: zarr.hierarchy.Group | zarr.core.Array, store: BaseStore, path: str, overwrite: bool
) -> zarr.hierarchy.Group | zarr.core.Array:
    if isinstance(node, zarr.hierarchy.Group):
        return copy_group(node=node, store=store, path=path, overwrite=overwrite)
    elif isinstance(node, zarr.core.Array):
        return copy_array(node=node, store=store, path=path, overwrite=overwrite)
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")  # pragma: no cover


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a zarr hierarchy from one location to another"
    )
    parser.add_argument("source", type=str, help="Path to the source zarr hierarchy")
    parser.add_argument("destination", type=str, help="Path to the destination zarr hierarchy")
    args = parser.parse_args()

    src, dst = args.source, args.destination
    root_src = zarr.open(src, mode="r")
    result = copy_node(node=root_src, store=zarr.NestedDirectoryStore(dst), path="", overwrite=True)

    print(f"successfully created {result} at {dst}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
