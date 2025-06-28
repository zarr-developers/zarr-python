# /// script
# requires-python = "==3.12"
# dependencies = [
#  "zarr==3.0.8"
# ]
# ///


import argparse

import zarr
from zarr.abc.store import Store

def copy_group(
    *, node: zarr.Group, store: Store, path: str, overwrite: bool
) -> zarr.Group:
    result = zarr.create_group(
        store=store,
        path=path,
        overwrite=overwrite,
        attributes=node.attrs.asdict(),
        zarr_format=node.metadata.zarr_format)
    for key, child in node.members():
        child_path = f"{path}/{key}"
        if isinstance(child, zarr.Group):
            copy_group(node=child, store=store, path=child_path, overwrite=overwrite)
        else:
            copy_array(node=child, store=store, overwrite=overwrite, path=child_path)
    return result


def copy_array(
    *, node: zarr.Array, store: Store, path: str, overwrite: bool
) -> zarr.Array:
    result = zarr.from_array(store, name=path, data=node, write_data=True)
    return result


def copy_node(
    node: zarr.Group | zarr.Array, store: Store, path: str, overwrite: bool
) -> zarr.Group | zarr.Array:
    if isinstance(node, zarr.Group):
        return copy_group(node=node, store=store, path=path, overwrite=overwrite)
    else:
        return copy_array(node=node, store=store, path=path, overwrite=overwrite)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a zarr hierarchy from one location to another"
    )
    parser.add_argument("source", type=str, help="Path to the source zarr hierarchy")
    parser.add_argument("destination", type=str, help="Path to the destination zarr hierarchy")
    args = parser.parse_args()

    src, dst = args.source, args.destination
    root_src = zarr.open(src, mode="r")
    result = copy_node(node=root_src, store=dst, path="", overwrite=True)

    print(f"successfully created {result} at {dst}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
