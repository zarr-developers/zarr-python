# /// script
# requires-python = ">=3.11"
# dependencies = [
#  "zarr==3.0.8"
# ]
# ///

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
