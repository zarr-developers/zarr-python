from typing import Annotated, Literal, cast

import typer

from zarr.core.metadata.converter.converter_v2_v3 import convert_v2_to_v3, remove_metadata
from zarr.core.sync import sync

app = typer.Typer()


@app.command()  # type: ignore[misc]
def convert(
    store: Annotated[
        str,
        typer.Argument(
            help="Store or path to directory in file system or name of zip file e.g. 'data/example-1.zarr', 's3://example-bucket/example'..."
        ),
    ],
    path: Annotated[str | None, typer.Option(help="The path within the store to open")] = None,
) -> None:
    """Convert all v2 metadata in a zarr hierarchy to v3. This will create a zarr.json file at each level
    (for every group / array). V2 files (.zarray, .zattrs etc.) will be left as-is.
    """
    convert_v2_to_v3(store=store, path=path)


@app.command()  # type: ignore[misc]
def clear(
    store: Annotated[
        str,
        typer.Argument(
            help="Store or path to directory in file system or name of zip file e.g. 'data/example-1.zarr', 's3://example-bucket/example'..."
        ),
    ],
    zarr_format: Annotated[
        int,
        typer.Argument(
            help="Which format's metadata to remove - 2 or 3.",
            min=2,
            max=3,
        ),
    ],
    path: Annotated[str | None, typer.Option(help="The path within the store to open")] = None,
) -> None:
    """Remove all v2 (.zarray, .zattrs, .zgroup, .zmetadata) or v3 (zarr.json) metadata files from the given Zarr.
    Note - this will remove metadata files at all levels of the hierarchy (every group and array).
    """
    sync(remove_metadata(store=store, zarr_format=cast(Literal[2, 3], zarr_format), path=path))


if __name__ == "__main__":
    app()
