from typing import Annotated

import typer

from zarr.core.metadata.converter.converter_v2_v3 import convert_v2_to_v3

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
def clear() -> None:
    print("Clearing...")


if __name__ == "__main__":
    app()
