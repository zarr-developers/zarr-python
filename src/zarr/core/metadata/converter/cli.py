import logging
from typing import Annotated, Literal, cast

import typer

from zarr.core.metadata.converter.converter_v2_v3 import convert_v2_to_v3, remove_metadata
from zarr.core.sync import sync

app = typer.Typer()

logger = logging.getLogger(__name__)


def _set_logging_config(verbose: bool) -> None:
    if verbose:
        lvl = logging.INFO
    else:
        lvl = logging.WARNING
    fmt = "%(message)s"
    logging.basicConfig(level=lvl, format=fmt)


def _set_verbose_level() -> None:
    logging.getLogger().setLevel(logging.INFO)


@app.command()  # type: ignore[misc]
def convert(
    store: Annotated[
        str,
        typer.Argument(
            help="Store or path to directory in file system or name of zip file e.g. 'data/example-1.zarr', 's3://example-bucket/example'..."
        ),
    ],
    path: Annotated[str | None, typer.Option(help="The path within the store to open")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Enable a dry-run: files that would be converted are logged, but no new files are actually created."
        ),
    ] = False,
) -> None:
    """Convert all v2 metadata in a zarr hierarchy to v3. This will create a zarr.json file at each level
    (for every group / array). V2 files (.zarray, .zattrs etc.) will be left as-is.
    """
    if dry_run:
        _set_verbose_level()
        logger.info(
            "Dry run enabled - no new files will be created. Log of files that would be created on a real run:"
        )

    convert_v2_to_v3(store=store, path=path, dry_run=dry_run)


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
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Enable a dry-run: files that would be deleted are logged, but no files are actually removed."
        ),
    ] = False,
) -> None:
    """Remove all v2 (.zarray, .zattrs, .zgroup, .zmetadata) or v3 (zarr.json) metadata files from the given Zarr.
    Note - this will remove metadata files at all levels of the hierarchy (every group and array).
    """
    if dry_run:
        _set_verbose_level()
        logger.info(
            "Dry run enabled - no files will be deleted. Log of files that would be deleted on a real run:"
        )

    sync(
        remove_metadata(
            store=store, zarr_format=cast(Literal[2, 3], zarr_format), path=path, dry_run=dry_run
        )
    )


@app.callback()  # type: ignore[misc]
def main(
    verbose: Annotated[
        bool,
        typer.Option(
            help="enable verbose logging - will print info about metadata files being deleted / saved."
        ),
    ] = False,
) -> None:
    """
    Convert metadata from v2 to v3. See available commands below - access help for individual commands with
    cli.py COMMAND --help.
    """
    _set_logging_config(verbose)


if __name__ == "__main__":
    app()
