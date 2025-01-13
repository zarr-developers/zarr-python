import zarr


def test_exports() -> None:
    """
    Ensure that everything in __all__ can be imported.
    """
    from zarr import __all__

    for export in __all__:
        getattr(zarr, export)
