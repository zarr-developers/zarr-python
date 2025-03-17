import zarr


def test_exports() -> None:
    """
    Ensure that everything in __all__ can be imported.
    """
    from zarr import __all__

    for export in __all__:
        getattr(zarr, export)


def test_print_debug_info() -> None:
    """
    Ensure that print_debug_info does not raise an error
    """
    from zarr import print_debug_info

    print_debug_info()
