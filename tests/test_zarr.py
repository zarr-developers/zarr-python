import pytest

import zarr


def test_exports() -> None:
    """
    Ensure that everything in __all__ can be imported.
    """
    from zarr import __all__

    for export in __all__:
        getattr(zarr, export)


def test_print_debug_info(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Ensure that print_debug_info does not raise an error
    """
    from importlib.metadata import version

    from zarr import __version__, print_debug_info

    print_debug_info()
    captured = capsys.readouterr()
    # test that at least some of what we expect is
    # printed out
    assert f"zarr: {__version__}" in captured.out
    assert f"numpy: {version('numpy')}" in captured.out
