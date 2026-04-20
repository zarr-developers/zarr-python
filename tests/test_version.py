"""Tests for zarr._version module"""
import pytest
import importlib
import sys


def test_version_is_available():
    """Test that __version__ is available and is a string."""
    from zarr import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format():
    """Test that version follows basic format."""
    from zarr import __version__
    assert isinstance(__version__, str)
    # Basic check: should contain a dot or dash
    assert '.' in __version__ or '-' in __version__


def test_version_is_not_unknown_in_normal_case():
    from zarr import __version__
    assert __version__ != "unknown"

