"""Pytest configuration for benchmark tests."""

import pytest

# Filter CodSpeed instrumentation warnings that can occur intermittently
# when registering benchmark results. This is a known issue with the
# CodSpeed walltime instrumentation hooks.
# See: https://github.com/CodSpeedHQ/pytest-codspeed


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Failed to set executed benchmark:RuntimeWarning",
    )
