# A common file that can be used to add constants, functions,
# convenience classes, etc. that are shared across multiple tests

import platform
import sys

import pytest

IS_WASM = sys.platform == "emscripten" or platform.machine() in ["wasm32", "wasm64"]


def asyncio_tests_wrapper(func):
    if IS_WASM:
        return func
    else:
        return pytest.mark.asyncio(func)
