# This file only exists to not incur circular import issues
# TODO: find a better location for this or keep it here

from __future__ import annotations

import platform
import sys

IS_WASM: bool = sys.platform == "emscripten" or platform.machine() in ["wasm32", "wasm64"]
