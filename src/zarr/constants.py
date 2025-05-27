import platform
import sys

IS_WASM: bool =  sys.platform == "emscripten" or platform.machine() in ["wasm32", "wasm64"]
