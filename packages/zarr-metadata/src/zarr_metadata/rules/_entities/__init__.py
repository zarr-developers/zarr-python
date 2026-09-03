"""Per-entity composition rules, discovered automatically.

Every module here owns the rules for one codec or chunk grid and
registers them with `zarr_metadata.rules._registry` at import time.
Adding a new entity means adding a module here and nothing else: this
package imports every sibling module on import, so there is no
registration list to update and no document-rule module to edit.

That auto-discovery is the deliberate answer to two failure modes. A
hand-written registry lets a rule be defined and never registered, and a
hand-written import list lets a whole module be defined and never
imported; both produce rules that silently never run. `tests/rules/
test_registry.py` closes the remaining gap by asserting that every codec
and chunk grid the package models is either registered here or listed as
deliberately rule-free.
"""

from __future__ import annotations

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")

__all__: list[str] = []
