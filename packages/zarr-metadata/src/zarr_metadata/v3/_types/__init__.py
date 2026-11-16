"""The types of the v3 extensions, one module per kind of extension.

Every module in this package holds types only and imports no definition,
so a definition of any kind can import the types of any other at run
time. Each name is public in the extension's own submodule, which
re-exports it.
"""
