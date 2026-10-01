"""MkDocs hook that renders validation-marked code fences as ordinary code blocks.

The docs validation convention (see ``tests/test_docs.py`` and the contributing
guide) requires every python fence to carry ``exec="true"``, ``test="true"``, or
``exec="false" reason="..."``. Markdown Exec's superfences fence only claims
``exec="true"`` blocks; without this hook the remaining marked fences fail
superfences validation and their contents spill into the page as raw markdown
(e.g. the PEP 723 header of the custom dtype example rendered as headings).

This hook registers a second ``python`` fence, tried when Markdown Exec's
declines, that strips the validation attributes and delegates to the standard
superfences highlighter so the block renders exactly like a plain code fence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from markdown import Markdown
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

# Mirrors markdown_exec's _to_bool: everything but these means "true".
_FALSY = {"", "no", "off", "false", "0"}


def _validator(
    language: str,
    inputs: dict[str, str],
    options: dict[str, Any],
    attrs: dict[str, Any],
    md: Markdown,
) -> bool:
    """Claim fences marked test="true" or exec="false"; leave the rest alone."""
    if "exec" not in inputs and "test" not in inputs:
        # Plain fence: let the default superfences pathway highlight it.
        return False
    if str(inputs.get("exec", "false")).lower() not in _FALSY:
        # Executable fence: Markdown Exec's own custom fence handles it.
        return False
    # Consume the validation attributes so they don't leak into the output.
    inputs.clear()
    return True


def _formatter(
    source: str,
    language: str,
    css_class: str,
    options: dict[str, Any],
    md: Markdown,
    classes: list[str] | None = None,
    id_value: str = "",
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Render with the same highlighter superfences uses for plain fences."""
    fenced = md.preprocessors["fenced_code_block"]
    fenced.get_hl_settings()
    return fenced.highlight(
        src=source,
        language=language,
        options={},
        md=md,
        classes=classes,
        id_value=id_value,
        attrs=attrs or {},
    )


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    superfences = config.setdefault("mdx_configs", {}).setdefault("pymdownx.superfences", {})
    custom_fences = superfences.setdefault("custom_fences", [])
    custom_fences.append(
        {
            "name": "python",
            "class": "python",
            "validator": _validator,
            "format": _formatter,
        }
    )
    return config


# The deprecations page holds these markers where its generated tables go. The tables
# are built at build time, from the source's ``@deprecated`` decorators and from
# ``zarr._deprecations``, rather than in a markdown-exec block because the docs test
# harness runs every ``exec="true"`` block under pytest, where griffe is not installed.
_DEPRECATED_API_MARKER = "<!-- deprecated-api-table -->"
_DECLARED_DEPRECATIONS_MARKER = "<!-- declared-deprecations-table -->"


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    if _DEPRECATED_API_MARKER not in markdown and _DECLARED_DEPRECATIONS_MARKER not in markdown:
        return markdown
    extensions = _griffe_extensions()
    return markdown.replace(_DEPRECATED_API_MARKER, extensions.deprecated_objects_table()).replace(
        _DECLARED_DEPRECATIONS_MARKER, extensions.declared_deprecations_table()
    )


def _griffe_extensions() -> ModuleType:
    """Import the sibling ``mkdocs_griffe_extensions.py``.

    MkDocs loads this hook by file path, so the repository root is not on ``sys.path``
    and the sibling module is loaded the same way.
    """
    path = Path(__file__).with_name("mkdocs_griffe_extensions.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
