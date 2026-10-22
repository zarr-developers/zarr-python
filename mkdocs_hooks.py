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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


# The metadata compatibility page holds this marker where the list of
# non-conformant and unregistered metadata goes. It is generated at build time from
# ``zarr.core.metadata.ledger``, which ``tests/test_ledger.py`` checks against the
# source, so the page lists what the code reads and writes.
_LEDGER_MARKER = "<!-- metadata-ledger -->"


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    if _LEDGER_MARKER not in markdown:
        return markdown
    return markdown.replace(_LEDGER_MARKER, ledger_markdown())


def ledger_markdown() -> str:
    """Render ``zarr.core.metadata.ledger.LEDGER`` as Markdown: the metadata the current
    zarr still writes, then the metadata it only reads, then which other implementations
    were tested."""
    from zarr.core.metadata.ledger import LEDGER, READERS_TESTED

    sections = []
    for heading, still_written in (("Still written", True), ("No longer written", False)):
        items = [
            f"- **{entry.title}** ({entry.conformance})\n"
            f"    - Written by: {entry.written_by}\n"
            f"    - Current behavior: {entry.reading}\n"
            f"    - Other implementations: {entry.other_readers}"
            for entry in LEDGER.values()
            if entry.still_written is still_written
        ]
        sections.append(f"## {heading}\n\n" + "\n".join(items))
    sections.append(f"## Other implementations tested\n\n{READERS_TESTED}.")
    return "\n\n".join(sections)
