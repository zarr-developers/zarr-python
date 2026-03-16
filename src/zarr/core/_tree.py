import sys
from collections import deque
from collections.abc import Sequence
from html import escape as html_escape
from typing import Any

from zarr.core.group import AsyncGroup


class TreeRepr:
    """
    A simple object with a tree-like repr for the Zarr Group.

    Note that this object and it's implementation isn't considered part
    of Zarr's public API.
    """

    def __init__(self, text: str, html: str, truncated: str = "") -> None:
        self._text = text
        self._html = html
        self._truncated = truncated

    def __repr__(self) -> str:
        if self._truncated:
            return self._truncated + self._text
        return self._text

    def _repr_mimebundle_(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        text = self._truncated + self._text if self._truncated else self._text
        # For jupyter support.
        html_body = self._truncated + self._html if self._truncated else self._html
        html = (
            '<pre style="white-space:pre;overflow-x:auto;line-height:normal;'
            "font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">"
            f"{html_body}</pre>\n"
        )
        return {"text/plain": text, "text/html": html}


async def group_tree_async(
    group: AsyncGroup,
    max_depth: int | None = None,
    *,
    max_nodes: int = 500,
    plain: bool = False,
) -> TreeRepr:
    members: list[tuple[str, Any]] = []
    truncated = False
    async for item in group.members(max_depth=max_depth):
        if len(members) == max_nodes:
            truncated = True
            break
        members.append(item)
    members.sort(key=lambda key_node: key_node[0])

    # Set up styling tokens: ANSI bold for terminals, HTML <b> for Jupyter,
    # or empty strings when plain=True (useful for LLMs, logging, files).
    if plain:
        ansi_open = ansi_close = html_open = html_close = ""
    else:
        # Avoid emitting ANSI escape codes when output is piped or in CI.
        use_ansi = sys.stdout.isatty()
        ansi_open = "\x1b[1m" if use_ansi else ""
        ansi_close = "\x1b[0m" if use_ansi else ""
        html_open = "<b>"
        html_close = "</b>"

    # Group members by parent key so we can render the tree level by level.
    nodes: dict[str, list[tuple[str, Any]]] = {}
    for key, node in members:
        if key.count("/") == 0:
            parent_key = ""
        else:
            parent_key = key.rsplit("/", 1)[0]
        nodes.setdefault(parent_key, []).append((key, node))

    # Render the tree iteratively (not recursively) to avoid hitting
    # Python's recursion limit on deeply nested hierarchies.
    # Each stack frame is (prefix_string, remaining_children_at_this_level).
    text_lines = [f"{ansi_open}{group.name}{ansi_close}"]
    html_lines = [f"{html_open}{html_escape(group.name)}{html_close}"]
    stack = [("", deque(nodes.get("", [])))]
    while stack:
        prefix, remaining = stack[-1]
        if not remaining:
            stack.pop()
            continue
        key, node = remaining.popleft()
        name = key.rsplit("/")[-1]
        escaped_name = html_escape(name)
        # if we popped the last item then remaining will
        # now be empty - that's how we got past the if not remaining
        # above, but this can still be true.
        is_last = not remaining
        connector = "└── " if is_last else "├── "
        if isinstance(node, AsyncGroup):
            text_lines.append(f"{prefix}{connector}{ansi_open}{name}{ansi_close}")
            html_lines.append(f"{prefix}{connector}{html_open}{escaped_name}{html_close}")
        else:
            text_lines.append(
                f"{prefix}{connector}{ansi_open}{name}{ansi_close} {node.shape} {node.dtype}"
            )
            html_lines.append(
                f"{prefix}{connector}{html_open}{escaped_name}{html_close}"
                f" {html_escape(str(node.shape))} {html_escape(str(node.dtype))}"
            )
        # Descend into children with an accumulated prefix:
        # Example showing how prefix accumulates:
        #   /
        #   ├── a              prefix = ""
        #   │   ├── b          prefix = "" + "│   "
        #   │   │   └── x      prefix = "" + "│   " + "│   "
        #   │   └── c          prefix = "" + "│   "
        #   └── d              prefix = ""
        #       └── e          prefix = "" + "    "
        if children := nodes.get(key, []):
            if is_last:
                child_prefix = prefix + "    "
            else:
                child_prefix = prefix + "│   "
            stack.append((child_prefix, deque(children)))
    text = "\n".join(text_lines) + "\n"
    html = "\n".join(html_lines) + "\n"
    note = (
        f"Truncated at max_nodes={max_nodes}, some nodes and their children may be missing\n"
        if truncated
        else ""
    )
    return TreeRepr(text, html, truncated=note)
