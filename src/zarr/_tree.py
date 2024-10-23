import io

from zarr.core.group import AsyncGroup

try:
    import rich
    import rich.console
    import rich.tree
except ImportError as e:
    raise ImportError("'rich' is required for Group.tree") from e


class TreeRepr:
    def __init__(self, tree: rich.tree.Tree) -> None:
        self.tree = tree

    def __repr__(self) -> str:
        console = rich.console.Console(file=io.StringIO())
        console.print(self.tree)
        return str(console.file.getvalue())


async def group_tree_async(group: AsyncGroup, max_depth: int | None = None) -> TreeRepr:
    tree = rich.tree.Tree(label=f"[b]{group.name}[/b]")
    nodes = {"": tree}
    members = sorted([x async for x in group.members(max_depth=max_depth)])

    for key, node in members:
        if key.count("/") == 0:
            parent_key = ""
        else:
            parent_key = key.rsplit("/", 1)[0]
        parent = nodes[parent_key]

        # We want what the spec calls the node "name", the part excluding all leading
        # /'s and path segments. But node.name includes all that, so we build it here.
        name = key.rsplit("/")[-1]
        if isinstance(node, AsyncGroup):
            label = f"[b]{name}[/b]"
        else:
            label = f"[b]{name}[/b] {node.shape} {node.dtype}"
        nodes[key] = parent.add(label)

    return TreeRepr(tree)
