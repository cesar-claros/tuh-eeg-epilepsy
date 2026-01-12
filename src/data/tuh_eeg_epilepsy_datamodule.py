#%%
from pathlib import Path

def build_tree(
    root: Path,
    max_depth: int | None = None,
    include_hidden: bool = False,
) -> list[str]:
    """
    Returns a list of lines representing the folder/file structure under `root`.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        # If a file is provided, just show it
        return [root.name]

    def is_hidden(p: Path) -> bool:
        # Cross-platform-ish: dotfiles on Unix, and dot-prefix on Windows too.
        return p.name.startswith(".")

    lines: list[str] = [root.name]

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth >= max_depth:
            return

        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            lines.append(prefix + "└── [Permission denied]")
            return

        if not include_hidden:
            entries = [e for e in entries if not is_hidden(e)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + entry.name)

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(root, "", 0)
    return lines

# %%
epilepsy_flag = 1
root = Path(f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy/').expanduser().resolve()
lines = build_tree(root, max_depth=4)
text = "\n".join(lines)
output = f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy_tree.txt'
out_path = Path(output).expanduser().resolve()
out_path.write_text(text, encoding="utf-8")
# %%
