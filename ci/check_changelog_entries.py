"""
Check changelog entries have the correct filename structure.

Usage:
    python check_changelog_entries.py [DIRECTORY]

DIRECTORY defaults to the repo-root `changes/`.
"""

import sys
from pathlib import Path

VALID_CHANGELOG_TYPES = ["feature", "bugfix", "doc", "removal", "misc"]
REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_DIRECTORY = REPO_ROOT / "changes"


def is_int(s: str) -> bool:
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def check(directory: Path) -> int:
    print(f"Looking for changelog entries in {directory}")
    entries = list(directory.glob("*"))
    entries = [e for e in entries if e.name not in [".gitignore", "README.md"]]
    print(f"Found {len(entries)} entries")
    print()

    bad_suffix = [e for e in entries if e.suffix != ".md"]
    bad_issue_no = [e for e in entries if not is_int(e.name.split(".")[0])]
    # Only flag bad_type for files that have already passed the prior two
    # checks; otherwise `e.name.split(".")[1]` may raise IndexError on a
    # malformed name like `notes.md`.
    bad_type = [
        e
        for e in entries
        if e.suffix == ".md"
        and is_int(e.name.split(".")[0])
        and e.name.split(".")[1] not in VALID_CHANGELOG_TYPES
    ]

    if bad_suffix or bad_issue_no or bad_type:
        if bad_suffix:
            print("Changelog entries without .md suffix")
            print("-------------------------------------")
            print("\n".join(p.name for p in bad_suffix))
            print()
        if bad_issue_no:
            print("Changelog entries without integer issue number")
            print("----------------------------------------------")
            print("\n".join(p.name for p in bad_issue_no))
            print()
        if bad_type:
            print("Changelog entries without valid type")
            print("------------------------------------")
            print("\n".join(p.name for p in bad_type))
            print(f"Valid types are: {VALID_CHANGELOG_TYPES}")
            print()
        return 1
    return 0


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DIRECTORY
    sys.exit(check(directory))
