"""
Check changelog entries have the correct filename structure.
"""

import sys
from pathlib import Path

VALID_CHANGELOG_TYPES = ["feature", "bugfix", "doc", "removal", "misc"]
CHANGELOG_DIRECTORY = (Path(__file__).parent.parent / "changes").resolve()


def is_int(s: str) -> bool:
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


if __name__ == "__main__":
    print(f"Looking for changelog entries in {CHANGELOG_DIRECTORY}")
    entries = CHANGELOG_DIRECTORY.glob("*")
    entries = [e for e in entries if e.name not in [".gitignore", "README.md"]]
    print(f"Found {len(entries)} entries")
    print()

    bad_suffix = [e for e in entries if e.suffix != ".rst"]
    bad_issue_no = [e for e in entries if not is_int(e.name.split(".")[0])]
    bad_type = [e for e in entries if e.name.split(".")[1] not in VALID_CHANGELOG_TYPES]

    if len(bad_suffix) or len(bad_issue_no) or len(bad_type):
        if len(bad_suffix):
            print("Changelog entries without .rst suffix")
            print("-------------------------------------")
            print("\n".join([p.name for p in bad_suffix]))
            print()
        if len(bad_issue_no):
            print("Changelog entries without integer issue number")
            print("----------------------------------------------")
            print("\n".join([p.name for p in bad_issue_no]))
            print()
        if len(bad_type):
            print("Changelog entries without valid type")
            print("------------------------------------")
            print("\n".join([p.name for p in bad_type]))
            print(f"Valid types are: {VALID_CHANGELOG_TYPES}")
            print()
        sys.exit(1)

    sys.exit(0)
