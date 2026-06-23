"""Print the CHANGELOG.md section body for a single version.

Usage:
    python .github/scripts/changelog_section.py 0.7.1 [path/to/CHANGELOG.md]

Extracts the lines beneath the ``## [<version>] ...`` heading up to (but not
including) the next ``## [`` heading, so the release workflow can use them as
GitHub Release notes. Exits non-zero if the section is missing, so the release
fails loudly rather than publishing empty notes.
"""

import sys
from pathlib import Path


def extract(changelog: str, version: str) -> str:
    """Return the body of the ``## [<version>]`` section, stripped."""
    lines = changelog.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"## [{version}]"):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"No changelog section found for version {version!r}")
    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("## ["):
            break
        body.append(line)
    return "\n".join(body).strip()


def main() -> None:
    version = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("CHANGELOG.md")
    print(extract(path.read_text(encoding="utf-8"), version))


if __name__ == "__main__":
    main()
