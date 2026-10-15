"""Check that every internal link in a built MkDocs site resolves.

Usage:
    python .github/scripts/check_site_links.py [path/to/site] [base-path]

``mkdocs build --strict`` validates links in Markdown pages, but not links
inside notebooks rendered by mkdocs-jupyter, which are copied into the HTML
as written. This walks every HTML page of the built site and checks that each
relative ``href`` points to an existing file or directory (with an
``index.html``). External links, in-page anchors and ``mailto:`` links are
skipped. Absolute links (``/...``) are resolved against ``base-path``, the URL
path the site is served under (``/ode_filters/`` for GitHub Pages; default
``/``). Exits non-zero, listing every broken link, if any is found.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def _is_internal(href: str) -> bool:
    parts = urlsplit(href)
    return not (parts.scheme or parts.netloc or href.startswith("#"))


def _resolves(page: Path, href: str, site: Path, base_path: str) -> bool:
    path = unquote(urlsplit(href).path)
    if not path:
        return True
    if path.startswith("/"):
        if not path.startswith(base_path):
            return False
        target = site / path[len(base_path) :]
    else:
        target = page.parent / path
    target = target.resolve()
    if target.is_dir():
        return (target / "index.html").is_file()
    return target.is_file()


def broken_links(site: Path, base_path: str = "/") -> list[tuple[Path, str]]:
    """Return ``(page, href)`` for every internal link that does not resolve."""
    broken = []
    for page in sorted(site.rglob("*.html")):
        collector = _HrefCollector()
        collector.feed(page.read_text(encoding="utf-8"))
        for href in collector.hrefs:
            if _is_internal(href) and not _resolves(page, href, site, base_path):
                broken.append((page.relative_to(site), href))
    return broken


def main() -> None:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    base_path = "/" + (sys.argv[2] if len(sys.argv) > 2 else "").strip("/") + "/"
    base_path = base_path.replace("//", "/")
    if not site.is_dir():
        raise SystemExit(f"No built site at {site}; run `mkdocs build` first.")
    broken = broken_links(site, base_path)
    for page, href in broken:
        print(f"{page}: broken link {href!r}")
    if broken:
        raise SystemExit(f"{len(broken)} broken internal link(s).")
    print("All internal links resolve.")


if __name__ == "__main__":
    main()
