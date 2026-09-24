"""Controlled local Markdown documentation updates.

The memory engine may prepare documentation updates, but it never writes to a
repository file implicitly. A caller must explicitly set ``approved=True``.
Updates are section-scoped and use stable markers, so repeated evolution runs
replace the same section instead of endlessly appending duplicate text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_START = "<!-- minimemory:section:{name}:start -->"
_END = "<!-- minimemory:section:{name}:end -->"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


def _markers(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError("section name must contain only letters, numbers, '_' or '-' and be <= 80 characters")
    return _START.format(name=name), _END.format(name=name)


def read_documentation(path: str | Path = "documentation.md") -> str:
    """Read a Markdown documentation file."""
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def update_documentation(
    path: str | Path,
    section: str,
    content: str,
    *,
    approved: bool = False,
    heading: Optional[str] = None,
) -> dict:
    """Create or replace one marked Markdown section.

    Nothing is written unless ``approved=True``. This is intentional: learned
    content should not silently modify a source repository or its docs.
    """
    if not approved:
        return {"updated": False, "reason": "approval_required", "path": str(path), "section": section}
    if not isinstance(content, str) or not content.strip():
        raise ValueError("documentation content must be non-empty")

    start, end = _markers(section)
    target = Path(path)
    existing = read_documentation(target)
    block = f"{start}\n"
    if heading:
        block += f"### {heading.strip()}\n\n"
    block += content.strip() + "\n"
    block += f"{end}\n"

    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(existing):
        updated = pattern.sub(block.rstrip("\n"), existing, count=1)
        action = "replaced"
    else:
        separator = "\n" if existing.endswith("\n") or not existing else "\n\n"
        updated = existing + separator + block
        action = "created"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return {"updated": True, "action": action, "path": str(target), "section": section}
