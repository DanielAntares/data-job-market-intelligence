"""Compile the taxonomy into matchers and extract skills from text.

Matching rules:
  * Literal aliases are wrapped in non-alphanumeric boundaries, so "Java" does
    NOT fire on "JavaScript" and "SQL" does NOT fire inside "MySQL".
  * Special characters in names (``C++``, ``C#``, ``Node.js``) are matched
    literally; the boundaries use ``[A-Za-z0-9]`` only, so a trailing period in
    prose ("...Python.") still matches.
  * Ambiguous names (``R``, ``Go``, ``Excel``) carry their own context-aware
    ``pattern`` instead of plain aliases.
"""
from __future__ import annotations

import re

from .taxonomy import TAXONOMY, Skill

# A boundary that treats only letters/digits as "inside a word". Crucially it
# excludes '.', '+', '#', so prose punctuation around a skill doesn't block a
# match, while still preventing partial-word hits.
_BOUND_L = r"(?<![A-Za-z0-9])"
_BOUND_R = r"(?![A-Za-z0-9])"


def _compile(skill: Skill) -> re.Pattern:
    if skill.pattern is not None:
        return re.compile(skill.pattern)
    # Longest alias first so the regex prefers the most specific surface form.
    alts = sorted((re.escape(a) for a in skill.aliases), key=len, reverse=True)
    body = "|".join(alts)
    return re.compile(_BOUND_L + "(?:" + body + ")" + _BOUND_R, re.IGNORECASE)


class SkillExtractor:
    def __init__(self, taxonomy: list[Skill] = TAXONOMY):
        self._matchers = [(s.name, s.category, _compile(s)) for s in taxonomy]

    def extract(self, text: str | None) -> list[str]:
        """Return the sorted, de-duplicated canonical skill names found in text."""
        return [name for name, _ in self.extract_with_categories(text)]

    def extract_with_categories(self, text: str | None) -> list[tuple[str, str]]:
        """Return sorted unique ``(skill, category)`` pairs found in text."""
        if not text:
            return []
        found = {
            (name, category)
            for name, category, rx in self._matchers
            if rx.search(text)
        }
        return sorted(found)


# Module-level singleton so callers don't recompile ~70 regexes per posting.
default_extractor = SkillExtractor()
