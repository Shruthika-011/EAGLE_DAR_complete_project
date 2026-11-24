"""Question understanding for the four PS-04 question types. Standard library only.

Official types -> internal letters
    type 1  A  count buildings of a damage level (image or one quadrant)
    type 2  B  which quadrant has the most damaged (or destroyed) buildings
    type 3  C  is half X more damaged than the opposite half
    type 4  D  not answerable from the images -> "cannot be determined"

Anything that does not match a supported pattern becomes type D with a reason.
The parser never guesses: an ambiguous question (two damage levels, a missing
scope, a half compared with a non-opposite half) is treated as unsupported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from reasoning.spatial import HALVES, QUADRANTS, is_valid_half_pair

# Damage levels a question can name -> the class ids they cover.
LEVELS = {
    "no-damage": frozenset({0}),
    "major-damage": frozenset({1}),        # includes buildings that would be 'minor'
    "destroyed": frozenset({2}),
    "damaged": frozenset({1, 2}),          # major + destroyed
}

# Levels that no longer exist as separate classes: questions naming them are
# reported as "cannot be determined" instead of being silently re-mapped.
_REMOVED_LEVELS = {
    "minor-damage": "'minor damage' is not a separate class (it is merged into major damage)",
    "un-classified": "'un-classified' is not a class",
}

_QUAD_RE = re.compile(r"\b(top|bottom)\s*[- ]\s*(left|right)\s+quadrant\b")
_HALF_RE = re.compile(r"\b(left|right|top|bottom)\s+half\b")


@dataclass(frozen=True)
class Query:
    qtype: str                       # "A" | "B" | "C" | "D"
    level: str | None = None         # key of LEVELS (A, B)
    scope_kind: str | None = None    # "image" | "quadrant" | "half" (A)
    scope_name: str | None = None
    halves: tuple | None = None      # (first, second) (C)
    reason: str | None = None        # why a question is type D


def _normalise(text: str) -> str:
    t = text.lower().strip()
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", t)


def _levels_mentioned(t: str) -> set:
    t = re.sub(r"\([^)]*\)", " ", t)      # "(major or destroyed)" is a definition, not a 2nd level
    found = set()
    if re.search(r"\bundamaged\b|\bno[- ]damage\b|\bunharmed\b", t):
        found.add("no-damage")
    if re.search(r"\bminor[- ]damage\b|\bminor damage\b|\bminorly damaged\b", t):
        found.add("minor-damage")
    if re.search(r"\bmajor[- ]damage\b|\bmajor damage\b|\bmajorly damaged\b", t):
        found.add("major-damage")
    if re.search(r"\bdestroyed\b", t):
        found.add("destroyed")
    if re.search(r"\bdamaged\b", t):        # \b: "undamaged" does not match
        found.add("damaged")
    if re.search(r"\bun-?classified\b", t):
        found.add("un-classified")
    return found


def _unsupported(reason: str) -> Query:
    return Query("D", reason=reason)


def parse_question(text: str) -> Query:
    t = _normalise(text)
    if not t:
        return _unsupported("empty question")

    quad = _QUAD_RE.search(t)
    halves = _HALF_RE.findall(t)
    levels = _levels_mentioned(t)

    removed = levels & set(_REMOVED_LEVELS)
    if removed:
        return _unsupported(_REMOVED_LEVELS[sorted(removed)[0]])

    # ---- type B: which quadrant ... most damaged / destroyed ---------------
    if re.search(r"\bwhich quadrant\b", t):
        if not re.search(r"\b(most|largest|highest|greatest)\b", t):
            return _unsupported("quadrant question without a most/largest comparison")
        if len(levels) != 1 or not (levels & {"damaged", "destroyed"}):
            return _unsupported("quadrant question must name 'damaged' or 'destroyed' buildings")
        return Query("B", level=next(iter(levels)))

    # ---- type C: compare two halves ----------------------------------------
    if len(halves) >= 2 or (halves and re.search(r"\bhalf\b.*\bthan\b", t)):
        if len(halves) != 2:
            return _unsupported("half comparison must name exactly two halves")
        if not re.search(r"\bmore\b", t):
            return _unsupported("only 'more damaged than' comparisons are supported")
        if levels - {"damaged"}:
            return _unsupported("half comparison is defined for damaged buildings only")
        a, b = halves
        if not is_valid_half_pair(a, b):
            return _unsupported(f"'{a}' and '{b}' are not opposite halves")
        return Query("C", halves=(a, b))

    # ---- type A: count of a damage level ------------------------------------
    if re.search(r"\b(how many|count|number of)\b", t):
        if len(levels) != 1:
            return _unsupported("count question must name exactly one damage level")
        level = next(iter(levels))
        if quad:
            return Query("A", level=level, scope_kind="quadrant", scope_name=f"{quad.group(1)}-{quad.group(2)}")
        if len(halves) == 1:
            return Query("A", level=level, scope_kind="half", scope_name=halves[0])
        if re.search(r"\bquadrant\b", t):
            return _unsupported("unrecognised quadrant name")
        if re.search(r"\bimage\b", t):
            return Query("A", level=level, scope_kind="image", scope_name=None)
        return _unsupported("count question has no scope (image, quadrant or half)")

    return _unsupported("question is outside the supported damage-count/comparison types")


__all__ = ["Query", "parse_question", "LEVELS", "QUADRANTS", "HALVES"]
