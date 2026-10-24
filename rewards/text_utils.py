"""Small text helpers shared by rewards."""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` (balanced braces); empty string if absent."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    i = start + len("\\boxed{")
    depth, begin = 1, i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[begin : i - 1].strip() if depth == 0 else ""


def strip_thinking(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block so rewards judge the final answer only."""
    return re.sub(r"^\s*<think>.*?</think>", "", text, flags=re.DOTALL)


_NORM_SUBS = [
    (r"\\left|\\right|\\!|\\,|\\;|\\ ", ""),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\^\{\\circ\}|\^\\circ|°", ""),
    (r"\\text\{[^}]*\}", ""),
    (r"\\\$|\$", ""),
    (r"\s+", ""),
]


def normalize_math(s: str) -> str:
    s = str(s).strip().rstrip(".")
    for pat, rep in _NORM_SUBS:
        s = re.sub(pat, rep, s)
    return s.lower()
