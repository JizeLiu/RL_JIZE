"""Rule-based instruction-following reward (IFEval style).

Each sample carries its constraints in ``extra_info["constraints"]`` as a list of dicts
(or a JSON string of that list)::

    [{"type": "min_words", "n": 200}, {"type": "include_keywords", "keywords": ["rain", "sun"]}]

Score = 1.0 if every constraint holds, else 0.0 (``mode="all"``, default), or the satisfied fraction
(``mode="fraction"``). Add a checker with ``@checker("my_type")``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from rewards.registry import register_reward
from rewards.text_utils import strip_thinking

_CHECKERS: dict[str, Callable[..., bool]] = {}


def checker(name: str):
    def deco(fn):
        _CHECKERS[name] = fn
        return fn

    return deco


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


@checker("min_words")
def _min_words(text, n, **_):
    return len(_words(text)) >= n


@checker("max_words")
def _max_words(text, n, **_):
    return len(_words(text)) <= n


@checker("include_keywords")
def _include(text, keywords, case_sensitive=False, **_):
    hay = text if case_sensitive else text.lower()
    return all((k if case_sensitive else k.lower()) in hay for k in keywords)


@checker("exclude_keywords")
def _exclude(text, keywords, case_sensitive=False, **_):
    hay = text if case_sensitive else text.lower()
    return not any((k if case_sensitive else k.lower()) in hay for k in keywords)


@checker("num_paragraphs")
def _paragraphs(text, n, sep="\n\n", **_):
    return len([p for p in text.split(sep) if p.strip()]) == n


@checker("num_bullets")
def _bullets(text, n, **_):
    return len(re.findall(r"^\s*[-*]\s+\S", text, flags=re.MULTILINE)) == n


@checker("starts_with")
def _starts(text, prefix, **_):
    return text.strip().startswith(prefix)


@checker("ends_with")
def _ends(text, suffix, **_):
    return text.strip().endswith(suffix)


@checker("all_lowercase")
def _lower(text, **_):
    return text == text.lower()


@checker("all_uppercase")
def _upper(text, **_):
    return text == text.upper()


@checker("no_commas")
def _no_commas(text, **_):
    return "," not in text


@checker("wrap_in_quotes")
def _quotes(text, **_):
    t = text.strip()
    return len(t) >= 2 and t[0] == '"' and t[-1] == '"'


@checker("title_in_angle_brackets")
def _title(text, **_):
    return re.search(r"<<[^\n]+>>", text) is not None


@checker("json_format")
def _json(text, **_):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        json.loads(t)
        return True
    except ValueError:
        return False


@checker("regex")
def _regex(text, pattern, **_):
    return re.search(pattern, text, flags=re.DOTALL) is not None


def supported_constraints() -> list[str]:
    return sorted(_CHECKERS)


@register_reward("instruction_following")
def instruction_following_reward(solution_str: str, ground_truth=None, extra_info=None, *, mode: str = "all") -> float:
    constraints = (extra_info or {}).get("constraints")
    if isinstance(constraints, str):
        constraints = json.loads(constraints)
    if not constraints:
        raise ValueError("instruction_following reward needs extra_info['constraints']")
    text = strip_thinking(solution_str)
    results = []
    for c in constraints:
        c = dict(c)
        ctype = c.pop("type")
        if ctype not in _CHECKERS:
            raise KeyError(f"unknown constraint type {ctype!r}; supported: {supported_constraints()}")
        results.append(bool(_CHECKERS[ctype](text, **c)))
    if mode == "fraction":
        return sum(results) / len(results)
    return float(all(results))
