import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "verl"))

from rewards import dispatch, llm_judge
from rewards.instruction_following import instruction_following_reward, supported_constraints
from rewards.math_reward import math_verify_reward
from rewards.text_utils import extract_boxed, normalize_math, strip_thinking


def run(coro):
    return asyncio.run(coro)


def test_extract_and_normalize():
    assert extract_boxed(r"x \boxed{\dfrac{1}{2}}") == r"\dfrac{1}{2}"
    assert normalize_math(r"\dfrac{1}{2}") == normalize_math(r"\frac{1}{2}")
    assert normalize_math("12 ") == "12"
    assert strip_thinking("<think>hmm</think>ans") == "ans"


def test_math_reward_basic():
    assert math_verify_reward(r"so \boxed{42}", "42") == 1.0
    assert math_verify_reward(r"so \boxed{41}", "42") == 0.0
    assert math_verify_reward("no box 42", "42") == 0.0
    assert math_verify_reward(r"<think>\boxed{1}</think>\boxed{2}", "2") == 1.0


def test_instruction_following_modes():
    cons = [{"type": "min_words", "n": 3}, {"type": "no_commas"}, {"type": "include_keywords", "keywords": ["Sun"]}]
    ok = "the sun is bright"
    bad = "sun, shine"
    assert instruction_following_reward(ok, None, {"constraints": cons}) == 1.0
    assert instruction_following_reward(bad, None, {"constraints": cons}) == 0.0
    assert instruction_following_reward(bad, None, {"constraints": cons}, mode="fraction") == pytest.approx(1 / 3)
    assert instruction_following_reward(ok, None, {"constraints": json.dumps(cons)}) == 1.0


def test_instruction_following_errors():
    with pytest.raises(ValueError):
        instruction_following_reward("x", None, {})
    with pytest.raises(KeyError):
        instruction_following_reward("x", None, {"constraints": [{"type": "nope"}]})
    assert {"json_format", "num_bullets", "regex"} <= set(supported_constraints())


def test_json_and_bullets():
    assert instruction_following_reward('```json\n{"a":1}\n```', None, {"constraints": [{"type": "json_format"}]}) == 1.0
    assert instruction_following_reward("- a\n- b", None, {"constraints": [{"type": "num_bullets", "n": 2}]}) == 1.0


def test_parse_score():
    assert llm_judge.parse_score("fine.\nScore: 8", 10) == 0.8
    assert llm_judge.parse_score("Score: 99", 10) == 1.0
    assert llm_judge.parse_score("none", 10) is None


def test_llm_judge_with_fake_backend_and_retry():
    calls = []

    async def fake(prompt):
        calls.append(prompt)
        return "garbage" if len(calls) == 1 else "ok\nScore: 7"

    llm_judge.set_chat_fn(fake)
    try:
        score = run(llm_judge.llm_judge_reward("resp", "ref", {"question": "Q?"}, use_reference=True, retries=2))
    finally:
        llm_judge.set_chat_fn(None)
    assert score == pytest.approx(0.7)
    assert len(calls) == 2 and "Q?" in calls[0] and "[Reference answer]" in calls[0]


def test_llm_judge_failure_returns_fail_score():
    async def boom(prompt):
        raise RuntimeError("down")

    llm_judge.set_chat_fn(boom)
    try:
        score = run(llm_judge.llm_judge_reward("r", None, {"question": "q"}, retries=0, fail_score=-1.0))
    finally:
        llm_judge.set_chat_fn(None)
    assert score == -1.0


def test_dispatch_routes_and_stable_keys():
    dispatch.reset_cache()
    a = run(dispatch.compute_score("math/gsm", r"\boxed{3}", "3", {}))
    b = run(dispatch.compute_score("ifeval/x", "hi there", None, {"constraints": [{"type": "min_words", "n": 5}]}))
    assert a["score"] == 1.0 and a["acc"] == 1.0
    assert b["score"] == 0.0 and b["acc"] == 0.0
    assert set(a) == set(b) and "reward/llm_judge" in a


def test_dispatch_weighted_mix(monkeypatch, tmp_path):
    routes = tmp_path / "r.yaml"
    routes.write_text(
        "routes:\n  - match: 'mix/*'\n    rewards:\n"
        "      - {name: math_verify, weight: 0.5}\n"
        "      - {name: instruction_following, weight: 0.5, args: {mode: fraction}}\n"
    )
    monkeypatch.setenv("RLX_REWARD_ROUTES", str(routes))
    dispatch.reset_cache()
    try:
        out = run(dispatch.compute_score("mix/a", r"\boxed{1}", "1", {"constraints": [{"type": "min_words", "n": 50}]}))
        assert out["score"] == pytest.approx(0.5)
        assert out["reward/math_verify"] == 1.0 and out["reward/instruction_following"] == 0.0
    finally:
        dispatch.reset_cache()


def test_dispatch_unknown_reward_fails_fast(monkeypatch, tmp_path):
    routes = tmp_path / "r.yaml"
    routes.write_text("routes:\n  - match: '*'\n    rewards:\n      - {name: typo_reward}\n")
    monkeypatch.setenv("RLX_REWARD_ROUTES", str(routes))
    dispatch.reset_cache()
    try:
        with pytest.raises(KeyError):
            run(dispatch.compute_score("x", "y", "z", {}))
    finally:
        dispatch.reset_cache()


def test_entry_file_loads_through_verl_loader():
    verl_utils = pytest.importorskip("verl.utils.import_utils")
    import inspect
    from pathlib import Path

    fn = verl_utils.load_extern_object(
        module_path=str(Path(__file__).resolve().parents[1] / "rewards" / "entry.py"), object_name="compute_score"
    )
    assert inspect.iscoroutinefunction(fn)
    out = run(fn(data_source="math/x", solution_str=r"\boxed{5}", ground_truth="5", extra_info={}))
    assert out["score"] == 1.0
