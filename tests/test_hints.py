import pytest

from rlx.selfdistill.hints import extract_boxed, plan_hints, render_hint_messages


def test_extract_boxed_nested_and_last():
    assert extract_boxed(r"so \boxed{1} then \boxed{\frac{a}{b}} done") == r"\frac{a}{b}"
    assert extract_boxed("no answer") == ""
    assert extract_boxed(r"\boxed{unclosed") == ""


def test_render_replaces_last_user_turn_only():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is 1+1?"},
    ]
    out = render_hint_messages(msgs, solution="1+1=2", answer="2")
    assert out[0] == msgs[0]
    assert "What is 1+1?" in out[1]["content"]
    assert "Reference Solution" in out[1]["content"] and "1+1=2" in out[1]["content"]
    assert "Correct final answer: 2" in out[1]["content"]
    assert msgs[1]["content"] == "What is 1+1?"  # input not mutated


def test_render_without_solution_omits_reference_block():
    out = render_hint_messages([{"role": "user", "content": "Q"}], solution="", answer="42")
    assert "Reference Solution" not in out[0]["content"]
    assert "Correct final answer: 42" in out[0]["content"]


def test_positive_and_negative_templates_are_identical_in_shape():
    msgs = [{"role": "user", "content": "Q"}]
    pos = render_hint_messages(msgs, solution="good", answer="1")[0]["content"]
    neg = render_hint_messages(msgs, solution="bad", answer="2")[0]["content"]
    assert pos.replace("good", "X").replace("1", "Y") == neg.replace("bad", "X").replace("2", "Y")


def test_render_requires_user_message():
    with pytest.raises(ValueError):
        render_hint_messages([{"role": "system", "content": "s"}], solution="", answer="")


def test_plan_hints_excludes_self_and_respects_outcomes():
    uids = ["a"] * 4 + ["b"] * 2
    rewards = [1, 0, 0, 1, 1, 1]
    plans = plan_hints(uids, rewards, positive_source="rollout", n_neg=2, seed=1)
    for i in range(4):
        pos = plans[i].positive
        assert pos is not None and pos.row != i and rewards[pos.row] == 1 and uids[pos.row] == "a"
        for neg in plans[i].negatives:
            assert neg.row != i and rewards[neg.row] == 0 and uids[neg.row] == "a"
    # group b is all-correct: no negatives, and the positive sibling is the other row
    assert plans[4].negatives == [] and plans[4].positive.row == 5


def test_plan_hints_rollout_without_correct_sibling():
    plans = plan_hints(["a", "a"], [0, 0], positive_source="rollout", n_neg=1)
    assert plans[0].positive is None and plans[1].positive is None
    assert plans[0].negatives[0].row == 1


def test_plan_hints_gt_and_fewer_negatives_than_k():
    plans = plan_hints(["a", "a", "a"], [1, 0, 1], positive_source="gt", n_neg=4)
    assert all(p.positive.kind == "gt" for p in plans)
    assert [len(p.negatives) for p in plans] == [1, 0, 1]


def test_plan_hints_deterministic_per_seed():
    uids = ["a"] * 8
    rewards = [1, 0, 0, 0, 0, 1, 0, 0]
    a = plan_hints(uids, rewards, n_neg=2, seed=7)
    b = plan_hints(uids, rewards, n_neg=2, seed=7)
    assert a == b
