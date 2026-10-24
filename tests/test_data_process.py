import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from data_process import difficulty, to_verl, vote_gt
from data_process.io_utils import iter_jsonl, read_records, write_jsonl


def run(c):
    return asyncio.run(c)


def fake_sampler(answers_by_question: dict[str, list[str]]):
    """Returns canned responses per question text; cycles if n > len."""

    async def sample(messages, n, temperature, max_tokens):
        q = messages[-1]["content"].split("\n\n")[0]
        ans = answers_by_question[q]
        return [rf"reasoning \boxed{{{ans[i % len(ans)]}}}" for i in range(n)]

    return sample


async def fake_verify(record, response):
    return f"\\boxed{{{record['ground_truth']}}}" in response


RECORDS = [
    {"id": "a", "question": "Q-good", "ground_truth": "1", "data_source": "math/t"},
    {"id": "b", "question": "Q-wronglabel", "ground_truth": "5", "data_source": "math/t"},
    {"id": "c", "question": "Q-split", "ground_truth": "3", "data_source": "math/t"},
]


def _args(tmp_path, **kw):
    import argparse

    base = dict(
        input=str(tmp_path / "in.jsonl"), output=str(tmp_path / "out.jsonl"), field_map=None, n=2, temperature=1.0,
        max_tokens=16, suffix="", record_concurrency=4, min_agree=0.5, relabel_ratio=None,
        drop_all_correct=True, drop_all_wrong=False, min_pass_rate=0.0, max_pass_rate=1.0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_vote_keeps_supported_labels_and_reports_consensus(tmp_path):
    write_jsonl(tmp_path / "in.jsonl", RECORDS)
    m1 = fake_sampler({"Q-good": ["1"], "Q-wronglabel": ["7"], "Q-split": ["3"]})
    m2 = fake_sampler({"Q-good": ["1"], "Q-wronglabel": ["7"], "Q-split": ["4"]})
    m3 = fake_sampler({"Q-good": ["1"], "Q-wronglabel": ["7"], "Q-split": ["3"]})
    args = _args(tmp_path, min_agree=0.6)
    counts = run(vote_gt.run(args, {"m1": m1, "m2": m2, "m3": m3}, fake_verify))
    assert counts == {"keep": 2, "reject": 1}
    kept = {r["id"] for r in iter_jsonl(args.output)}
    assert kept == {"a", "c"}  # c: 2 of 3 models support
    rejected = list(iter_jsonl(args.output + ".rejected.jsonl"))
    assert rejected[0]["id"] == "b" and rejected[0]["vote"]["consensus_answer"] == "7"
    assert rejected[0]["vote"]["consensus_ratio"] == 1.0


def test_vote_relabel(tmp_path):
    write_jsonl(tmp_path / "in.jsonl", RECORDS[1:2])
    s = fake_sampler({"Q-wronglabel": ["7"]})
    args = _args(tmp_path, relabel_ratio=0.8)
    counts = run(vote_gt.run(args, {"m1": s, "m2": s}, fake_verify))
    assert counts == {"relabel": 1}
    rec = next(iter_jsonl(args.output))
    assert rec["ground_truth"] == "7" and rec["original_ground_truth"] == "5"


def test_vote_is_resumable(tmp_path):
    write_jsonl(tmp_path / "in.jsonl", RECORDS)
    calls = []

    async def counting(messages, n, t, m):
        calls.append(messages[-1]["content"])
        return [r"\boxed{1}"] * n

    args = _args(tmp_path)
    run(vote_gt.run(args, {"m": counting}, fake_verify))
    first = len(calls)
    run(vote_gt.run(args, {"m": counting}, fake_verify))
    assert len(calls) == first  # nothing re-sampled


def test_bucket_and_select():
    assert difficulty.bucket(0, 5) == "all_wrong"
    assert difficulty.bucket(5, 5) == "all_correct"
    assert difficulty.bucket(1, 5) == "hard"
    assert difficulty.bucket(3, 5) == "medium"
    assert difficulty.bucket(4, 5) == "easy"


def test_difficulty_drops_all_correct_keeps_all_wrong(tmp_path):
    recs = [
        {"id": "easy", "question": "E", "ground_truth": "1", "data_source": "math/t"},
        {"id": "hard0", "question": "H0", "ground_truth": "1", "data_source": "math/t"},
        {"id": "mid", "question": "M", "ground_truth": "1", "data_source": "math/t"},
    ]
    write_jsonl(tmp_path / "in.jsonl", recs)
    sampler = fake_sampler({"E": ["1"], "H0": ["2"], "M": ["1", "2", "2", "1", "2"]})
    args = _args(tmp_path, n=5)
    counts = run(difficulty.run(args, sampler, fake_verify))
    kept = {r["id"]: r for r in iter_jsonl(args.output)}
    assert set(kept) == {"hard0", "mid"}
    assert kept["hard0"]["difficulty"] == "all_wrong" and kept["mid"]["n_correct"] == 2
    assert counts["all_correct"] == 1 and counts["kept"] == 2

    args2 = _args(tmp_path, n=5, output=str(tmp_path / "out2.jsonl"), drop_all_wrong=True)
    run(difficulty.run(args2, sampler, fake_verify))
    assert {r["id"] for r in iter_jsonl(args2.output)} == {"mid"}


def test_to_verl_schema_roundtrip(tmp_path):
    recs = [
        {"id": "1", "question": "What is 2+2?", "ground_truth": "4", "data_source": "math/t", "solution": "2+2=4",
         "pass_rate": 0.4, "difficulty": "hard"},
        {"id": "2", "question": "Write 3 words", "data_source": "ifeval/t",
         "constraints": [{"type": "min_words", "n": 3}, {"type": "no_commas"}]},
    ]
    write_jsonl(tmp_path / "t.jsonl", recs)
    out = tmp_path / "train.parquet"
    to_verl.main(["--input", str(tmp_path / "t.jsonl"), "--output", str(out)])
    df = pd.read_parquet(out)
    row = df.iloc[0]
    assert row["prompt"][0]["role"] == "user" and "\\boxed" in row["prompt"][0]["content"]
    assert row["reward_model"]["ground_truth"] == "4"
    assert row["extra_info"]["solution"] == "2+2=4" and row["extra_info"]["pass_rate"] == 0.4
    cons = json.loads(df.iloc[1]["extra_info"]["constraints"])
    assert cons[1] == {"type": "no_commas"}  # no null-padded keys
    assert json.loads(Path(str(out) + ".stats.json").read_text())["rows"] == 2


def test_to_verl_requires_label_and_source():
    with pytest.raises(ValueError):
        to_verl.to_verl_row({"id": "x", "question": "q", "ground_truth": "1"}, 0, "train", "", None)
    with pytest.raises(ValueError):
        to_verl.to_verl_row({"id": "x", "question": "q", "data_source": "math/t"}, 0, "train", "", None)


def test_field_map(tmp_path):
    write_jsonl(tmp_path / "r.jsonl", [{"problem": "p", "answer": "a"}])
    recs = read_records(tmp_path / "r.jsonl", {"question": "problem", "ground_truth": "answer"})
    assert recs[0]["question"] == "p" and recs[0]["ground_truth"] == "a" and recs[0]["id"] == "0"
