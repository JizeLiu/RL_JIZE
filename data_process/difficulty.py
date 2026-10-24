"""Difficulty filtering by sampling each prompt ``--n`` times with one model.

Per record we store ``pass_rate`` / ``n_correct`` and a ``difficulty`` bucket. Default selection:

* all answers correct  -> dropped (zero GRPO advantage, wastes rollout compute)
* all answers wrong    -> kept   (can still be learned from, e.g. by RLSD/RLCSD with reference hints)
* anything in between  -> kept

``--drop-all-wrong`` also removes the unsolved prompts; ``--max-pass-rate`` / ``--min-pass-rate`` give
finer control.

    python -m data_process.difficulty --input voted.jsonl --output train.jsonl \
        --endpoint policy=http://n1:8000/v1 --n 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter

from data_process.io_utils import JsonlAppender, iter_jsonl, parse_field_map, read_records, write_jsonl
from data_process.llm_client import SampleFn, make_openai_sampler, parse_endpoints
from data_process.prompts import DEFAULT_SUFFIX, build_messages
from data_process.verify import VerifyFn, reward_router_verify

logger = logging.getLogger(__name__)


def bucket(n_correct: int, n: int) -> str:
    if n_correct == 0:
        return "all_wrong"
    if n_correct == n:
        return "all_correct"
    rate = n_correct / n
    return "hard" if rate <= 0.4 else ("medium" if rate <= 0.7 else "easy")


def select(record: dict, *, drop_all_correct: bool, drop_all_wrong: bool, min_pass: float, max_pass: float) -> bool:
    d = record["difficulty"]
    if d == "all_correct" and drop_all_correct:
        return False
    if d == "all_wrong" and drop_all_wrong:
        return False
    return min_pass <= record["pass_rate"] <= max_pass


async def measure(record: dict, sampler: SampleFn, verify: VerifyFn, *, n: int, temperature: float,
                  max_tokens: int, suffix: str) -> dict:
    responses = await sampler(build_messages(record, suffix), n, temperature, max_tokens)
    flags = await asyncio.gather(*[verify(record, r) for r in responses])
    out = dict(record)
    out["n_samples"] = len(flags)
    out["n_correct"] = int(sum(flags))
    out["pass_rate"] = out["n_correct"] / max(len(flags), 1)
    out["difficulty"] = bucket(out["n_correct"], len(flags))
    return out


async def run(args, sampler: SampleFn, verify: VerifyFn) -> dict[str, int]:
    records = read_records(args.input, parse_field_map(args.field_map))
    cache = JsonlAppender(args.output + ".measured.jsonl")
    todo = [r for r in records if r["id"] not in cache.done]
    sem = asyncio.Semaphore(args.record_concurrency)

    async def work(r):
        async with sem:
            cache.write(await measure(r, sampler, verify, n=args.n, temperature=args.temperature,
                                      max_tokens=args.max_tokens, suffix=args.suffix))

    await asyncio.gather(*[work(r) for r in todo])
    cache.close()

    kept, counts = [], Counter()
    for r in iter_jsonl(args.output + ".measured.jsonl"):
        counts[r["difficulty"]] += 1
        if select(r, drop_all_correct=args.drop_all_correct, drop_all_wrong=args.drop_all_wrong,
                  min_pass=args.min_pass_rate, max_pass=args.max_pass_rate):
            kept.append(r)
    write_jsonl(args.output, kept)
    counts["kept"] = len(kept)
    return dict(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--field-map", action="append")
    p.add_argument("--endpoint", action="append", required=True, help="name=http://host:port/v1[@model]; first is used")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--drop-all-correct", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-all-wrong", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-pass-rate", type=float, default=0.0)
    p.add_argument("--max-pass-rate", type=float, default=1.0)
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--record-concurrency", type=int, default=128)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    name, (url, model) = next(iter(parse_endpoints(args.endpoint).items()))
    sampler = make_openai_sampler(url, model, concurrency=args.record_concurrency)
    print("difficulty result:", asyncio.run(run(args, sampler, reward_router_verify)))


if __name__ == "__main__":
    main()
