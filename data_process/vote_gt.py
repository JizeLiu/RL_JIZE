"""Multi-model voting to audit ground-truth labels.

Each model answers every question ``--n`` times. A model *supports* the label when more than half of
its answers verify against it. A record is kept when at least ``--min-agree`` of the models support
the label. For rejected records we also report whether the models agree on a different answer
(``consensus_answer``), which usually points at a wrong label; ``--relabel-ratio`` can promote such a
consensus to the new label.

    python -m data_process.vote_gt --input raw.jsonl --output voted.jsonl \
        --endpoint qwen=http://n1:8000/v1 --endpoint llama=http://n1:8001/v1 --n 2 --min-agree 0.67
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


def _boxed_answers(responses: list[str]) -> list[str]:
    from rewards.text_utils import extract_boxed, normalize_math

    return [normalize_math(a) for a in (extract_boxed(r) for r in responses) if a]


async def vote_record(
    record: dict,
    samplers: dict[str, SampleFn],
    verify: VerifyFn,
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    suffix: str,
) -> dict:
    messages = build_messages(record, suffix)

    async def one_model(name: str, sampler: SampleFn):
        responses = await sampler(messages, n, temperature, max_tokens)
        flags = await asyncio.gather(*[verify(record, r) for r in responses])
        return name, responses, list(flags)

    results = await asyncio.gather(*[one_model(k, s) for k, s in samplers.items()])
    support = {name: sum(flags) * 2 > len(flags) for name, _, flags in results}
    out = dict(record)
    out["vote"] = {
        "supports": support,
        "n_support": sum(support.values()),
        "n_models": len(support),
        "per_model_correct": {name: sum(flags) / max(len(flags), 1) for name, _, flags in results},
    }
    all_answers = [a for _, responses, _ in results for a in _boxed_answers(responses)]
    if all_answers:
        answer, count = Counter(all_answers).most_common(1)[0]
        out["vote"]["consensus_answer"] = answer
        out["vote"]["consensus_ratio"] = count / len(all_answers)
    return out


def decide(voted: dict, min_agree: float, relabel_ratio: float | None) -> tuple[str, dict]:
    """Return (``keep`` | ``relabel`` | ``reject``, record)."""
    vote = voted["vote"]
    if vote["n_support"] / vote["n_models"] >= min_agree:
        return "keep", voted
    if relabel_ratio is not None and vote.get("consensus_ratio", 0.0) >= relabel_ratio:
        new = dict(voted)
        new["original_ground_truth"] = voted.get("ground_truth")
        new["ground_truth"] = vote["consensus_answer"]
        new["gt_source"] = "model_consensus"
        return "relabel", new
    return "reject", voted


async def run(args, samplers: dict[str, SampleFn], verify: VerifyFn) -> dict[str, int]:
    records = read_records(args.input, parse_field_map(args.field_map))
    cache = JsonlAppender(args.output + ".votes.jsonl")
    todo = [r for r in records if r["id"] not in cache.done]
    sem = asyncio.Semaphore(args.record_concurrency)

    async def work(r):
        async with sem:
            cache.write(
                await vote_record(
                    r, samplers, verify, n=args.n, temperature=args.temperature, max_tokens=args.max_tokens,
                    suffix=args.suffix,
                )
            )

    await asyncio.gather(*[work(r) for r in todo])
    cache.close()

    kept, rejected, counts = [], [], Counter()
    for r in iter_jsonl(args.output + ".votes.jsonl"):
        verdict, rec = decide(r, args.min_agree, args.relabel_ratio)
        counts[verdict] += 1
        (rejected if verdict == "reject" else kept).append(rec)
    write_jsonl(args.output, kept)
    write_jsonl(args.output + ".rejected.jsonl", rejected)
    return dict(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--field-map", action="append", help="target=source, e.g. question=problem")
    p.add_argument("--endpoint", action="append", required=True, help="name=http://host:port/v1[@model]")
    p.add_argument("--n", type=int, default=2, help="answers per model per question")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--min-agree", type=float, default=0.5, help="fraction of models that must support the label")
    p.add_argument("--relabel-ratio", type=float, default=None, help="promote a consensus answer shared by this share of answers")
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--record-concurrency", type=int, default=128)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    samplers = {
        name: make_openai_sampler(url, model, concurrency=args.record_concurrency)
        for name, (url, model) in parse_endpoints(args.endpoint).items()
    }
    counts = asyncio.run(run(args, samplers, reward_router_verify))
    print("vote result:", counts)


if __name__ == "__main__":
    main()
