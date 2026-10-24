"""Convert curated JSONL (or parquet) into the parquet layout VERL's RLHFDataset reads.

    python -m data_process.to_verl --input train.jsonl --output data/train.parquet --split train \
        --default-data-source math/mine

Also writes ``<output>.stats.json`` with the data_source / difficulty histogram for the run log.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from data_process.io_utils import parse_field_map, read_records
from data_process.prompts import DEFAULT_SUFFIX, build_messages

PASSTHROUGH = ("pass_rate", "n_correct", "difficulty", "gt_source")


def to_verl_row(record: dict, index: int, split: str, suffix: str, default_data_source: str | None) -> dict:
    data_source = record.get("data_source") or default_data_source
    if not data_source:
        raise ValueError(f"record {record['id']} has no data_source (use --default-data-source)")
    if record.get("ground_truth") is None and not record.get("constraints"):
        raise ValueError(f"record {record['id']} has neither ground_truth nor constraints")
    extra = {
        "index": index,
        "split": split,
        "id": record["id"],
        "question": record["question"],
        "solution": record.get("solution") or "",
        # JSON string: heterogeneous constraint dicts do not fit one parquet struct schema.
        "constraints": json.dumps(record["constraints"]) if record.get("constraints") else "",
    }
    for key in PASSTHROUGH:
        if key in record:
            extra[key] = record[key]
    return {
        "data_source": data_source,
        "prompt": build_messages(record, suffix),
        "ability": record.get("ability", data_source.split("/")[0]),
        "reward_model": {"style": "rule", "ground_truth": str(record.get("ground_truth", ""))},
        "extra_info": extra,
    }


def convert(records: list[dict], split: str, suffix: str, default_data_source: str | None) -> pd.DataFrame:
    return pd.DataFrame([to_verl_row(r, i, split, suffix, default_data_source) for i, r in enumerate(records)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--field-map", action="append")
    p.add_argument("--suffix", default=DEFAULT_SUFFIX, help="appended to the question; '' disables")
    p.add_argument("--default-data-source")
    args = p.parse_args(argv)

    records = read_records(args.input, parse_field_map(args.field_map))
    df = convert(records, args.split, args.suffix, args.default_data_source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    stats = {
        "rows": len(df),
        "data_source": dict(Counter(df["data_source"])),
        "difficulty": dict(Counter(r.get("difficulty", "n/a") for r in records)),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
