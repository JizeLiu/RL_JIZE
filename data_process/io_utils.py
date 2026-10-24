"""Record IO. The working format between stages is JSONL (one dict per line); only the final stage
writes the parquet that VERL reads."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd


def parse_field_map(items: list[str] | None) -> dict[str, str]:
    """``["question=problem", "ground_truth=answer"]`` -> {target: source}."""
    out = {}
    for item in items or []:
        target, _, source = item.partition("=")
        if not source:
            raise ValueError(f"--field-map entries look like target=source, got {item!r}")
        out[target] = source
    return out


def read_records(path: str | Path, field_map: dict[str, str] | None = None) -> list[dict]:
    path = Path(path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    elif path.suffix in (".jsonl", ".json"):
        with path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"unsupported input format: {path}")
    for target, source in (field_map or {}).items():
        for r in records:
            if source in r:
                r[target] = r[source]
    for i, r in enumerate(records):
        r.setdefault("id", str(i))
        r["id"] = str(r["id"])
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class JsonlAppender:
    """Append-only writer that makes long, preemptible Slurm jobs resumable: finished ids are skipped
    on restart."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {r["id"]: r for r in iter_jsonl(self.path)}
        self._f = self.path.open("a")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        self.done[record["id"]] = record

    def close(self) -> None:
        self._f.close()
