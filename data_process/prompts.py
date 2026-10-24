"""Prompt construction shared by the sampling tools and the VERL converter."""

from __future__ import annotations

DEFAULT_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."


def build_messages(record: dict, suffix: str = DEFAULT_SUFFIX) -> list[dict]:
    content = record["question"] if not suffix else f"{record['question']}\n\n{suffix}"
    messages = []
    if record.get("system_prompt"):
        messages.append({"role": "system", "content": record["system_prompt"]})
    messages.append({"role": "user", "content": content})
    return messages
