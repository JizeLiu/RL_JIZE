# Dataset schema

Working format between stages: JSONL, one record per line.

| field | required | notes |
|---|---|---|
| `id` | auto | string; generated from the row index if missing; used for resume/caching |
| `question` | yes | raw problem text (no chat template, no answer-format suffix) |
| `ground_truth` | for verifiable rewards | final answer (math) or reference answer (knowledge / judge) |
| `data_source` | yes | routes the sample to a reward in `rewards/routes.yaml`, e.g. `math/dapo`, `ifeval/v1`, `knowledge/wiki` |
| `solution` | optional | reference reasoning trace; used as the privileged hint by OPSD / RLSD / RLCSD |
| `constraints` | for `ifeval/*` | list of `{"type": ..., ...}` (see `rewards/instruction_following.py`) |
| `system_prompt` | optional | prepended as a system message |

Stages append metadata: `vote` (vote_gt), `pass_rate`, `n_correct`, `n_samples`, `difficulty` (difficulty).

Final parquet for VERL (`to_verl`), the layout VERL's `RLHFDataset` and reward loop expect:

```
data_source : str
prompt      : [{"role": "user", "content": "<question>\n\n<suffix>"}]
ability     : str
reward_model: {"style": "rule", "ground_truth": <answer>}
extra_info  : {"index": int, "split": str, "question": str, "solution": str, "constraints": ..., "pass_rate": float, ...}
```

`extra_info.question` feeds the LLM judge; `extra_info.solution` feeds the self-distillation hints.
