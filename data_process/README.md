# data_process:数据处理流水线

工作格式为 JSONL,字段见 [`SCHEMA.md`](SCHEMA.md)。各阶段可以单独运行,都支持断点续跑。
采样阶段需要 OpenAI 兼容服务(vLLM / SGLang);在 Slurm 上可用 `slurm/data/run_with_vllm.sbatch` 自动拉起。
判分复用训练用的奖励路由(`rewards/`),保证数据筛选与训练判定一致。

```
raw.jsonl ─► vote_gt ─► difficulty ─► to_verl ─► train.parquet
```

## 1. 多模型投票校验 GT:`vote_gt`

每个模型对每题采样 `--n` 次;某模型的答案中超过一半通过校验则视为"支持该标签";支持的模型占比 ≥ `--min-agree` 才保留。
被拒绝的样本写入 `*.rejected.jsonl`,并给出 `consensus_answer` / `consensus_ratio`(多个模型一致认为的另一个答案,通常说明标签有误);
`--relabel-ratio` 可把高共识答案提升为新标签。

```bash
python -m data_process.vote_gt --input raw.jsonl --output voted.jsonl \
  --endpoint qwen=http://n1:8000/v1 --endpoint llama=http://n1:8001/v1 --n 2 --min-agree 0.6
```

## 2. 难度筛选:`difficulty`

同一模型对每题采样 `--n`(默认 5)次,按通过率分档:`all_wrong / hard / medium / easy / all_correct`。
默认**剔除全对**,**保留全错**(`--drop-all-wrong` 可剔除);还可用 `--min-pass-rate` / `--max-pass-rate` 进一步限制。

```bash
python -m data_process.difficulty --input voted.jsonl --output hard.jsonl \
  --endpoint policy=http://n1:8000/v1 --n 5
```

注意:全错样本可能是标签错误或题目过难,建议先经过第 1 步投票。

## 3. 转 VERL 格式:`to_verl`

```bash
python -m data_process.to_verl --input hard.jsonl --output train.parquet --split train
```

同时输出 `train.stats.json`(`data_source` 分布等)。`--field-map question=problem` 可重命名输入字段。

## 在 Slurm 上一键运行

```bash
MODELS="qwen=/shared/models/Qwen3-8B,llama=/shared/models/Llama-3.1-8B-Instruct" TP=1 \
CMD='python -m data_process.vote_gt --input raw.jsonl --output voted.jsonl --n 2 --min-agree 0.6 {ENDPOINTS}' \
sbatch --partition=gpu --gres=gpu:8 --nodes=1 slurm/data/run_with_vllm.sbatch
```
