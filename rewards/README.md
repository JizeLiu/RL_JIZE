# rewards:团队奖励函数库

训练时 VERL 通过 `reward.custom_reward_function.path=rewards/entry.py`、`name=compute_score` 调用本目录
(`slurm/recipes/_shared/common.sh` 已配置)。`rewards/` 不依赖 `rlx`,也可在数据处理里复用。

| 文件 | 内容 |
|---|---|
| `math_reward.py` | `math_verify`:取 `\boxed{}` 答案并校验(安装 `math_verify` 库时使用符号等价) |
| `instruction_following.py` | IFEval 风格规则检查,约束放在 `extra_info["constraints"]`;`mode=all|fraction` |
| `llm_judge.py` | 异步 OpenAI 兼容接口的 LLM judge,带重试、并发限制、`fail_score` |
| `dispatch.py` + `routes.yaml` | 按 `data_source` 通配符路由到一个或多个加权奖励;未匹配时回退 VERL 的 `default_compute_score` |
| `registry.py` | `@register_reward("name")` |

返回值始终包含 `score`、`acc` 以及每个子奖励的 `reward/<name>`,便于在 wandb 中拆分查看。

## 新增奖励

```python
from rewards.registry import register_reward

@register_reward("my_reward")
def my_reward(solution_str, ground_truth, extra_info, **args) -> float: ...
```

然后在 `routes.yaml` 中引用 `name: my_reward`。放在仓库外的模块用 `RLX_REWARD_MODULES=pkg.mod1,pkg.mod2` 加载;
用 `RLX_REWARD_ROUTES=/path/routes.yaml` 覆盖路由文件。

## LLM judge 环境变量

`RLX_JUDGE_BASE_URL`、`RLX_JUDGE_MODEL`、`RLX_JUDGE_API_KEY`、`RLX_JUDGE_CONCURRENCY`、`RLX_JUDGE_TIMEOUT`。
计算节点通常不能联网,judge 需是集群内可达的服务(例如用 vLLM 起一个)。
