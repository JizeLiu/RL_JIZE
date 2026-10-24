# rlx:自蒸馏算法(RLSD / OPSD / RLCSD)

## 工作方式

`rlx.selfdistill.trainer.SelfDistillTrainer` 用 `@register_trainer("selfdistill_sync")` 注册,继承 VERL 的
`PPOTrainerSync`。它先走 VERL 原有的 GRPO 优势计算,然后:

1. 为每条 rollout 规划 hint(`hints.py`):正 hint = 数据集 GT 解答(`positive_source=gt`)或同组正确的兄弟 rollout
   (`rollout`);负 hint = 同组错误的兄弟 rollout(不会选到自己),正负 hint 使用同一模板。
2. 用"带 hint 的 prompt"让 teacher 对 student 已生成的 response 做一次前向,得到 token 级 log-prob。
   teacher = 同一份权重(`teacher_mode=actor`,在线)或冻结的 ref(`teacher_mode=ref`,需要 ref 模型)。
3. 用 `core.py` 中的纯 torch 函数重塑优势,再写回 TransferQueue 的 `advantages` / `returns`。

| 方法 | 优势 |
|---|---|
| `opsd` | `clip(Δ)`,Δ = log p_teacher − log p_student(采样 token) |
| `rlsd` | `A·((1−λ) + λ·clip(exp(sign(A)·Δ), 1±ε_w))`,λ 线性衰减 |
| `rlcsd` | 正 hint 与 K 个负 hint 的对比信号 → tanh 缩放 → token mask → 保号 clamp → 双路径权重 |

## 启用

```bash
export RLX_PLUGINS=rlx.selfdistill.trainer            # recipe 里已设置
export VERL_USE_EXTERNAL_MODULES=rlx.plugin           # train.sbatch 在 RLX_PLUGINS 非空时自动设置
python -m verl.trainer.main_ppo trainer.v1.trainer_mode=selfdistill_sync +selfdistill.method=rlsd ...
```

`rlx.plugin` 只导入 `RLX_PLUGINS`(逗号分隔)里列出的模块;未设置时什么都不做。

配置项全部在顶层 `selfdistill` 节点(VERL 不认识的 key,用 `+selfdistill.xxx=` 追加),见 `rlx/selfdistill/config.py`;
未知 key 会报错。

## 新增算法

1. 在 `rlx/<name>/` 写模块,继承 `PPOTrainerSync`(或 VERL 的其它 trainer),用 `@register_trainer("<name>")` 注册;
   或用 `register_adv_est` / `register_policy_loss` 注册优势估计器和策略损失。
2. 在 `RLX_PLUGINS` 中加入该模块路径。
3. 新建 `slurm/recipes/<name>/{recipe.sh,resources.env}`,`tests/test_recipes.py` 会自动覆盖新 recipe 的配置组合。

## 升级 VERL

```bash
git -C third_party/verl fetch --tags && git -C third_party/verl checkout <new tag>
echo <new tag> > VERL_VERSION
python scripts/check_verl_compat.py     # 逐项列出我们依赖的 VERL 内部接口是否仍然存在
python -m pytest tests -q
```

任何 FAIL 行都对应要修的扩展点;不要去改 `third_party/verl`(`scripts/check_verl_clean.sh` 会检查)。
