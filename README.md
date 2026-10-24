# RL_JIZE (rlx)

基于 [VERL](https://github.com/volcengine/verl) 的 LLM 强化学习扩展框架。VERL 作为未修改的 git submodule 固定在
`third_party/verl`(v0.9.1),所有自研内容都在本仓库的独立目录里,通过 VERL 的官方扩展点接入,不 patch VERL 源码。

| 目录 | 作用 |
|---|---|
| `third_party/verl` | VERL 子模块(只读,版本见 `VERL_VERSION`) |
| `rlx/` | 新算法:RLSD / OPSD / RLCSD(自蒸馏族),通过 `VERL_USE_EXTERNAL_MODULES` 插件加载 |
| `rewards/` | 奖励函数库:数学、指令遵循、LLM-as-judge,统一路由入口 |
| `data_process/` | 数据处理:多模型投票校验 GT、难度筛选、转 VERL parquet |
| `slurm/` | Slurm 训练脚本、提交脚本、各算法 recipe |
| `scripts/` | VERL 版本兼容性检查、recipe 校验 |
| `tests/` | CPU 单测 |

## 设计原则

1. **不改 VERL**。GRPO / PPO 等原有算法与性能完全不受影响:未设置 `RLX_PLUGINS` 时插件什么都不做,
   `grpo` / `ppo` / `justrl2` recipe 只是纯 Hydra 覆盖参数,与直接跑 VERL 等价。
2. **松耦合**。`rewards/` 与 `data_process/` 不依赖 `rlx`;`rlx` 只依赖 VERL 的公开扩展点
   (`register_trainer`、`VERL_USE_EXTERNAL_MODULES`),并有静态检查脚本盯住这些内部接口。
3. **新算法 = 对 GRPO 优势的后处理**。自蒸馏算法继承 VERL 的同步 PPOTrainer,只在 `_compute_advantage` 之后
   额外做"带特权提示(hint)的 teacher 前向",再重塑优势;训练/rollout 路径完全沿用 VERL。

## 快速开始

```bash
git clone --recurse-submodules <this repo> && cd RL_JIZE
pip install -e third_party/verl        # 按 VERL 文档安装其依赖(GPU 环境)
pip install -e ".[data,rewards,test]"

cp slurm/site.env.example slurm/site.env   # 填分区、模型路径、数据路径、python 环境
python -m pytest tests -q                  # CPU 单测
python scripts/check_verl_compat.py        # VERL 内部接口检查
python scripts/validate_recipes.py         # 所有 recipe 能否通过 VERL 的 Hydra 配置组合
scripts/check_verl_clean.sh                # 确认子模块未被修改

slurm/submit.sh rlsd --dry-run             # 只生成运行目录和命令,不提交
slurm/submit.sh rlsd --nodes 4 actor_rollout_ref.rollout.n=8
```

## 算法一览

| recipe | 说明 | 额外开销 |
|---|---|---|
| `grpo` | VERL 原生 GRPO | 无 |
| `ppo` | VERL 原生 PPO(GAE + critic) | 无 |
| `justrl2` | JustRL 论文的极简 GRPO 配置(纯参数) | 无 |
| `rlsd` | 验证器优势定方向,特权自教师(同权重 + hint 提示)调节 token 级幅度 | +1 次前向 |
| `opsd` | 以特权自教师的 token 级 log-ratio 作为优势 | +1 次前向 |
| `rlcsd` | 正/负 hint 对比信号 + 双路径权重折进优势 | +(1+K) 次前向 |

说明:VERL 的 RL trainer 没有 DPO,因此这里没有 DPO recipe。"justRL2" 按 JustRL 论文配置实现。

## 文档

- [`rlx/README.md`](rlx/README.md):自蒸馏算法原理、配置项、如何新增算法、升级 VERL
- [`rewards/README.md`](rewards/README.md):奖励函数与路由
- [`data_process/README.md`](data_process/README.md):数据处理流水线
- [`slurm/README.md`](slurm/README.md):集群部署与 recipe
- [`docs/STATUS.md`](docs/STATUS.md):验证状态与已知限制
