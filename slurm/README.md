# slurm:集群部署

```
slurm/
  site.env.example      集群相关配置(复制为 site.env,已 gitignore)
  submit.sh             提交入口
  train.sbatch          通用训练 job:起 Ray 集群 → python -m verl.trainer.main_ppo
  common/               env.sh / ray_cluster.sh / overrides.sh
  recipes/<algo>/       recipe.sh(Hydra 参数)+ resources.env(默认节点数/时长)
  recipes/_shared/      公共参数(common.sh)、自蒸馏公共参数(selfdistill.sh)
  data/run_with_vllm.sbatch   数据处理用的 vLLM 启动器
```

## 提交

```bash
slurm/submit.sh <recipe> [--dry-run] [--nodes N] [--time HH:MM:SS] [--name NAME] [hydra.key=value ...]
slurm/submit.sh rlsd --nodes 4 actor_rollout_ref.rollout.n=8 +selfdistill.lam_start=0.3
FOO=bar slurm/submit.sh grpo       # 常用超参也可通过环境变量(见 recipes/_shared/common.sh)
```

每次提交会在 `$RUN_ROOT/<recipe>_<时间>/` 下创建运行目录,记录 `meta.txt`(rlx 与 verl 的 commit、是否有未提交修改)、
`overrides.txt`、recipe 快照、`hydra_args.txt`、日志与 checkpoint。命令行 `key=value` 优先级最高,会覆盖 recipe;
重复的 `+key` 会自动去重(Hydra 不允许重复追加)。

## Recipe

`grpo`、`ppo`、`justrl2`、`rlsd`、`opsd`、`rlcsd`。新增算法见 `rlx/README.md`。`rlcsd` 使用
`use_kl_loss=True, kl_loss_coef=0.0`,目的是让 VERL 创建 ref 模型作为冻结 teacher。

## 注意

- 环境变量(包括 `RLX_PLUGINS`)必须在 `ray start` 之前导出,`train.sbatch` 已按此顺序处理。
- 计算节点一般离线:模型与数据使用共享文件系统的绝对路径,`HF_HUB_OFFLINE=1`。
- 多节点 Ray 启动用 `scontrol show hostnames` + `srun --overlap`,若站点策略不同,改 `common/ray_cluster.sh` 即可。
