# 验证状态与已知限制

## 已验证(CPU)

- `pytest tests`:算法数学、hint 规划、奖励路由、数据处理、Slurm 覆盖参数去重、recipe 组合。
- 真实 VERL(v0.9.1)可导入,插件与 trainer 注册生效。
- 所有 recipe 通过 VERL 真实 Hydra 配置组合,且 `SelfDistillConfig` 能解析;stock recipe(grpo/ppo/justrl2)不带 selfdistill 节点。
- 自蒸馏 trainer 的数据通路用假 TransferQueue + 假 worker group 跑通。
- `train.sbatch` 用假的 sbatch/srun/ray 走过一遍。
- `scripts/check_verl_compat.py`:我们依赖的 VERL 内部接口检查。

## 未验证(需要 GPU 集群)

- 真实 Ray / 真实 TransferQueue / 真实模型上的端到端训练。
- 自蒸馏 teacher 前向的数值行为与收敛效果;各超参默认值是按论文设定的起点,不是调优结果。
- 多节点 Ray 引导在你们站点上的表现(网络接口、`srun --overlap` 策略)。

## 已知偏差与限制

- RLSD / OPSD 的 teacher 是在线 actor(同权重 + hint);RLCSD 论文使用固定/快照 teacher,这里 recipe 默认 `ref`(冻结初始模型),同样支持 `actor`。
- OPSD 实现为"采样 token 上的优势",不是对完整词表分布的稠密 KL。
- teacher 前向对整个 batch 计算,包含优势全为 0 的组(未做跳过优化)。额外开销:RLSD/OPSD +1 次前向,RLCSD +(1+K) 次。
- VERL 的 RL trainer 没有 DPO,所以没有 DPO recipe。
- "justRL2" 按 JustRL 论文超参实现。
- 自蒸馏 trainer 要求同步模式、非 bypass 模式。
