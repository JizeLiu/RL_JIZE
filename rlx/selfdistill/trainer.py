"""Self-distillation trainer plugin for VERL's v1 (TransferQueue) trainer.

How it plugs in (VERL source is untouched):

* ``rlx.plugin`` imports this module inside every VERL process (``VERL_USE_EXTERNAL_MODULES``), which
  runs ``@register_trainer("selfdistill_sync")``.
* The recipe selects it with ``trainer.v1.trainer_mode=selfdistill_sync``.
* The only overridden hook is ``_compute_advantage``: stock VERL computes the GRPO advantage, then
  we rewrite the ``advantages`` field in TransferQueue. Rollout, reward, policy loss, optimizer and
  checkpointing are all stock VERL, so throughput of those stages is unchanged. The additional cost is
  one actor/ref forward per hint slot (RLSD/OPSD: 1, RLCSD: 1 + K).

The teacher is the same weights as the student, queried on ``[hinted prompt | rollout]`` through VERL's
own ``compute_log_prob`` worker method. Hinted sequences are written to a scratch TransferQueue
partition and cleared right after use.

Version coupling: this file depends on ``verl.trainer.ppo.v1.trainer_base.PPOTrainer`` internals
(``_compute_advantage``, ``actor_rollout_ref`` worker groups, KVBatchMeta). ``scripts/check_verl_compat.py``
statically verifies those symbols against the pinned submodule.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import transfer_queue as tq
from tensordict import TensorDict
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_from_nested, response_to_nested

from rlx.selfdistill import core
from rlx.selfdistill.config import SelfDistillConfig
from rlx.selfdistill.hints import (
    DEFAULT_TEMPLATE,
    HintSource,
    extract_boxed,
    plan_hints,
    render_hint_messages,
)

logger = logging.getLogger(__name__)

SD_PARTITION = "rlx_selfdistill"


def _as_list(x) -> list:
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


@register_trainer("selfdistill_sync")
class SelfDistillTrainer(PPOTrainerSync):
    """Synchronous v1 trainer whose advantage is post-processed by OPSD / RLSD / RLCSD."""

    # VERL branches on ``trainer_mode == "sync"`` in several places (replay buffer, checkpointing, ...).
    # We need a distinct registry name but identical behaviour, so pin the effective mode to "sync".
    @property
    def trainer_mode(self) -> str:
        return "sync"

    @trainer_mode.setter
    def trainer_mode(self, value: str) -> None:
        pass

    def __init__(self, config):
        super().__init__(config)
        self.sd = SelfDistillConfig.from_config(config)
        rollout_corr = config.algorithm.get("rollout_correction", None)
        if self.sd.method != "none" and rollout_corr and rollout_corr.get("bypass_mode", False):
            raise ValueError(
                "selfdistill needs recomputed old_log_probs from the training engine; "
                "disable algorithm.rollout_correction.bypass_mode"
            )
        if self.sd.method == "rlcsd" and self.sd.teacher_mode == "ref" and not self.use_reference_policy:
            raise ValueError(
                "selfdistill.teacher_mode=ref needs a reference model: set actor_rollout_ref.actor.use_kl_loss=true "
                "(with kl_loss_coef=0 if you do not want the KL term), or use teacher_mode=actor"
            )
        self._template = DEFAULT_TEMPLATE
        if self.sd.template_path:
            self._template = Path(self.sd.template_path).read_text()
        logger.info("selfdistill trainer: %s", self.sd)

    # ------------------------------------------------------------------ hook

    def _compute_advantage(self, batch, metrics: dict):
        batch = super()._compute_advantage(batch, metrics)
        if self.sd.method == "none":
            return batch
        start = time.perf_counter()
        self._apply_selfdistill(batch, metrics)
        metrics["timing_s/selfdistill"] = time.perf_counter() - start
        return batch

    # ------------------------------------------------------------- main logic

    def _apply_selfdistill(self, batch, metrics: dict) -> None:
        sd = self.sd
        keys, part = batch.keys, batch.partition_id
        select = ["uid", "prompts", "responses", "response_mask", "rm_scores", "old_log_probs", "advantages", "raw_prompt"]
        if sd.positive_source == "gt":
            select += ["reward_model", "extra_info"]
        data = tq.kv_batch_get(keys=keys, partition_id=part, select_fields=select)

        uids = [str(u) for u in _as_list(data["uid"])]
        rewards = [float(t.sum()) for t in data["rm_scores"].unbind()]
        prompts = list(data["prompts"].unbind())
        responses = list(data["responses"].unbind())
        raw_prompts = [list(p) for p in list(data["raw_prompt"])]
        extra_info = list(data["extra_info"]) if "extra_info" in data.keys() else None
        reward_model = list(data["reward_model"]) if "reward_model" in data.keys() else None

        resp_mask_nested = data["response_mask"]
        mask = resp_mask_nested.to_padded_tensor(padding=0.0).float()
        adv = data["advantages"].to_padded_tensor(padding=0.0).float()
        old_lp = data["old_log_probs"].to_padded_tensor(padding=0.0).float()

        n_neg = sd.n_neg if sd.method == "rlcsd" else 0
        plans = plan_hints(
            uids,
            rewards,
            positive_source=sd.positive_source,
            n_neg=n_neg,
            correct_threshold=sd.correct_threshold,
            seed=sd.seed * 1_000_003 + int(self.global_steps),
        )

        ctx = _HintContext(
            trainer=self,
            prompts=prompts,
            responses=responses,
            raw_prompts=raw_prompts,
            extra_info=extra_info,
            reward_model=reward_model,
        )

        pos_lp, pos_valid = self._teacher_slot(keys, resp_mask_nested, ctx, [p.positive for p in plans], slot=0)
        pos_lp = self._fit_width(pos_lp, mask.shape[1])

        if sd.method == "opsd":
            delta = core.privileged_gain(pos_lp, old_lp) * pos_valid[:, None]
            new_adv = core.opsd_advantage(delta, mask, clip=sd.opsd_clip)
            stats = {}
        elif sd.method == "rlsd":
            delta = core.privileged_gain(pos_lp, old_lp) * pos_valid[:, None]
            lam = core.lambda_schedule(int(self.global_steps), sd.lam_start, sd.lam_end, sd.lam_decay_steps)
            new_adv = core.rlsd_advantage(adv, delta, mask, eps_w=sd.eps_w, lam=lam)
            stats = {"lambda": lam}
        else:
            neg_lps, neg_valid = [], []
            for k in range(n_neg):
                sources = [p.negatives[k] if k < len(p.negatives) else None for p in plans]
                lp_k, valid_k = self._teacher_slot(keys, resp_mask_nested, ctx, sources, slot=k + 1)
                neg_lps.append(self._fit_width(lp_k, mask.shape[1]))
                neg_valid.append(valid_k)
            neg_lps_t = torch.stack(neg_lps, dim=1)
            neg_valid_t = torch.stack(neg_valid, dim=1)
            e_ctr = core.contrastive_signal(pos_lp, neg_lps_t, neg_valid_t)
            params = core.RLCSDParams(
                lam=sd.rlcsd_lam,
                tau=sd.rlcsd_tau,
                delta=sd.rlcsd_delta,
                eta=sd.rlcsd_eta,
                two_path=sd.rlcsd_two_path,
            )
            new_adv, stats = core.rlcsd_advantage(adv, e_ctr, mask, params)
            row_ok = pos_valid & neg_valid_t.any(dim=1)
            new_adv = torch.where(row_ok[:, None], new_adv, adv)
            delta = e_ctr

        maskb = mask.bool()
        metrics.update(
            {
                "selfdistill/hint_coverage": float(pos_valid.float().mean()),
                "selfdistill/signal_abs_mean": core.masked_mean(delta.abs(), maskb),
                "selfdistill/adv_abs_mean_before": core.masked_mean(adv.abs(), maskb),
                "selfdistill/adv_abs_mean_after": core.masked_mean(new_adv.abs(), maskb),
                **{f"selfdistill/{k}": v for k, v in stats.items()},
            }
        )

        nested = response_to_nested(new_adv, resp_mask_nested)
        fields = TensorDict({"advantages": nested, "returns": nested}, batch_size=len(keys))
        tq.kv_batch_put(keys=keys, partition_id=part, fields=fields)

    # --------------------------------------------------------- teacher passes

    @staticmethod
    def _fit_width(t: torch.Tensor, width: int) -> torch.Tensor:
        if t.shape[1] == width:
            return t
        if t.shape[1] > width:
            return t[:, :width]
        return torch.nn.functional.pad(t, (0, width - t.shape[1]))

    def _teacher_slot(self, keys, resp_mask_nested, ctx, sources, slot: int):
        """Run one teacher forward (one hint per row). Returns (logp (B,T), valid (B,))."""
        seqs, valid = [], []
        for i, src in enumerate(sources):
            if src is None:
                seqs.append(torch.cat([ctx.prompts[i], ctx.responses[i]]))
                valid.append(False)
            else:
                seqs.append(torch.cat([ctx.hinted_prompt_ids(i, src), ctx.responses[i]]))
                valid.append(True)

        tkeys = [f"{k}::sd{slot}" for k in keys]
        input_ids = torch.nested.as_nested_tensor(seqs, layout=torch.jagged)
        position_ids = torch.nested.as_nested_tensor([torch.arange(len(s)) for s in seqs], layout=torch.jagged)
        td = TensorDict(
            {"input_ids": input_ids, "position_ids": position_ids, "response_mask": resp_mask_nested},
            batch_size=len(seqs),
        )
        resp_lens = resp_mask_nested.offsets().diff().tolist()
        tags = [
            {"status": "success", "seq_len": len(s), "prompt_len": len(s) - r, "response_len": r}
            for s, r in zip(seqs, resp_lens, strict=True)
        ]
        meta = tq.kv_batch_put(keys=tkeys, partition_id=SD_PARTITION, fields=td, tags=tags)
        meta.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        try:
            if self.sd.teacher_mode == "ref":
                if self.ref_in_actor:
                    meta.extra_info["no_lora_adapter"] = True
                    output = self.actor_rollout_wg.compute_log_prob(meta)
                else:
                    output = self.ref_policy_wg.compute_ref_log_prob(meta)
            else:
                output = self.actor_rollout_wg.compute_log_prob(meta)
            assert len(output) == len(meta)
            res = tq.kv_batch_get(keys=tkeys, partition_id=SD_PARTITION, select_fields=["log_probs", "response_mask"])
            lp = response_from_nested(res["log_probs"], res["response_mask"]).to_padded_tensor(padding=0.0).float()
        finally:
            tq.kv_clear(tkeys, SD_PARTITION)
        return lp, torch.tensor(valid, dtype=torch.bool)


class _HintContext:
    """Builds the token ids of ``hinted prompt`` for a row from a ``HintSource``."""

    def __init__(self, trainer: SelfDistillTrainer, prompts, responses, raw_prompts, extra_info, reward_model):
        self.trainer = trainer
        self.prompts = prompts
        self.responses = responses
        self.raw_prompts = raw_prompts
        self.extra_info = extra_info
        self.reward_model = reward_model
        self.tokenizer = trainer.tokenizer
        self.template_kwargs = dict(trainer.config.data.get("apply_chat_template_kwargs", {}) or {})

    def _reference(self, src: HintSource, row: int) -> tuple[str, str]:
        sd = self.trainer.sd
        if src.kind == "gt":
            info = self.extra_info[row] if self.extra_info is not None else {}
            solution = str((info or {}).get(sd.gt_solution_key, "") or "")
            rm = self.reward_model[row] if self.reward_model is not None else {}
            answer = str((rm or {}).get("ground_truth", ""))
            return solution, answer
        ids = self.responses[src.row].tolist()[: sd.max_hint_tokens]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, extract_boxed(text) or "(not stated)"

    def hinted_prompt_ids(self, row: int, src: HintSource) -> torch.Tensor:
        solution, answer = self._reference(src, row)
        messages = render_hint_messages(self.raw_prompts[row], solution=solution, answer=answer, template=self.trainer._template)
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.template_kwargs
        )
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=self.prompts[row].dtype)
