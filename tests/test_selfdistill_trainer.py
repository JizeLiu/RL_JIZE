"""Exercises SelfDistillTrainer._apply_selfdistill against an in-memory TransferQueue and a fake actor.

Needs the real (unmodified) VERL source importable; skipped otherwise. It validates the data plumbing
(field names, nested-tensor handling, scratch partition cleanup, write-back) but NOT GPU numerics.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "verl"))
pytest.importorskip("verl.trainer.ppo.v1.trainer_base")
pytest.importorskip("transfer_queue")

from omegaconf import OmegaConf  # noqa: E402
from tensordict import NonTensorStack, TensorDict  # noqa: E402
from transfer_queue import KVBatchMeta  # noqa: E402

import rlx.selfdistill.trainer as trainer_mod  # noqa: E402
from rlx.selfdistill.config import SelfDistillConfig  # noqa: E402


class FakeTQ:
    def __init__(self):
        self.store: dict[tuple[str, str], dict] = {}

    def kv_batch_put(self, keys, partition_id, fields=None, tags=None, **_):
        for i, k in enumerate(keys):
            row = self.store.setdefault((partition_id, k), {})
            for name in fields.keys():
                v = fields[name]
                if isinstance(v, torch.Tensor):
                    row[name] = v.unbind()[i] if v.is_nested else v[i]
                else:
                    row[name] = list(v)[i]
        return KVBatchMeta(keys=list(keys), tags=tags or [{} for _ in keys], partition_id=partition_id)

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        out = {}
        for name in select_fields:
            vals = [self.store[(partition_id, k)][name] for k in keys]
            if all(isinstance(v, torch.Tensor) for v in vals):
                out[name] = torch.nested.as_nested_tensor(vals, layout=torch.jagged)
            else:
                out[name] = NonTensorStack(*vals)
        return TensorDict(out, batch_size=len(keys))

    def kv_clear(self, keys, partition_id):
        for k in keys:
            self.store.pop((partition_id, k), None)


class FakeWorkerGroup:
    """log_probs[p] = -0.01 * ids[p+1] + 0.001 * seq_len: longer (hinted) contexts get higher log-prob."""

    def __init__(self, tq):
        self.tq = tq
        self.calls = []

    def compute_log_prob(self, meta):
        self.calls.append(meta)
        data = self.tq.kv_batch_get(meta.keys, meta.partition_id, ["input_ids"])
        outs = []
        for ids in data["input_ids"].unbind():
            nxt = torch.roll(ids, -1).float()
            outs.append(-0.01 * nxt + 0.001 * len(ids))
        self.tq.kv_batch_put(
            meta.keys,
            meta.partition_id,
            TensorDict({"log_probs": torch.nested.as_nested_tensor(outs, layout=torch.jagged)}, batch_size=len(outs)),
        )
        return meta

    compute_ref_log_prob = compute_log_prob


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
        assert not tokenize
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 90 + 5 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def build_trainer(monkeypatch, sd_overrides):
    tq = FakeTQ()
    monkeypatch.setattr(trainer_mod, "tq", tq)
    t = object.__new__(trainer_mod.SelfDistillTrainer)
    t.sd = SelfDistillConfig(**sd_overrides)
    t.sd.validate()
    t._template = trainer_mod.DEFAULT_TEMPLATE
    t.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"temperature": 1.0}}, "data": {"apply_chat_template_kwargs": {}}}
    )
    t.global_steps = 1
    t.tokenizer = FakeTokenizer()
    t.ref_in_actor = False
    t.actor_rollout_wg = FakeWorkerGroup(tq)
    t.ref_policy_wg = FakeWorkerGroup(tq)
    return t, tq


def populate(tq, rewards, advs, uids):
    keys = [f"k{i}" for i in range(len(uids))]
    prompts, responses, masks, rm, ad, olp = [], [], [], [], [], []
    for i in range(len(uids)):
        p = torch.arange(5, 5 + 4 + i)  # variable prompt length
        r = torch.arange(10, 10 + 3 + (i % 2))  # variable response length
        prompts.append(p)
        responses.append(r)
        masks.append(torch.ones(len(r), dtype=torch.int64))
        score = torch.zeros(len(r))
        score[-1] = rewards[i]
        rm.append(score)
        ad.append(torch.full((len(r),), advs[i]))
        ids = torch.cat([p, r])
        nxt = torch.roll(ids, -1).float()
        olp.append((-0.01 * nxt + 0.001 * len(ids))[-len(r) - 1 : -1])  # student logp as the fake actor yields
    jag = lambda xs: torch.nested.as_nested_tensor(xs, layout=torch.jagged)  # noqa: E731
    fields = TensorDict(
        {
            "uid": NonTensorStack(*uids),
            "prompts": jag(prompts),
            "responses": jag(responses),
            "response_mask": jag(masks),
            "rm_scores": jag(rm),
            "advantages": jag(ad),
            "old_log_probs": jag(olp),
            "raw_prompt": NonTensorStack(*[[{"role": "user", "content": f"Q{i}?"}] for i in range(len(uids))]),
            "reward_model": NonTensorStack(*[{"ground_truth": "7"} for _ in uids]),
            "extra_info": NonTensorStack(*[{"solution": "work it out"} for _ in uids]),
        },
        batch_size=len(uids),
    )
    meta = tq.kv_batch_put(keys, "train", fields)
    return meta


def read_adv(tq, keys):
    out = tq.kv_batch_get(keys, "train", ["advantages", "returns"])
    return [t.clone() for t in out["advantages"].unbind()], [t.clone() for t in out["returns"].unbind()]


def test_rlsd_amplifies_positive_and_shrinks_negative(monkeypatch):
    t, tq = build_trainer(
        monkeypatch,
        dict(method="rlsd", positive_source="gt", eps_w=0.5, lam_start=1.0, lam_end=1.0, lam_decay_steps=0),
    )
    uids = ["g", "g", "h", "h"]
    batch = populate(tq, rewards=[1, 0, 1, 1], advs=[1.0, -1.0, 0.0, 0.0], uids=uids)
    metrics = {}
    t._apply_selfdistill(batch, metrics)

    adv, returns = read_adv(tq, batch.keys)
    # hint makes teacher logp > student logp  => delta > 0
    assert torch.all(adv[0] > 1.0)  # positive advantage amplified
    assert torch.all(adv[1] > -1.0) and torch.all(adv[1] < 0.0)  # negative blame attenuated, sign kept
    assert torch.all(adv[2] == 0) and torch.all(adv[3] == 0)  # uniform groups untouched
    assert all(torch.equal(a, r) for a, r in zip(adv, returns, strict=True))
    assert [len(a) for a in adv] == [3, 4, 3, 4]
    assert metrics["selfdistill/hint_coverage"] == 1.0
    assert not [k for k in tq.store if k[0] == "rlx_selfdistill"], "scratch partition must be cleared"
    assert len(t.actor_rollout_wg.calls) == 1


def test_rlcsd_runs_k_plus_one_passes_and_keeps_sign(monkeypatch):
    t, tq = build_trainer(monkeypatch, dict(method="rlcsd", positive_source="gt", n_neg=2, teacher_mode="actor"))
    uids = ["g"] * 4
    batch = populate(tq, rewards=[1, 0, 0, 1], advs=[0.9, -0.9, -0.9, 0.9], uids=uids)
    metrics = {}
    t._apply_selfdistill(batch, metrics)

    assert len(t.actor_rollout_wg.calls) == 3  # 1 positive + K=2 negative slots
    adv, _ = read_adv(tq, batch.keys)
    signs = [0.9, -0.9, -0.9, 0.9]
    for a, s in zip(adv, signs, strict=True):
        assert torch.all(a * s >= 0)
    assert "selfdistill/mod_token_frac" in metrics
    assert not [k for k in tq.store if k[0] == "rlx_selfdistill"]


def test_rlcsd_ref_teacher_uses_ref_worker(monkeypatch):
    t, tq = build_trainer(monkeypatch, dict(method="rlcsd", n_neg=1, teacher_mode="ref"))
    batch = populate(tq, rewards=[1, 0], advs=[1.0, -1.0], uids=["g", "g"])
    t._apply_selfdistill(batch, {})
    assert len(t.ref_policy_wg.calls) == 2 and len(t.actor_rollout_wg.calls) == 0


def test_opsd_advantage_is_teacher_gap_only(monkeypatch):
    t, tq = build_trainer(monkeypatch, dict(method="opsd", positive_source="gt", opsd_clip=5.0))
    batch = populate(tq, rewards=[0, 0], advs=[0.0, 0.0], uids=["g", "g"])
    t._apply_selfdistill(batch, {})
    adv, _ = read_adv(tq, batch.keys)
    assert all(torch.all(a > 0) for a in adv)  # reward-free: positive gap even with zero outcome advantage


def test_rollout_hints_without_correct_sibling_leave_advantage_untouched(monkeypatch):
    t, tq = build_trainer(monkeypatch, dict(method="rlsd", positive_source="rollout", eps_w=0.5, lam_start=1.0))
    batch = populate(tq, rewards=[0, 0, 1, 0], advs=[0.0, 0.0, 1.5, -0.5], uids=["a", "a", "b", "b"])
    before = [t_.clone() for t_ in tq.kv_batch_get(batch.keys, "train", ["advantages"])["advantages"].unbind()]
    metrics = {}
    t._apply_selfdistill(batch, metrics)
    after, _ = read_adv(tq, batch.keys)
    # rows 2 (only correct in its group) has no correct sibling -> unchanged
    assert torch.equal(after[2], before[2])
    # row 3 gets row 2 as positive hint -> modulated
    assert not torch.equal(after[3], before[3]) and torch.all(after[3] <= 0)
    assert metrics["selfdistill/hint_coverage"] == pytest.approx(0.25)


def test_trainer_mode_is_pinned_to_sync():
    t = object.__new__(trainer_mod.SelfDistillTrainer)
    t.trainer_mode = "selfdistill_sync"
    assert t.trainer_mode == "sync"
