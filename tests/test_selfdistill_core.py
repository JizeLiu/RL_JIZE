import math

import pytest
import torch

from rlx.selfdistill import core


def _setup(b=4, t=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, t)
    mask[1, 4:] = 0
    mask[3, 2:] = 0
    adv = torch.tensor([1.0, -1.0, 0.0, 0.7]).unsqueeze(1).expand(b, t).clone() * mask
    delta = torch.randn(b, t, generator=g) * mask
    return adv, delta, mask


def test_privileged_gain_is_detached():
    s = torch.zeros(2, 3, requires_grad=True)
    d = core.privileged_gain(torch.ones(2, 3), s)
    assert not d.requires_grad


def test_rlsd_never_flips_sign_and_respects_clip():
    adv, delta, mask = _setup()
    delta = delta * 10  # extreme values
    out = core.rlsd_advantage(adv, delta, mask, eps_w=0.2, lam=1.0)
    assert torch.all(out * adv >= 0)
    ratio = out[mask.bool() & (adv != 0)] / adv[mask.bool() & (adv != 0)]
    assert ratio.min() >= 0.8 - 1e-6 and ratio.max() <= 1.2 + 1e-6
    assert torch.all(out[mask == 0] == 0)


def test_rlsd_direction_aware_weights():
    # positive A: tokens the teacher likes (delta>0) get more credit; negative A: reversed.
    mask = torch.ones(2, 1)
    adv = torch.tensor([[1.0], [-1.0]])
    delta = torch.tensor([[0.1], [0.1]])
    out = core.rlsd_advantage(adv, delta, mask, eps_w=0.5, lam=1.0)
    assert out[0, 0] > 1.0  # amplified positive credit
    assert out[1, 0] > -1.0 and out[1, 0] < 0  # attenuated blame (|A| shrinks)


def test_rlsd_lambda_zero_is_grpo():
    adv, delta, mask = _setup()
    out = core.rlsd_advantage(adv, delta, mask, lam=0.0)
    assert torch.allclose(out, adv)


def test_lambda_schedule():
    assert core.lambda_schedule(0, 0.5, 0.0, 50) == 0.5
    assert core.lambda_schedule(25, 0.5, 0.0, 50) == pytest.approx(0.25)
    assert core.lambda_schedule(500, 0.5, 0.0, 50) == 0.0
    assert core.lambda_schedule(3, 0.5, 0.2, 0) == 0.2


def test_opsd_clip():
    delta = torch.tensor([[10.0, -10.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = core.opsd_advantage(delta, mask, clip=5.0)
    assert out.tolist() == [[5.0, -5.0, 0.0]]


def test_contrastive_signal_matches_logmeanexp():
    b, k, t = 2, 3, 4
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(b, t, generator=g)
    negs = torch.randn(b, k, t, generator=g)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    e = core.contrastive_signal(pos, negs, valid)
    ref0 = pos[0] - torch.log(torch.exp(negs[0]).mean(0))
    ref1 = pos[1] - negs[1, 0]
    assert torch.allclose(e[0], ref0, atol=1e-5)
    assert torch.allclose(e[1], ref1, atol=1e-5)


def test_contrastive_signal_no_negatives_is_zero():
    pos = torch.randn(1, 3)
    negs = torch.randn(1, 2, 3)
    e = core.contrastive_signal(pos, negs, torch.zeros(1, 2, dtype=torch.bool))
    assert torch.all(e == 0)
    assert torch.isfinite(e).all()


def test_rlcsd_sign_preserving_and_inactive_rows_untouched():
    adv, _, mask = _setup()
    e = torch.randn(4, 6) * 5 * mask
    params = core.RLCSDParams(lam=1.0, tau=1.0, delta=0.05, eta=1.0, two_path=False)
    out, stats = core.rlcsd_advantage(adv, e, mask, params)
    assert torch.all(out * adv >= 0)
    assert torch.all(out[2] == 0)  # uniform-outcome row (A == 0) stays zero
    assert 0.0 <= stats["mod_token_frac"] <= 1.0


def test_rlcsd_mask_threshold_selects_tokens():
    mask = torch.ones(1, 4)
    adv = torch.ones(1, 4)
    e = torch.tensor([[0.0, 0.001, 3.0, -3.0]])
    params = core.RLCSDParams(lam=0.5, tau=1.0, delta=0.1, eta=1.0, two_path=False)
    out, stats = core.rlcsd_advantage(adv, e, mask, params)
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0  # below threshold: unchanged
    assert out[0, 2] == pytest.approx(1.0 + 0.5 * math.tanh(3.0))
    assert out[0, 3] == pytest.approx(1.0 - 0.5 * math.tanh(3.0))
    assert stats["mod_token_frac"] == pytest.approx(0.5)


def test_rlcsd_negative_advantage_clamped_at_zero():
    mask = torch.ones(1, 1)
    adv = torch.tensor([[-0.1]])
    e = torch.tensor([[50.0]])  # r ~ +lam: would flip the sign without the clamp
    out, _ = core.rlcsd_advantage(adv, e, mask, core.RLCSDParams(lam=0.5, tau=1.0, delta=0.01, two_path=False))
    assert out.item() == 0.0


def test_rlcsd_two_path_equals_paper_objective():
    """Token-mean of the weighted advantage == paper's per-rollout two-path average (rho == 1)."""
    g = torch.Generator().manual_seed(3)
    b, t = 3, 8
    mask = torch.ones(b, t)
    adv = torch.tensor([1.0, -0.5, 0.8]).unsqueeze(1).expand(b, t).clone()
    e = torch.randn(b, t, generator=g) * 2
    params = core.RLCSDParams(lam=0.5, tau=1.3, delta=0.2, eta=0.5, two_path=True)
    out, _ = core.rlcsd_advantage(adv, e, mask, params)

    r = params.lam * torch.tanh(e / params.tau)
    m = r.abs() > params.delta
    shifted = adv + r
    tilde = torch.where(adv >= 0, shifted.clamp(min=0), shifted.clamp(max=0))
    paper = 0.0
    for i in range(b):
        u, mm = ~m[i], m[i]
        j = 0.0
        if u.any():
            j += adv[i][u].mean().item()
        if mm.any():
            j += params.eta * tilde[i][mm].mean().item()
        paper += j / b
    ours = out.sum().item() / mask.sum().item()
    assert ours == pytest.approx(paper, rel=1e-5)
