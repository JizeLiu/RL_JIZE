"""Pure-torch math for the self-distillation family (OPSD / RLSD / RLCSD).

Everything here operates on padded ``(B, T)`` tensors and has no dependency on VERL, so it can be
unit-tested on CPU. The trainer plugin (``rlx.selfdistill.trainer``) is the only place that talks
to VERL; it feeds these functions and writes the result back as the ``advantages`` field, which
the stock VERL policy loss (``vanilla`` PPO clip) then consumes unchanged.

Notation (see arXiv 2604.03128 for RLSD, arXiv 2606.11709 for RLCSD):
    A       sequence-level outcome advantage (GRPO), broadcast over tokens, shape (B, T)
    logp_T  log pi(y_t | x, hint, y_<t)  -- same weights, privileged context
    logp_S  log pi(y_t | x, y_<t)        -- same weights, plain context
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EXP_CLAMP = 20.0
_ROW_EPS = 1e-8


def privileged_gain(logp_teacher: torch.Tensor, logp_student: torch.Tensor) -> torch.Tensor:
    """Delta_t = sg(log P_T(y_t) - log P_S(y_t)). Always detached."""
    return (logp_teacher - logp_student).detach()


def lambda_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linear interpolation of the RLSD mixing coefficient from ``start`` to ``end``."""
    if decay_steps <= 0:
        return end
    frac = min(max(step / decay_steps, 0.0), 1.0)
    return start + (end - start) * frac


def opsd_advantage(delta: torch.Tensor, mask: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    """On-policy self-distillation as a token-level advantage.

    Minimising the sampled-token reverse KL via the log-derivative trick gives
    ``A_t = log P_T(y_t) - log P_S(y_t)``. The verifier reward plays no role. ``clip`` applies the
    pointwise clipping used by OPSD to tame style-token outliers.
    """
    adv = delta.clone()
    if clip is not None and clip > 0:
        adv = adv.clamp(-clip, clip)
    return adv * mask


def rlsd_advantage(
    adv: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    eps_w: float = 0.2,
    lam: float = 1.0,
) -> torch.Tensor:
    """RLSD: the verifier decides the sign, the privileged teacher decides the magnitude.

    ``w_t = exp(sign(A) * Delta_t)`` is clipped to ``[1-eps_w, 1+eps_w]`` and mixed with the
    uniform advantage: ``A_t = A * ((1-lam) + lam * w_t)``. Since ``w_t > 0`` the sign of ``A`` can
    never be flipped.
    """
    sign = torch.sign(adv)
    w = torch.exp((sign * delta).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    w = w.clamp(1.0 - eps_w, 1.0 + eps_w)
    return adv * ((1.0 - lam) + lam * w) * mask


def contrastive_signal(
    logp_pos: torch.Tensor,
    logp_negs: torch.Tensor,
    neg_valid: torch.Tensor,
) -> torch.Tensor:
    """e_ctr,t = log P_T(.|pos hint) - log mean_k P_T(.|neg hint_k)  (RLCSD Eq. 7).

    Args:
        logp_pos: (B, T) teacher log-probs under the positive hint.
        logp_negs: (B, K, T) teacher log-probs under K negative hints.
        neg_valid: (B, K) bool, False where a negative slot is padding.

    Rows with no valid negative hint get ``e_ctr = 0``.
    """
    neg = logp_negs.masked_fill(~neg_valid.unsqueeze(-1), float("-inf"))
    count = neg_valid.sum(dim=1).clamp(min=1).to(logp_pos.dtype).unsqueeze(-1)  # (B, 1)
    log_mean_neg = torch.logsumexp(neg, dim=1) - torch.log(count)
    e = logp_pos - log_mean_neg
    has_neg = neg_valid.any(dim=1, keepdim=True)
    return torch.where(has_neg, e, torch.zeros_like(e)).detach()


@dataclass
class RLCSDParams:
    lam: float = 0.5  # lambda: scale of r_t
    tau: float = 1.0  # tanh temperature
    delta: float = 0.1  # token mask threshold on |r_t|
    eta: float = 1.0  # weight of the modulated path
    two_path: bool = True  # fold the two-path loss aggregation into per-token advantage weights
    max_path_weight: float = 20.0  # numerical guard for tiny token sets in two-path weighting


def rlcsd_advantage(
    adv: torch.Tensor,
    e_ctr: torch.Tensor,
    mask: torch.Tensor,
    params: RLCSDParams,
) -> tuple[torch.Tensor, dict[str, float]]:
    """RLCSD: verifier-anchored, contrastively modulated advantage (Eqs. 9-15).

    1. ``r_t = lam * tanh(e_ctr / tau)``, ``m_t = |r_t| > delta``.
    2. Modulated tokens get ``max(0, A + r_t)`` (A >= 0) or ``min(0, A + r_t)`` (A < 0), so the
       verifier direction is never reversed.
    3. Two-path aggregation. The paper averages unmodulated and modulated tokens separately per
       rollout and weights the modulated path by ``eta``. The PPO-clip surrogate is positively
       homogeneous in the advantage, so under ``token-mean`` loss aggregation this is *exactly*
       equivalent to scaling each token's advantage by ``Lbar/|U_g|`` (unmodulated) or
       ``eta*Lbar/|M_g|`` (modulated), where ``Lbar`` is the mean response length. That keeps the
       stock VERL ``vanilla`` loss usable with no custom loss function.

    Rows whose outcome advantage is zero (uniform-outcome groups) are left untouched; the paper
    discards those groups.

    Returns the final advantage and a dict of scalar diagnostics.
    """
    maskf = mask.to(adv.dtype)
    r = params.lam * torch.tanh(e_ctr / params.tau)
    row_active = (adv.abs() * maskf).sum(dim=1, keepdim=True) > _ROW_EPS  # (B, 1)
    m = (r.abs() > params.delta) & mask.bool() & row_active

    shifted = adv + r
    clamped = torch.where(adv >= 0, shifted.clamp(min=0.0), shifted.clamp(max=0.0))
    a_mod = torch.where(m, clamped, adv)

    if params.two_path:
        n_total = maskf.sum(dim=1, keepdim=True)  # |y_g|
        n_mod = (m.to(adv.dtype) * maskf).sum(dim=1, keepdim=True)
        n_unmod = n_total - n_mod
        l_bar = maskf.sum() / max(float((n_total > 0).sum().item()), 1.0)
        w_unmod = (l_bar / n_unmod.clamp(min=1.0)).clamp(max=params.max_path_weight)
        w_mod = (params.eta * l_bar / n_mod.clamp(min=1.0)).clamp(max=params.max_path_weight)
        weight = torch.where(m, w_mod, w_unmod)
        out = torch.where(row_active, a_mod * weight, adv)
    else:
        out = a_mod
    out = out * maskf

    active_tokens = (maskf * row_active.to(adv.dtype)).sum().clamp(min=1.0)
    stats = {
        "mod_token_frac": float((m.to(adv.dtype) * maskf).sum() / active_tokens),
        "e_ctr_abs_mean": float((e_ctr.abs() * maskf * row_active.to(adv.dtype)).sum() / active_tokens),
        "r_abs_mean": float((r.abs() * maskf * row_active.to(adv.dtype)).sum() / active_tokens),
    }
    return out, stats


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.to(x.dtype)
    denom = m.sum()
    if denom.item() == 0:
        return math.nan
    return float((x * m).sum() / denom)
