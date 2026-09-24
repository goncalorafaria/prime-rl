"""Detached top-k score centering (arXiv:2609.20807, equations 7–12)."""

import torch
from torch import Tensor


def importance_weights(log_ratio: Tensor, weighting: str, cap: float, low: float, high: float) -> Tensor:
    if weighting == "none":
        return torch.ones_like(log_ratio)
    if weighting == "is":
        return log_ratio.exp()
    if weighting == "tis":
        return torch.exp(torch.minimum(log_ratio, log_ratio.new_tensor(cap).log()))
    if weighting == "mis":
        keep = (log_ratio >= log_ratio.new_tensor(low).log()) & (log_ratio <= log_ratio.new_tensor(high).log())
        return torch.where(keep, log_ratio, torch.zeros_like(log_ratio)).exp() * keep
    raise ValueError(f"Unknown score-centering weighting: {weighting}")


def centered_logprob(
    trainer_logprobs: Tensor,
    sampler_logprobs: Tensor,
    trainer_head_logprobs: Tensor,
    sampler_head_logprobs: Tensor,
    head_mask: Tensor,
    *,
    weighting: str = "none",
    cap: float = 2.0,
    low: float = 0.2,
    high: float = 5.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a surrogate whose gradient is the centered, optionally weighted score.

    Inputs have shapes [...], [..., k]. Missing head entries are masked. Sampler
    probabilities must describe the actual temperature/filtered behavior policy.
    The sampler tail is approximated by rescaled trainer probabilities. Rewards,
    loss masks, and component normalization belong to the caller.
    """
    # Mask before arithmetic: -inf padding must never enter a 0 * logp product.
    head_logp = torch.where(head_mask, trainer_head_logprobs, 0.0)
    head_logq = torch.where(head_mask, sampler_head_logprobs, 0.0)
    with torch.no_grad():
        p = head_logp.exp() * head_mask
        q = head_logq.exp() * head_mask
        p_tail = (1.0 - p.sum(-1)).clamp_min(eps)
        q_tail = (1.0 - q.sum(-1)).clamp_min(0.0)
        rho = q_tail / p_tail
        if weighting == "none":
            tail_scale = rho
        elif weighting == "tis":
            tail_scale = torch.minimum(torch.ones_like(rho), cap * rho)
        elif weighting == "mis":
            tail_scale = ((rho >= 1.0 / high) & (rho <= 1.0 / low)).to(p.dtype)
        elif weighting == "is":
            # A zero-mass sampler tail can arise from top-p/top-k filtering.
            tail_scale = (q_tail > 0).to(p.dtype)
        else:
            raise ValueError(f"Unknown score-centering weighting: {weighting}")
        weights = importance_weights(trainer_logprobs - sampler_logprobs, weighting, cap, low, high)
        # Compute q*w stably even when q is tiny or exactly zero after filtering.
        if weighting == "none":
            qw = q
        elif weighting == "tis":
            qw = torch.minimum(p, cap * q)
        elif weighting == "mis":
            delta = head_logp - head_logq
            keep = (delta >= p.new_tensor(low).log()) & (delta <= p.new_tensor(high).log())
            qw = p * keep
        else:
            qw = torch.where(torch.isfinite(head_logq), p, 0.0)
        coefficients = (qw - tail_scale.unsqueeze(-1) * p) * head_mask
    correction = (coefficients * head_logp).sum(-1)
    return weights * trainer_logprobs - correction, {
        "sc/head_mass": q.sum(-1),
        "sc/correction": correction.detach(),
        "sc/weight": weights,
    }


def dppo_score_correction(
    logits: Tensor,
    head_ids: Tensor,
    head_logq: Tensor,
    advantages: Tensor,
    active: Tensor,
    *,
    low: float,
    high: float,
    chunk_size: int = 128,
) -> Tensor:
    """Surrogate whose gradient is E_qhat[m(A, a) (p/qhat) grad log p].

    Reconstruct the unlogged sampler tail proportional to detached trainer p.
    Evaluate the absolute-probability DPPO mask over the entire vocabulary:
    unlike ratio clipping, its tail mask is not constant. The advantage sign
    is held fixed for this expectation. Only active RL tokens are evaluated.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    ids = head_ids.reshape(-1, head_ids.shape[-1])
    logq = head_logq.reshape_as(ids)
    adv = advantages.reshape(-1)
    indices = active.reshape(-1).nonzero().flatten()
    correction = flat_logits[:, 0] * 0.0
    for rows in indices.split(chunk_size):
        if rows.numel() == 0:
            continue
        lp = flat_logits[rows].log_softmax(-1)
        with torch.no_grad():
            valid = ids[rows] >= 0
            safe_ids = ids[rows].clamp_min(0)
            qhead = torch.where(valid, logq[rows], 0.0).exp() * valid
            p = lp.exp()
            in_head = torch.zeros_like(p).scatter_add(-1, safe_ids, valid.to(p.dtype)) > 0
            p_tail = p.masked_fill(in_head, 0.0)
            tail_mass = p_tail.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
            q = p_tail * ((1.0 - qhead.sum(-1, keepdim=True)).clamp_min(0.0) / tail_mass)
            q.scatter_add_(-1, safe_ids, qhead)
            diff = p - q
            invalid = torch.where(adv[rows, None] > 0, diff > high, diff < -low)
            # q * (p/q) = p only on sampler support.
            coefficients = p * (~invalid & (q > 0))
        correction = correction.index_copy(0, rows, (coefficients * lp).sum(-1))
    return correction.reshape_as(advantages)
