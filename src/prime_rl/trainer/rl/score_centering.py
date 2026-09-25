"""Sampler-expectation correction for upstream IPO and IcePop policy gradients."""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from prime_rl.configs.trainer import IcePopLossConfig, IPOLossConfig
from prime_rl.trainer.models.layers.lm_head import sampling_replay_mask


def masked_score_correction(
    logits: Tensor,
    head_ids: Tensor,
    head_logprobs: Tensor,
    labels: Tensor,
    active: Tensor,
    config: IPOLossConfig | IcePopLossConfig,
    sampling_mask: Tensor | None = None,
    chunk_size: int = 128,
) -> Tensor:
    """Gradient surrogate E_qhat[keep(a) * p(a)/qhat(a) * grad log p(a)].

    Inputs are label-aligned and logits are temperature-scaled. Replayed sampling
    masks constrain p and the approximated q tail to the same support. Logged
    sampler probabilities are exact; the unlogged tail is proportional to detached
    trainer p. All expectation coefficients and loss masks are stop-gradient.
    Chunk checkpointing avoids retaining a second sequence-by-vocabulary graph.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if head_ids.shape != head_logprobs.shape or head_ids.shape[:-1] != active.shape:
        raise ValueError("Sampler heads must align with active token positions")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    ids = head_ids.reshape(-1, head_ids.shape[-1])
    logq = head_logprobs.reshape_as(ids)
    labels = labels.reshape(-1)
    masks = sampling_mask.reshape(-1, sampling_mask.shape[-1]) if sampling_mask is not None else None
    indices = active.reshape(-1).nonzero().flatten()
    correction = flat_logits[:, 0] * 0.0

    def chunk(z: Tensor, h: Tensor, hq: Tensor, target: Tensor, support_ids: Tensor | None) -> Tensor:
        valid = h >= 0
        if bool((~valid.any(-1)).any()):
            raise ValueError("Missing sampler probabilities on an active RL token")
        if bool((h >= z.shape[-1]).any()):
            raise ValueError("Sampler token id exceeds model vocabulary")
        safe_ids = h.clamp_min(0)
        support = torch.ones_like(z, dtype=torch.bool)
        if support_ids is not None:
            # Match upstream replay eligibility, including its full-vocab fallback.
            replay = sampling_replay_mask(support_ids, target)
            counts = torch.zeros_like(z, dtype=torch.int32).scatter_add_(
                -1, support_ids.clamp_min(0).long(), (support_ids >= 0).int()
            )
            support = (counts > 0) | ~replay.unsqueeze(-1)
        if bool((valid & ~support.gather(-1, safe_ids)).any()):
            raise ValueError("Sampler head falls outside the replayed sampling support")
        lp = z.float().masked_fill(~support, -torch.inf).log_softmax(-1)
        safe_lp = torch.where(support, lp, 0.0)
        with torch.no_grad():
            p = lp.exp()
            qhead = torch.where(valid, hq, 0.0).exp() * valid
            in_head = torch.zeros_like(z, dtype=torch.int32).scatter_add_(-1, safe_ids, valid.int()) > 0
            tail = p.masked_fill(in_head, 0.0)
            mass = tail.sum(-1, keepdim=True)
            missing_mass = (1.0 - qhead.sum(-1, keepdim=True)).clamp_min(0.0)
            if bool(((mass == 0) & (missing_mass > 1e-5)).any()):
                raise ValueError("Sampler head mass is incomplete but no supported tail remains")
            q = tail * (missing_mass / mass.clamp_min(torch.finfo(p.dtype).tiny))
            q.scatter_add_(-1, safe_ids, qhead)
            if isinstance(config, IPOLossConfig):
                keep = (p - q).abs() <= config.eps
            elif isinstance(config, IcePopLossConfig):
                log_ratio = safe_lp - q.clamp_min(torch.finfo(p.dtype).tiny).log()
                keep = (log_ratio >= p.new_tensor(config.ratio_low).log()) & (
                    log_ratio <= p.new_tensor(config.ratio_high).log()
                )
            else:
                raise TypeError("Score centering supports IPO and IcePop")
            coefficients = p * (keep & (q > 0) & support)
        return (coefficients * safe_lp).sum(-1)

    for rows in indices.split(chunk_size):
        if rows.numel() == 0:
            continue
        values = checkpoint(
            chunk,
            flat_logits[rows],
            ids[rows],
            logq[rows],
            labels[rows],
            masks[rows] if masks is not None else None,
            use_reentrant=False,
        )
        correction = correction.index_copy(0, rows, values)
    return correction.reshape_as(active)
