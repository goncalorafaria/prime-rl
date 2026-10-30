"""Sampler-expectation correction for upstream IPO and IcePop policy gradients."""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from prime_rl.configs.trainer import IcePopLossConfig, IPOLossConfig
from prime_rl.trainer.models.layers.lm_head import sampling_replay_mask


def icepop_head_correction(
    head_logp: Tensor, head_logq: Tensor, valid: Tensor, config: IcePopLossConfig, eps: float = 1e-6
) -> Tensor:
    """Jasper's detached head residual with IcePop's inclusive ratio mask.

    The omitted sampler tail is proportional to the trainer tail. Its constant
    importance ratio makes alpha = q_tail/p_tail * w(p_tail/q_tail) either 0 or 1.
    No tail probabilities or normalized accepted-weight average are constructed.
    """
    safe_lp = torch.where(valid, head_logp, 0.0)
    with torch.no_grad():
        p = safe_lp.exp() * valid
        q = torch.where(valid, head_logq, 0.0).exp() * valid
        low = p.new_tensor(config.ratio_low).log()
        high = p.new_tensor(config.ratio_high).log()
        log_ratio = safe_lp - head_logq
        keep = valid & (log_ratio >= low) & (log_ratio <= high)
        p_tail = (1.0 - p.sum(-1, keepdim=True)).clamp_min(eps)
        q_tail = (1.0 - q.sum(-1, keepdim=True)).clamp_min(eps)
        tail_log_ratio = p_tail.log() - q_tail.log()
        alpha = ((tail_log_ratio >= low) & (tail_log_ratio <= high)).to(p.dtype)
        # q * MIS(p/q) = p * keep; avoids overflowing explicit importance ratios.
        residual = p * (keep.to(p.dtype) - alpha)
    return (residual * safe_lp).sum(-1)



def mc_score_correction(logp: Tensor, logq: Tensor, valid: Tensor, config) -> Tensor:
    """IID q probes estimate E_q[keep * p/q * grad log p], with fixed weights."""
    if logp.shape[-1] != 32 or not bool(valid.all()):
        raise ValueError("Every active MC32 position must contain all 32 draws")
    with torch.no_grad():
        delta = logp - logq
        if isinstance(config, IcePopLossConfig):
            keep = (delta >= logp.new_tensor(config.ratio_low).log()) & (
                delta <= logp.new_tensor(config.ratio_high).log())
        elif isinstance(config, IPOLossConfig):
            keep = (logp.exp() - logq.exp()).abs() <= config.eps
        else:
            raise TypeError("MC32 supports IcePop and IPO")
        # Mask in log space before exp so rejected large ratios cannot overflow.
        weights = delta.masked_fill(~keep, -torch.inf).exp()
    return (weights * logp).mean(-1)


def mc_forward_kl(logp: Tensor, logq: Tensor) -> Tensor:
    """Unbiased D_KL(q || p) estimate from IID q probes; may be negative."""
    return (logq.detach() - logp).mean(-1)

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
    IcePop uses a head-only residual; IPO retains the explicit vocabulary mask.
    Chunk checkpointing recomputes normalization during backward.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if head_ids.shape != head_logprobs.shape or head_ids.shape[:-1] != active.shape:
        raise ValueError("Sampler heads must align with active token positions")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    ids = head_ids.reshape(-1, head_ids.shape[-1])
    logq = head_logprobs.reshape_as(ids)
    labels = labels.reshape(-1)
    masks = sampling_mask.reshape(flat_logits.shape[0], sampling_mask.shape[-1]) if sampling_mask is not None else None
    indices = active.reshape(-1).nonzero().flatten()
    correction = flat_logits[:, 0] * 0.0

    def chunk(
        all_logits: Tensor, rows: Tensor, h: Tensor, hq: Tensor, target: Tensor, support_ids: Tensor | None
    ) -> Tensor:
        z = all_logits[rows].float()
        valid = h >= 0
        if bool((~valid.any(-1)).any()):
            raise ValueError("Missing sampler probabilities on an active RL token")
        if bool((h >= z.shape[-1]).any()):
            raise ValueError("Sampler token id exceeds model vocabulary")
        safe_ids = h.clamp_min(0)
        if isinstance(config, IcePopLossConfig) or config.score_centering_estimator == "mc32":
            normalizer = z.logsumexp(-1, keepdim=True)
            if support_ids is not None and support_ids.shape[-1] > 0:
                replay = sampling_replay_mask(support_ids, target)
                support_valid = support_ids >= 0
                supported_logits = z.gather(-1, support_ids.clamp_min(0).long())
                # Fallback rows need a finite normalization even if their mask is empty.
                supported_logits = torch.where(
                    replay.unsqueeze(-1), supported_logits.masked_fill(~support_valid, -torch.inf), 0.0
                )
                normalizer = torch.where(replay.unsqueeze(-1), supported_logits.logsumexp(-1, keepdim=True), normalizer)
                sorted_ids = support_ids.sort(-1).values.contiguous()
                locations = torch.searchsorted(sorted_ids, safe_ids.contiguous())
                present = sorted_ids.gather(-1, locations.clamp_max(sorted_ids.shape[-1] - 1)) == safe_ids
                if bool((valid & replay.unsqueeze(-1) & ~present).any()):
                    raise ValueError("Sampler head falls outside the replayed sampling support")
            head_logp = z.gather(-1, safe_ids) - normalizer
            if config.score_centering_estimator == "mc32":
                return mc_score_correction(head_logp, hq, valid, config)
            return icepop_head_correction(head_logp, hq, valid, config)
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
            # Match the FP32 probability accumulator before exponentiation.
            qhead = torch.where(valid, hq.to(p.dtype), 0.0).exp() * valid
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
            flat_logits,
            rows,
            ids[rows],
            logq[rows],
            labels[rows],
            masks[rows] if masks is not None else None,
            use_reentrant=False,
        )
        correction = correction.index_copy(0, rows, values)
    return correction.reshape_as(active)


def chunked_score_centered_lm_head(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    temperature: Tensor,
    head_ids: Tensor | None,
    head_logprobs: Tensor | None,
    active: Tensor,
    config: IPOLossConfig | IcePopLossConfig,
    chunk_size: int,
) -> dict[str, Tensor]:
    """Checkpoint projection and normalization together to bound vocabulary memory.

    Uses the same FP32 normalization and correction as the dense trainer path.
    Only token-sized outputs survive the forward pass; vocabulary logits are
    recomputed one chunk at a time during backward.
    """
    shape = labels.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    temperature = temperature.reshape(-1)
    active = active.reshape(-1)
    if head_ids is not None:
        head_ids = head_ids.reshape(-1, head_ids.shape[-1])
        head_logprobs = head_logprobs.reshape_as(head_ids)

    def project(h, w, target, temp, ids, logq, enabled):
        # The dense model wrapper casts raw logits to FP32 before scaling.
        z = (h @ w.t()).float() / temp.unsqueeze(-1)
        logz = z.logsumexp(-1)
        lp = z.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logz
        with torch.no_grad():
            entropy = logz - (z.softmax(-1) * z).sum(-1)
        correction = z[:, 0] * 0.0
        if ids is not None:
            correction = masked_score_correction(z, ids, logq, target, enabled, config, chunk_size=chunk_size)
        return lp, entropy, correction

    parts = []
    for start in range(0, labels.numel(), chunk_size):
        end = start + chunk_size
        parts.append(checkpoint(
            project, hidden[start:end], weight, labels[start:end], temperature[start:end],
            head_ids[start:end] if head_ids is not None else None,
            head_logprobs[start:end] if head_logprobs is not None else None,
            active[start:end], use_reentrant=False,
        ))
    return {
        key: torch.cat([part[i] for part in parts]).reshape(shape)
        for i, key in enumerate(("logprobs", "entropy", "score_correction"))
    }
