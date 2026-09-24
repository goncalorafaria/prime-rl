"""Batch-local group reduction across packing, truncation, DP and CP replicas."""

import msgspec
import pytest
import torch

from prime_rl.configs.trainer import AdamWConfig, IcePopLossConfig, IPOLossConfig
from prime_rl.trainer.batch import build_bin_cost, prepare_batch
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.transports.batch.types import MicroBatch, TrainingSample


@pytest.mark.parametrize("loss_config", [IPOLossConfig(), IcePopLossConfig()])
@pytest.mark.parametrize("cp_size", [1, 4])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("group_mean", [False, True])
def test_group_mean_packing_and_distributed_gradients(loss_config, cp_size, workers, group_mean):
    def sample(group, size, advantage, weights=None, ce=None):
        return TrainingSample(
            token_ids=[1] * size,
            mask=[False] + [True] * (size - 1),
            logprobs=[-0.7] * size,
            temperatures=[1.0] * size,
            env_name="rubrics",
            advantages=[advantage] * size,
            group_id=group,
            rl_weights=weights,
            ce_weights=ce,
        )

    samples = [
        sample("a", 7, 1.0, [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        sample("a", 3, 3.0, [0.0, 2.0, 2.0]),
        sample("b", 4, -2.0),
        sample("empty", 2, 0.0, [0.0, 0.0], [0.0, 1.0]),
    ]
    grid = prepare_batch(samples, 5, workers, build_bin_cost(None), pad_to_multiple_of=cp_size)
    batches = [msgspec.msgpack.decode(msgspec.msgpack.encode(mb), type=MicroBatch) for rank in grid for mb in rank]
    total_tokens = sum(
        sum(m and (mb.rl_weights is None or mb.rl_weights[i] != 0) for i, m in enumerate(mb.loss_mask))
        for mb in batches
    )
    leaves, expected_grads = [], []
    actual = torch.tensor(0.0)
    for mb in batches:
        logp = torch.full((len(mb.input_ids),), -0.7, requires_grad=True)
        leaves.append(logp)
        lengths = mb.sequence_lengths
        mask = torch.tensor(mb.loss_mask)
        adv = torch.tensor(mb.advantages)
        rw = torch.tensor(mb.rl_weights) if mb.rl_weights is not None else None
        cw = torch.tensor(mb.ce_weights) if mb.ce_weights is not None else None
        denom = torch.repeat_interleave(torch.tensor(mb.rl_group_denominators), torch.tensor(lengths))
        if not group_mean:
            denom = total_tokens
        expected_grads.append(-adv * mask * (rw if rw is not None else 1.0) / denom - (cw if cw is not None else 0.0))
        for _ in range(cp_size):
            loss, _ = compute_loss(
                list(logp.split(lengths)),
                list(torch.full_like(logp, -0.7).split(lengths)),
                None,
                list(adv.split(lengths)),
                list(mask.split(lengths)),
                list(rw.split(lengths)) if rw is not None else None,
                list(cw.split(lengths)) if cw is not None else None,
                None,
                setup_rl_loss_fn(loss_config),
                total_tokens * cp_size,
                cp_size,
                1,
                rl_group_denominators=mb.rl_group_denominators if group_mean else None,
                cp_size=cp_size,
            )
            actual = actual + loss
    # Group a: -(3*1 + 2*2*3)/5 = -3; group b: +2. CE contributes .7.
    torch.testing.assert_close(actual, torch.tensor(0.2 if group_mean else -0.425))
    actual.backward()
    for leaf, expected in zip(leaves, expected_grads, strict=True):
        torch.testing.assert_close(leaf.grad, expected)


def test_group_metadata_missing_is_not_invented():
    sample = TrainingSample(
        token_ids=[1, 2],
        mask=[False, True],
        logprobs=[0.0, -1.0],
        temperatures=[1.0, 1.0],
        env_name="test",
        advantages=[0.0, 1.0],
    )
    assert prepare_batch([sample], 8, 1, build_bin_cost(None))[0][0].rl_group_denominators is None


def test_adamw_controls_and_loss_defaults():
    config = AdamWConfig(lr=2e-6, eps=1e-10, betas1=0.95, betas2=0.95, weight_decay=0.0)
    assert config.eps == 1e-10
    assert AdamWConfig().eps == 1e-8
    assert IPOLossConfig().aggregation == "token_mean"
    assert IcePopLossConfig(aggregation="group_token_mean").aggregation == "group_token_mean"
    for eps in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            AdamWConfig(eps=eps)


@pytest.mark.parametrize("loss_config", [IPOLossConfig(), IcePopLossConfig()])
def test_group_denominator_includes_trust_region_rejected_tokens(loss_config):
    logp = torch.tensor([0.0, -1.0], requires_grad=True)
    loss, _ = compute_loss(
        [logp],
        [torch.tensor([-10.0, -1.0])],
        None,
        [torch.ones(2)],
        [torch.ones(2, dtype=torch.bool)],
        None,
        None,
        None,
        setup_rl_loss_fn(loss_config),
        2,
        1,
        1,
        rl_group_denominators=[2],
    )
    torch.testing.assert_close(loss, torch.tensor(-0.5))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([0.0, -0.5]))
