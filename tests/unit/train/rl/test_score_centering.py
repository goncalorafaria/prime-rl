"""Score centering against explicit small-vocabulary sampler expectations."""

import pytest
import torch

from prime_rl.configs.trainer import IcePopLossConfig, IPOLossConfig, TrainerConfig
from prime_rl.trainer.rl.loss import LossInputs, setup_rl_loss_fn
from prime_rl.trainer.rl.score_centering import masked_score_correction


@pytest.mark.parametrize("loss_type", ["ipo", "icepop"])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("advantage", [-2.0, 3.0])
def test_centered_score_has_zero_sampler_expectation(loss_type, truncated, advantage):
    config = (
        IPOLossConfig(eps=0.12, adv_tau=0.7)
        if loss_type == "ipo"
        else IcePopLossConfig(ratio_low=0.5, ratio_high=1.5, adv_tau=0.7)
    )
    q = torch.tensor([0.15, 0.35, 0.20, 0.30]) if not truncated else torch.tensor([0.20, 0.0, 0.30, 0.50])
    support = (q > 0).nonzero().flatten()
    logits = torch.tensor([0.2, -0.4, 0.7, -0.1], requires_grad=True)
    repeated = logits.expand(len(support), -1)
    mask = support.expand(len(support), -1) if truncated else None
    lp = (logits.masked_fill(q == 0, -torch.inf) if truncated else logits).log_softmax(-1)
    correction = masked_score_correction(
        repeated,
        support.expand(len(support), -1),
        q[support].log().expand(len(support), -1),
        support,
        torch.ones(len(support), dtype=torch.bool),
        config,
        sampling_mask=mask,
        chunk_size=2,
    )
    inputs = LossInputs(
        lp[support],
        q[support].log(),
        None,
        torch.full((len(support),), advantage),
        torch.ones(len(support), dtype=torch.bool),
        loss_weights=q[support],
        score_correction=correction,
    )
    baseline = setup_rl_loss_fn(config).loss(inputs).loss
    centered = setup_rl_loss_fn(config.model_copy(update={"score_centering": True})).loss(inputs).loss
    torch.testing.assert_close(centered, baseline)
    baseline_grad = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
    centered_grad = torch.autograd.grad(centered, logits)[0]
    assert baseline_grad.abs().max() > 1e-3
    torch.testing.assert_close(centered_grad, torch.zeros_like(logits), atol=2e-7, rtol=0)


def test_partial_head_tail_and_context_mask():
    logits = torch.tensor([[0.1, 0.2, -0.3], [0.4, 0.1, -0.2]], requires_grad=True)
    ids = torch.tensor([[0, -1], [-1, -1]])
    hq = torch.tensor([[torch.log(torch.tensor(0.8)), 0.0], [0.0, 0.0]])
    config = IPOLossConfig(eps=0.15)
    correction = masked_score_correction(
        logits, ids, hq, torch.tensor([0, 1]), torch.tensor([True, False]), config, chunk_size=1
    )
    lp = logits[0].log_softmax(-1)
    p = lp.detach().exp()
    q = torch.cat([torch.tensor([0.8]), 0.2 * p[1:] / p[1:].sum()])
    expected = (p * ((p - q).abs() <= config.eps) * lp).sum()
    torch.testing.assert_close(correction[0], expected)
    assert correction[1].item() == 0.0
    actual_grad = torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_centering_preserves_ipo_kl_gradient_and_rejects_missing_heads():
    config = IPOLossConfig(eps=0.2, kl_tau=0.03, score_centering=True)
    z = torch.tensor([[0.2, -0.7, 0.4]], requires_grad=True)
    q = torch.tensor([[0.6, 0.1, 0.3]])
    ids = torch.tensor([[0, 1, 2]])
    active = torch.tensor([True])
    correction = masked_score_correction(z, ids, q.log(), torch.tensor([0]), active, config)
    inputs = LossInputs(
        z.log_softmax(-1)[:, 0], q[:, 0].log(), None, torch.tensor([0.0]), active, score_correction=correction
    )
    actual = setup_rl_loss_fn(config).loss(inputs).loss
    expected = 0.03 * (inputs.trainer_logprobs - inputs.inference_logprobs).square().sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, retain_graph=True)[0], torch.autograd.grad(expected, z)[0]
    )
    inputs.score_correction = None
    with pytest.raises(ValueError, match="requires"):
        setup_rl_loss_fn(config).loss(inputs)
    with pytest.raises(ValueError, match="Missing sampler"):
        masked_score_correction(z, torch.full_like(ids, -1), q.log(), torch.tensor([0]), active, config)
    with pytest.raises(ValueError, match="outside"):
        masked_score_correction(
            z, ids, q.log(), torch.tensor([0]), active, config, sampling_mask=torch.tensor([[0, 1]])
        )


def test_centering_config_is_optional_and_requires_unfused_head():
    assert not IPOLossConfig().score_centering
    assert not IcePopLossConfig().score_centering
    with pytest.raises(ValueError, match="fused_lm_head"):
        TrainerConfig(loss={"type": "ipo", "score_centering": True})
    config = TrainerConfig(
        loss={"type": "ipo", "score_centering": True, "aggregation": "group_token_mean"},
        model={"fused_lm_head_token_chunk_size": "disabled"},
    )
    assert config.loss.score_centering


def test_sampler_head_survives_trace_packing_and_tensorization():
    import math

    import msgspec
    import verifiers.v1 as vf
    from renderers.client import _parse_sampler_head
    from verifiers.v1.clients.train import response_from_generate
    from verifiers.v1.graph import prepare_turn

    from prime_rl.orchestrator.trajectories import trace_to_samples
    from prime_rl.trainer.batch import build_bin_cost, prepare_batch
    from prime_rl.trainer.rl.data import DataLoader
    from prime_rl.transports.batch.types import MicroBatch

    choice = {
        "logprobs": {
            "content": [
                {
                    "token": "token_id:20",
                    "logprob": math.log(0.4),
                    "top_logprobs": [{"token": "token_id:22", "logprob": math.log(0.6)}],
                },
                {
                    "token": "token_id:21",
                    "logprob": math.log(0.25),
                    "top_logprobs": [{"token": "token_id:22", "logprob": math.log(0.75)}],
                },
            ]
        }
    }
    head_ids, head_lps = _parse_sampler_head(choice, [20, 21])
    assert head_ids == [[22, 20], [22, 21]]
    response = response_from_generate(
        {
            "prompt_ids": [10, 11, 12],
            "completion_ids": [20, 21],
            "completion_logprobs": [math.log(0.4), math.log(0.25)],
            "sampler_head_ids": head_ids,
            "sampler_head_logprobs": head_lps,
            "finish_reason": "stop",
            "content": "answer",
        },
        model="test",
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()), task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x"))
    )
    prepare_turn(trace, [vf.UserMessage(content="question")]).commit(response)
    trace.nodes[-1].advantages = [1.0, -1.0]
    # Exercise the real wire representation, not a hand-constructed Branch.
    trace = vf.Trace.model_validate(msgspec.msgpack.decode(msgspec.msgpack.encode(trace.model_dump(mode="python"))))
    branch = trace.branches[0]
    assert branch.sampler_head_ids == [[], [], [], [22, 20], [22, 21]]
    sample = trace_to_samples(trace, env_name="rubrics")[0]
    sample.group_id = "group"
    sample.temperatures = [1.0] * len(sample.token_ids)
    # Truncate one sampled token, then pad back to four tokens.
    packed = prepare_batch([sample], 4, 1, build_bin_cost(None), pad_to_multiple_of=4)[0][0]
    packed = msgspec.msgpack.decode(msgspec.msgpack.encode(packed), type=MicroBatch)
    tensors = object.__new__(DataLoader)._micro_batch_to_tensor(packed)
    assert tensors["sampler_head_ids"].tolist() == [[[-1, -1], [-1, -1], [-1, -1], [22, 20]]]
    torch.testing.assert_close(tensors["sampler_head_logprobs"][0, -1], torch.tensor(head_lps[0]))
    assert tensors["rl_group_denominators"] == [1]
    bad = choice.copy()
    bad["logprobs"] = {"content": [{"top_logprobs": []}, {"top_logprobs": []}]}
    with pytest.raises(ValueError):
        _parse_sampler_head(bad, [20, 21])


def test_score_correction_next_token_alignment_across_cp_shards():
    from prime_rl.trainer.rl.loss import shift_tensor_left, shift_tensor_right

    logits = torch.linspace(-1.0, 1.0, 32).reshape(1, 8, 4).requires_grad_()
    ids = torch.arange(4).expand(1, 8, 4).clone()
    q = torch.tensor([0.1, 0.2, 0.3, 0.4]).log().expand_as(logits)
    active = torch.tensor([[False, True, True, False, False, True, False, False]])
    ids[~active] = -1
    labels = shift_tensor_left(torch.arange(8).reshape(1, 8) % 4)
    args = [shift_tensor_left(ids, -1), shift_tensor_left(q), labels, shift_tensor_left(active)]
    config = IPOLossConfig(eps=0.15, score_centering=True)
    full = shift_tensor_right(masked_score_correction(logits, *args, config))
    shards = []
    for rank in range(4):
        sl = slice(rank * 2, rank * 2 + 2)
        shards.append(masked_score_correction(logits[:, sl], *(a[:, sl] for a in args), config))
    gathered = shift_tensor_right(torch.cat(shards, dim=1))
    torch.testing.assert_close(full, gathered)
    torch.testing.assert_close(
        torch.autograd.grad(full.sum(), logits, retain_graph=True)[0], torch.autograd.grad(gathered.sum(), logits)[0]
    )
    assert not full[~active].any()
