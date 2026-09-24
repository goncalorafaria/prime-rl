import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_rl_loss_fn

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_group_token_mean_balances_groups_with_different_token_counts():
    trainer_logprobs = [torch.zeros(2), torch.full((4,), -0.2)]
    inference_logprobs = [torch.zeros(2), torch.full((4,), -0.3)]
    advantages = [torch.full((2,), -1.0), torch.full((4,), -3.0)]
    loss_mask = [torch.ones(2, dtype=torch.bool), torch.ones(4, dtype=torch.bool)]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.5))
    group_loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=6,
        ce_scale=1,
        ref_kl_scale=1,
        rl_aggregation="group_token_mean",
        rl_group_token_counts=[2, 4],
        rl_num_groups=2,
    )
    token_loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=6,
        ce_scale=1,
        ref_kl_scale=1,
    )

    group_two_per_token = 3.0 * torch.exp(torch.tensor(0.1)) + 0.5 * 0.1**2
    assert torch.isclose(group_loss, (torch.tensor(1.0) + group_two_per_token) / 2)
    assert torch.isclose(token_loss, (2.0 + 4.0 * group_two_per_token) / 6)


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_rl_loss_fn_with_custom_config():
    """Test setup_rl_loss_fn with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    rl_loss_fn = setup_rl_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        ref_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = rl_loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_ce_component_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=2,
        ref_kl_scale=1,
    )

    # loss = -sum(member logprobs) / ce_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_ce_component_applies_weights():
    """ECHO-style observation training: the ce weight stream scales the NLL per token."""
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([0.1, 0.0, 0.1], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    # loss = 0.1 * (0.1 + 0.2) = 0.03
    assert torch.isclose(loss, torch.tensor(0.03, device=loss.device), atol=1e-6)


def test_explicit_rl_weights_match_absent_stream():
    """An explicit all-ones rl stream must equal the rl_weights=None hot path."""
    torch.manual_seed(0)
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda()]
    loss_mask = [torch.rand(50).cuda() > 0.3]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    loss_absent, _ = compute_loss(rl_weights=None, **kwargs)
    loss_explicit, _ = compute_loss(rl_weights=[torch.ones(50, dtype=torch.float32).cuda()], **kwargs)

    assert torch.equal(loss_absent, loss_explicit)


def test_disjoint_components_in_one_sequence():
    """ECHO/OPD-shaped sequence: rl, ce, and ref_kl on disjoint token sets."""
    n = 12
    torch.manual_seed(1)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    rl_weights = torch.zeros(n, dtype=torch.float32)
    rl_weights[:4] = 1.0
    ce_weights = torch.zeros(n, dtype=torch.float32)
    ce_weights[4:8] = 1.0
    ref_kl_weights = torch.zeros(n, dtype=torch.float32)
    ref_kl_weights[8:] = 1.0

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=[rl_weights.cuda()],
        ce_weights=[ce_weights.cuda()],
        ref_kl_weights=[ref_kl_weights.cuda()],
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert loss.shape == ()
    assert "nll" in metrics
    assert "ref_kl" in metrics
    assert "is_masked" in metrics


def test_empty_components_keep_backward_valid():
    """A fully truncated distillation sample (stamped streams survive truncation
    as all-zero prefixes) must train as a zero-gradient no-op, not crash backward."""
    trainer_logprobs = [torch.randn(6, dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(6, dtype=torch.float32).cuda()]
    loss_mask = [torch.zeros(6, dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(6, dtype=torch.float32).cuda()]
    ce_weights = [torch.zeros(6, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert trainer_logprobs[0].grad is not None
    assert torch.equal(trainer_logprobs[0].grad, torch.zeros_like(trainer_logprobs[0].grad))


def test_overlapping_components_sum():
    """Components may overlap on the same token (e.g. RL + a CE behavior-cloning
    regularizer): the total is the sum of each component computed alone, each
    over its own normalization."""
    n = 8
    torch.manual_seed(2)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    ce_weights = [torch.full((n,), 0.5, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=4,
        ce_scale=8,
        ref_kl_scale=1,
    )
    rl_only, _ = compute_loss(rl_weights=None, ce_weights=None, **kwargs)
    ce_only, _ = compute_loss(rl_weights=[torch.zeros(n, dtype=torch.float32).cuda()], ce_weights=ce_weights, **kwargs)
    both, _ = compute_loss(rl_weights=None, ce_weights=ce_weights, **kwargs)

    assert torch.isclose(both, rl_only + ce_only, atol=1e-6)


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )


@pytest.mark.parametrize("score_centering", [False, True])
@pytest.mark.parametrize("kl_tau", [0.0, 0.3])
def test_optional_sc_preserves_group_reduction_and_kl(score_centering, kl_tau):
    from prime_rl.trainer.rl.score_centering import dppo_score_correction

    z = torch.tensor([[0.3, -0.7, 1.2, 0.5]] * 5, dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    p = lp.detach().exp()
    q = torch.tensor([[0.6, 0.15, 0.1, 0.15]] * 5, dtype=z.dtype)
    actions = torch.tensor([2, 0, 0, 2, 3])
    rows = torch.arange(5)
    adv = torch.tensor([1.0, -0.7, -0.5, 1.2, 0.9], dtype=z.dtype)
    weights = torch.tensor([1.0, 0.0, 0.5, 1.0, 2.0], dtype=z.dtype)
    active = weights != 0
    correction = dppo_score_correction(
        z, torch.arange(4).expand(5, -1), q.log(), adv, active, low=0.1, high=0.1
    )
    config = DefaultLossConfig(
        score_centering=score_centering, aggregation="group_token_mean",
        dppo_mask_low=0.1, dppo_mask_high=0.1, adv_tau=0.7, kl_tau=kl_tau,
    )
    loss, _ = compute_loss(
        trainer_logprobs=lp[rows, actions].split([2, 3]),
        inference_logprobs=q.log()[rows, actions].split([2, 3]),
        ref_logprobs=None, advantages=adv.split([2, 3]),
        loss_mask=torch.ones(5, dtype=torch.bool).split([2, 3]),
        rl_weights=weights.split([2, 3]), ce_weights=None, ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(config), rl_scale=4, ce_scale=1, ref_kl_scale=1,
        rl_aggregation=config.aggregation, rl_group_token_counts=[1, 3], rl_num_groups=2,
        score_corrections=correction.split([2, 3]),
    )
    keep = torch.where(adv[:, None] > 0, p - q <= 0.1, p - q >= -0.1)
    scores = torch.eye(4, dtype=z.dtype)[None] - p[:, None, :]
    ratio = p[rows, actions] / q[rows, actions]
    log_ratio = ratio.log()
    drift = (p[:, :, None] * keep[:, :, None] * scores).sum(1)
    expected = -0.7 * adv[:, None] * ratio[:, None] * keep[rows, actions, None] * scores[rows, actions]
    if score_centering:
        expected += 0.7 * adv[:, None] * drift
    expected += 2 * kl_tau * log_ratio[:, None] * scores[rows, actions]
    reduction = weights / torch.tensor([1, 1, 3, 3, 3], dtype=z.dtype) / 2
    expected *= reduction[:, None]
    torch.testing.assert_close(torch.autograd.grad(loss, z)[0], expected)
    baseline_value = ((-0.7 * adv * ratio * keep[rows, actions] + kl_tau * log_ratio.square()) * reduction).sum()
    torch.testing.assert_close(loss, baseline_value)


def test_optional_sc_config_and_missing_correction():
    from prime_rl.configs.trainer import TrainerConfig, uses_score_centering

    assert not uses_score_centering(DefaultLossConfig())
    assert uses_score_centering(DefaultLossConfig(score_centering=True))
    with pytest.raises(ValueError, match="fused_lm_head"):
        TrainerConfig(loss={"score_centering": True})
    config = TrainerConfig(
        loss={"score_centering": True, "aggregation": "group_token_mean"},
        model={"fused_lm_head_token_chunk_size": "disabled"},
    )
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-0.7]), inference_logprobs=torch.tensor([-0.8]),
        ref_logprobs=None, advantages=torch.ones(1), loss_mask=torch.ones(1, dtype=torch.bool),
    )
    setup_rl_loss_fn(DefaultLossConfig())(inputs)
    with pytest.raises(ValueError, match="masked-score correction"):
        setup_rl_loss_fn(config.loss)(inputs)
