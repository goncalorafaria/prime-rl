import pytest
import torch

from prime_rl.trainer.rl.score_centering import centered_logprob, importance_weights


@pytest.mark.parametrize("weighting", ["none", "is", "tis", "mis"])
def test_full_head_matches_exact_centered_score(weighting):
    z = torch.tensor([[0.3, -0.7, 1.2], [-0.2, 0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    logp = z.log_softmax(-1)
    q = torch.tensor([[0.65, 0.25, 0.1], [0.1, 0.3, 0.6]], dtype=z.dtype, requires_grad=True)
    actions = torch.tensor([0, 2])
    rows = torch.arange(2)
    advantage = torch.tensor([1.3, -0.7], dtype=z.dtype)
    surrogate, _ = centered_logprob(
        logp[rows, actions],
        q.log()[rows, actions],
        logp,
        q.log(),
        torch.ones_like(logp, dtype=torch.bool),
        weighting=weighting,
        cap=2,
        low=0.5,
        high=2,
    )
    actual = torch.autograd.grad((advantage * surrogate).sum(), z, retain_graph=True)[0]
    p = logp.detach().exp()
    w = importance_weights(logp.detach() - q.detach().log(), weighting, 2, 0.5, 2)
    scores = torch.eye(3, dtype=z.dtype)[None] - p[:, None, :]
    expected = advantage[:, None] * (
        w[rows, actions, None] * scores[rows, actions] - (q.detach() * w)[:, :, None].mul(scores).sum(1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.grad(surrogate.sum(), q, allow_unused=True)[0] is None


@pytest.mark.parametrize("weighting", ["none", "tis", "mis"])
def test_topk_matches_explicit_tail_reconstruction(weighting):
    z = torch.tensor([[0.1, -0.3, 0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    head = torch.tensor([[0, 2]])
    qhead = torch.tensor([[0.2, 0.5]], dtype=z.dtype)
    p = lp.detach().exp()
    qhat = p * (0.3 / (p[0, 1] + p[0, 3]))
    qhat = qhat.scatter(1, head, qhead)
    # The sampled token is outside the logged head; its actual q is still used.
    sample_q = torch.tensor([0.17], dtype=z.dtype)
    approx, _ = centered_logprob(
        lp[:, 1],
        sample_q.log(),
        lp.gather(1, head),
        qhead.log(),
        torch.ones_like(head, dtype=torch.bool),
        weighting=weighting,
    )
    full, _ = centered_logprob(
        lp[:, 1], sample_q.log(), lp, qhat.log(), torch.ones_like(lp, dtype=torch.bool), weighting=weighting
    )
    ga = torch.autograd.grad(approx.sum(), z, retain_graph=True)[0]
    gb = torch.autograd.grad(full.sum(), z)[0]
    torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-12)


def test_plain_sc_matches_logit_ste_for_nonlinear_policy():
    theta = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[1.0, 2.0, -0.5], [0.2, -0.3, 1.0]], dtype=theta.dtype)
    z = torch.tanh(theta @ features)
    zq = torch.tensor([-0.5, 0.2, 0.8], dtype=theta.dtype)
    lp = z.log_softmax(-1)
    lq = zq.log_softmax(-1)
    sc, _ = centered_logprob(lp[1], lq[1], lp, lq, torch.ones_like(lp, dtype=torch.bool))
    ste = (z + (zq - z).detach()).log_softmax(-1)[1]
    ga = torch.autograd.grad(sc, theta, retain_graph=True)[0]
    gb = torch.autograd.grad(ste, theta)[0]
    torch.testing.assert_close(ga, gb)


def test_padding_and_onpolicy_are_finite():
    z = torch.tensor([[0.3, -0.4]], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    headp = torch.cat([lp, torch.full((1, 1), -torch.inf, dtype=z.dtype)], -1)
    for weighting in ["none", "is", "tis", "mis"]:
        value, metrics = centered_logprob(
            lp[:, 0], lp[:, 0].detach(), headp, headp.detach(), torch.tensor([[True, True, False]]), weighting=weighting
        )
        expected = torch.autograd.grad(lp[:, 0].sum(), z, retain_graph=True)[0]
        actual = torch.autograd.grad(value.sum(), z, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(torch.autograd.grad(value.sum(), z, retain_graph=True)[0]).all()


def test_filtered_sampler_with_single_action_has_zero_centered_update():
    z = torch.tensor([[.3, -.4, .8]], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    for weighting in ["none", "is", "tis", "mis"]:
        surrogate, _ = centered_logprob(lp[:, 0], torch.zeros(1, dtype=z.dtype), lp[:, :1],
                                       torch.zeros((1, 1), dtype=z.dtype), torch.ones((1, 1), dtype=torch.bool),
                                       weighting=weighting)
        grad = torch.autograd.grad(surrogate.sum(), z, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.zeros_like(z), atol=1e-12, rtol=0)


@pytest.mark.parametrize("partial_head", [False, True])
@pytest.mark.parametrize("advantage", [1.3, -0.7])
def test_dppo_correction_matches_masked_is_drift(partial_head, advantage):
    from prime_rl.trainer.rl.score_centering import dppo_score_correction

    z = torch.tensor([[0.3, -0.7, 1.2, 0.5]], dtype=torch.float64, requires_grad=True)
    p = z.detach().softmax(-1)
    ids = torch.tensor([[0, 2, -1]]) if partial_head else torch.tensor([[0, 1, 2, 3]])
    qhead = torch.tensor([[0.6, 0.1, 0.0]] if partial_head else [[0.6, 0.15, 0.1, 0.15]], dtype=z.dtype)
    lq = qhead.log().requires_grad_()
    adv = torch.tensor([advantage], dtype=z.dtype)
    correction = dppo_score_correction(z, ids, lq, adv, torch.tensor([True]), low=0.1, high=0.1)
    actual = torch.autograd.grad(correction.sum(), z, retain_graph=True)[0]
    if partial_head:
        q = p * (0.3 / (p[0, 1] + p[0, 3]))
        q[0, 0], q[0, 2] = 0.6, 0.1
    else:
        q = qhead
    keep = ((p - q) <= 0.1) if advantage > 0 else ((p - q) >= -0.1)
    scores = torch.eye(4, dtype=z.dtype)[None] - p[:, None, :]
    expected = ((q * (p / q) * keep)[:, :, None] * scores).sum(1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.grad(correction.sum(), lq, allow_unused=True)[0] is None


def test_dppo_unmasked_is_has_zero_drift_and_ignores_inactive_tokens():
    from prime_rl.trainer.rl.score_centering import dppo_score_correction

    z = torch.tensor([[0.2, -0.4, 0.8], [0.1, 0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 1, 2], [-1, -1, -1]])
    q = torch.tensor([[0.6, 0.3, 0.1], [float('nan')] * 3], dtype=z.dtype)
    correction = dppo_score_correction(
        z, ids, q.log(), torch.ones(2), torch.tensor([True, False]), low=1.0, high=1.0, chunk_size=1
    )
    grad = torch.autograd.grad(correction.sum(), z)[0]
    torch.testing.assert_close(grad, torch.zeros_like(z), atol=1e-12, rtol=0)
    assert correction[1] == 0


def test_dppo_filtered_sampler_centers_supported_score():
    from prime_rl.trainer.rl.score_centering import dppo_score_correction

    z = torch.tensor([[0.3, -0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    correction = dppo_score_correction(
        z, torch.tensor([[0]]), torch.zeros((1, 1), dtype=z.dtype),
        torch.ones(1), torch.tensor([True]), low=1.0, high=1.0
    )
    # The only possible action's IS score equals its expected IS score.
    ratio = z.log_softmax(-1)[:, 0].exp()
    grad = torch.autograd.grad((ratio - correction).sum(), z)[0]
    torch.testing.assert_close(grad, torch.zeros_like(z), atol=1e-12, rtol=0)
