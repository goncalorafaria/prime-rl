"""A policy update during dispatch must not relabel a LiteCast group."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.litecast.protocol import Publication
from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import Policy


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["train", "eval"])
async def test_group_keeps_immutable_model_across_policy_update(kind):
    first = Publication("pinning", "model", 1, "a" * 64, 100)
    second = Publication("pinning", "model", 2, "b" * 64, 100)
    policy = Policy(version=1, model_name=first.model_name)
    requests = []

    async def select(*args):
        # Readiness/client selection yields while the watcher advances policy.
        await asyncio.sleep(0)
        policy.version, policy.model_name = 2, second.model_name
        return SimpleNamespace()

    async def run(**kwargs):
        requests.append(kwargs["model_name"])
        return [SimpleNamespace()]

    pool = SimpleNamespace(litecast_config=True, select_train_client=select, get_eval_client=select)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=2),
        sampler=SimpleNamespace(pool=pool, samples_from_live_policy=True),
        requires_group_scoring=False,
        run=run,
    )
    envs = SimpleNamespace(get=lambda name: env)
    source = SimpleNamespace(next_example=lambda permits: {"env_name": "toy", "task_idx": 0, "eval_step": 1})
    dispatcher = RolloutDispatcher(
        train_envs=envs,
        eval_envs=envs,
        train_source=source,
        eval_source=source,
        policy_pool=pool,
        policy=policy,
        max_inflight_episodes=2,
        eval_max_inflight_episodes=2,
        tasks_per_minute=None,
        max_off_policy_steps=2,
    )
    group = dispatcher.next_fresh_group(kind, envs, 2)
    group_id = uuid.uuid4()
    dispatcher.groups[group_id] = group
    assert await dispatcher.schedule_group_rollout(group_id, group)
    assert policy.version == 2
    assert await dispatcher.schedule_group_rollout(group_id, group)
    for task, meta in list(dispatcher.inflight.items()):
        rollouts = await task
        assert meta.policy_version == 1
        assert meta.inference_model_name == first.model_name
        await dispatcher.emit_episode(meta, group, rollouts)
        assert rollouts[0].policy_version == 1
        assert rollouts[0].inference_model_name == first.model_name
    assert requests == [first.model_name, first.model_name]
