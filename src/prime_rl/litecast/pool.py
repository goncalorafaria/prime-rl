"""PrimeRL pool using the LiteRegistry gateway and independent LiteCast replicas."""

import time
from pathlib import Path

from prime_rl.litecast.distribution import Publisher
from prime_rl.litecast.protocol import base_alias
from prime_rl.utils.client import StaticInferencePool


class LitecastInferencePool(StaticInferencePool):
    def __init__(self, client_config, model_name, **kwargs):
        super().__init__(client_config, model_name, **kwargs)
        self.last_transfer_metrics: dict[str, float] = {}
        self.litecast_config = client_config.litecast
        self.publisher = Publisher(self.litecast_config, model_name)
        self.model_name = base_alias(self.litecast_config.run_id)

    @classmethod
    async def from_config(cls, client_config, model_name, **kwargs):
        pool = cls(client_config, model_name, **kwargs)
        try:
            await pool.publisher.start()
        except BaseException:
            await pool.stop()
            raise
        return pool

    @property
    def admin_clients(self):
        # The gateway is a data-plane front door, never an engine admin target.
        return []

    def update_model_name(self, model_name: str):
        # Names are content-addressed per publication; a fixed LoRA alias would
        # let a cached gateway route a new policy to a worker with old weights.
        pass

    async def wait_for_ready(self, model_name: str, timeout: int | None = None):
        await self.publisher.wait_ready(self.model_name, timeout or self._wait_for_ready_timeout)

    async def select_train_client(self, load):
        await self.publisher.wait_ready(self.model_name, self._wait_for_ready_timeout, force=False)
        return await super().select_train_client(load)

    async def get_eval_client(self):
        await self.publisher.wait_ready(self.model_name, self._wait_for_ready_timeout, force=False)
        return await super().get_eval_client()

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0):
        if weight_dir is None or lora_name is None:
            raise ValueError("LiteCast inference requires filesystem LoRA adapter checkpoints")
        started = time.perf_counter()
        publication = await self.publisher.publish(weight_dir, step)
        published = time.perf_counter()
        await self.publisher.wait_ready(publication.model_name, self.litecast_config.update_timeout)
        ready = time.perf_counter()
        self.last_transfer_metrics = {
            "litecast/policy_step": step,
            "litecast/payload_bytes": publication.size,
            "litecast/publish_seconds": published - started,
            "litecast/ready_wait_seconds": ready - published,
            "litecast/update_seconds": ready - started,
            **await self.publisher.transfer_metrics(publication),
        }
        self.model_name = publication.model_name

    async def stop(self):
        await self.publisher.close()
        await super().stop()
        for client in self._admin_clients + self._router_clients:
            await client.aclose()
