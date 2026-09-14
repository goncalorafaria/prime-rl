import tomllib
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.shared import ClientConfig, LitecastPoolConfig

ROOT = Path(__file__).resolve().parents[3]


def test_search_and_worker_configs_match():
    training = RLConfig.model_validate(tomllib.loads((ROOT / "examples/litecast-search/search-2b.toml").read_text()))
    worker = InferenceConfig.model_validate(
        tomllib.loads((ROOT / "examples/litecast-search/inference-l40.toml").read_text())
    )
    assert training.inference is None
    assert training.deployment.num_infer_gpus == 0
    assert training.trainer.model.name == worker.model.name == "Qwen/Qwen3.5-2B"
    assert training.trainer.model.lora.rank == worker.max_lora_rank
    assert training.orchestrator.train.sampling.top_p == 0.97
    assert training.orchestrator.model.client.litecast.retain_versions <= worker.max_cpu_loras
    assert worker.enable_lora


def test_reject_conflicting_discovery():
    config = LitecastPoolConfig(registry="redis://localhost:6379", run_id="r", origin_host="trainer")
    with pytest.raises(ValueError, match="incompatible"):
        ClientConfig(litecast=config, admin_base_url=["http://engine:8000"])


def test_reject_insufficient_adapter_retention():
    raw = tomllib.loads((ROOT / "examples/litecast-search/search-2b.toml").read_text())
    raw["orchestrator"]["model"]["client"]["litecast"]["retain_versions"] = 2
    with pytest.raises(ValueError, match="retain_versions"):
        RLConfig.model_validate(raw)


def test_launcher_materializes_source_list_without_cli_index_overrides():
    from prime_rl.litecast.launch import training_config

    config = RLConfig.model_validate(
        training_config(
            {
                "RUN_ID": "generated-test",
                "REGISTRY": "redis://registry:6379",
                "GATEWAY_URL": "http://gateway:1212/",
                "ADVERTISE_HOST": "trainer",
            }
        )
    )
    assert config.orchestrator.model.client.base_url == ["http://gateway:1212/v1"]
    assert config.orchestrator.train.source[0].legacy.args["search_server_url"] == "http://gateway:1212/search"
