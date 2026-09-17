"""Load local rubric inputs and environment before reserving training GPUs."""
import json
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, '/gscratch/ark/graf/primebeaker/src')
sys.path.insert(0, '/gscratch/ark/graf/literegistry-core/literegistry_tool_client/src')
from primebeaker.environments.articulated_harness_env import load_environment

root = Path('/gscratch/ark/graf/data/batchrubrics')
data = load_from_disk(str(root / 'rl_datadev_7ef8249d31225b43/active-rl'))
assert len(data['train']) == 8000 and len(data['validation']) == 800
config = json.loads((root / 'models/sloth2b-sft-step400/config.json').read_text())
assert config['model_type'] == 'qwen3_5'
print('RUBRIC_PREFLIGHT_OK train=8000 validation=800 model=qwen3_5 environment_imported=true', flush=True)
print('DATASET_COLUMNS', data['train'].column_names, flush=True)
import tomllib
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.inference import InferenceConfig
from literegistry.services.terminal_server import TerminalPipelineServer
p = Path('/gscratch/ark/graf/prime-rl-shardcast/examples/litecast-rubrics')
a = RLConfig.model_validate(tomllib.loads((p / 'train.toml').read_text()))
b = InferenceConfig.model_validate(tomllib.loads((p / 'inference.toml').read_text()))
assert a.deployment.num_train_gpus == 2 and a.trainer.model.cp == 1
print('DP_CONFIG_VALIDATED train_gpus=2 cp=1')
args = dict(tomllib.loads((p / 'train.toml').read_text())['orchestrator']['train']['source'][0]['legacy']['args'])
args['dataset'] = str(root / 'rl_datadev_7ef8249d31225b43/active-rl')
env = load_environment(**args)
print('ENVIRONMENT_CONSTRUCTED', type(env).__name__)

import importlib.util
assert a.trainer.model.attn == "flash_attention_2"
assert importlib.util.find_spec("flash_attn") is not None
print("ATTENTION_PREFLIGHT_OK flash_attention_2")
