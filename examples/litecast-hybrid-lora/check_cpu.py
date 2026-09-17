import ast
import os
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for path in (ROOT/'src/prime_rl/litecast').glob('*.py'):
    ast.parse(path.read_text())
ast.parse((ROOT/'src/prime_rl/inference/vllm/hybrid_lora.py').read_text())
runpy.run_path(str(ROOT/'examples/litecast-rubrics/preflight.py'))
code=pytest.main(['-q','--confcutdir=tests/unit/litecast','tests/unit/litecast/test_protocol.py','tests/unit/litecast/test_capacity.py'])
raise SystemExit(code)
