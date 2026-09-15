import importlib
import subprocess
import sys
import time
from pathlib import Path

started = time.monotonic()
modules = {
    name: importlib.import_module(name) for name in ("torch", "transformers", "vllm", "wandb", "reverse_text_v1")
}

for name in ("torch", "transformers", "vllm", "wandb"):
    module = modules[name]
    assert str(Path(module.__file__).resolve()).startswith(sys.prefix), module.__name__
assert sys.prefix.startswith("/tmp/litecast-runtime-graf/"), sys.prefix
assert sys.base_prefix.startswith("/tmp/litecast-runtime-graf/"), sys.base_prefix
for name in ("rl", "inference", "torchrun"):
    first = (Path(sys.prefix) / "bin" / name).read_text().splitlines()[0]
    assert first == "#!" + sys.prefix + "/bin/python", (name, first)
print(f"LOCAL_RUNTIME_IMPORT seconds={time.monotonic() - started:.2f} prefix={sys.prefix}", flush=True)
subprocess.run(
    [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-c",
        'import os, sys; assert sys.prefix == os.environ["UV_PROJECT_ENVIRONMENT"]; print("PROJECT_RUNTIME", sys.prefix)',
    ],
    check=True,
)
