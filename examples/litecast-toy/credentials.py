"""Inject local credentials into child processes without embedding secrets in job specs."""

import netrc
import os
from pathlib import Path


def configure(*, trainer):
    if not os.environ.get("HF_TOKEN"):
        path = Path(os.environ.get("LITECAST_HF_TOKEN_FILE", "/gscratch/ark/graf/.cache/huggingface/token"))
        os.environ["HF_TOKEN"] = path.read_text().strip()
    if not os.environ["HF_TOKEN"]:
        raise RuntimeError("HF_TOKEN is empty")
    if trainer:
        if not os.environ.get("WANDB_API_KEY"):
            path = os.environ.get("LITECAST_WANDB_TOKEN_FILE")
            if path:
                key = Path(path).read_text().strip()
            else:
                credentials = netrc.netrc(os.environ.get("NETRC", "/gscratch/ark/graf/.netrc"))
                entry = credentials.authenticators("api.wandb.ai")
                key = entry[2] if entry else ""
            if not key:
                raise RuntimeError("No WANDB_API_KEY available from the configured credential file")
            os.environ["WANDB_API_KEY"] = key
        os.environ.setdefault("WANDB_ENTITY", "graf")
        os.environ.setdefault("LITECAST_WANDB_USERNAME", "graf")
    print(f"CREDENTIALS_INJECTED HF_TOKEN=true WANDB_API_KEY={trainer}", flush=True)
