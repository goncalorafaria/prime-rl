"""Validate online W&B access inside the same container as training."""

import os
import time


def main():
    started = time.monotonic()
    print("WANDB_PREFLIGHT_STARTED", flush=True)
    credentials = os.environ.get("NETRC")
    if credentials and not os.environ.get("WANDB_API_KEY") and not os.access(credentials, os.R_OK):
        raise RuntimeError("The configured NETRC credentials file is not readable inside the training container")
    import wandb

    try:
        api = wandb.Api(timeout=30)
        if not api.api_key or not api.viewer:
            raise RuntimeError("W&B credentials did not authenticate")
    except Exception:
        raise RuntimeError(
            "W&B preflight failed inside the training container. Supply WANDB_API_KEY "
            "or NETRC pointing to a readable, mounted credentials file."
        ) from None
    print(f"WANDB_PREFLIGHT_OK seconds={time.monotonic() - started:.2f}", flush=True)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "online"
    main()
