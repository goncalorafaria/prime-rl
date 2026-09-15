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
        viewer = api.viewer
        if not api.api_key or not viewer:
            raise RuntimeError("W&B credentials did not authenticate")
    except Exception:
        raise RuntimeError(
            "W&B preflight failed inside the training container. Supply WANDB_API_KEY "
            "or NETRC pointing to a readable, mounted credentials file."
        ) from None
    expected = os.environ.get("LITECAST_WANDB_USERNAME")
    if expected and viewer.username != expected:
        raise RuntimeError(f"W&B credentials do not belong to expected user {expected}")
    print(
        f"WANDB_PREFLIGHT_OK username={viewer.username} "
        f"entity={os.environ.get('WANDB_ENTITY', 'default')} seconds={time.monotonic() - started:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "online"
    main()
