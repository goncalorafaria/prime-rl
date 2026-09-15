"""Stage the pinned toy dataset locally without Hub metadata discovery on reuse."""

import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

REVISION = "eacc9a0d76d9fd22e40008ab9d546008bdd7e432"
REPO = "PrimeIntellect/Reverse-Text-RL"


def stage():
    root = Path(os.environ.get("HF_HOME", "/tmp/litecast-hf")) / "dataset-snapshots"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"reverse-text-{REVISION}"
    with (root / "reverse-text.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / "ready.json"
        if marker.is_file() and (target / "train.parquet").is_file():
            expected = json.loads(marker.read_text())
            if expected == {"revision": REVISION, "bytes": (target / "train.parquet").stat().st_size}:
                print(f"DATASET_CACHE_HIT path={target}", flush=True)
                return target
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        deadline = time.monotonic() + 600
        delay = 10
        while True:
            try:
                source = hf_hub_download(
                    REPO,
                    "data/train-00000-of-00001.parquet",
                    repo_type="dataset",
                    revision=REVISION,
                    cache_dir=os.environ.get("HF_HUB_CACHE", "/tmp/litecast-hf/hub"),
                )
                break
            except HfHubHTTPError as exc:
                status = exc.response.status_code
                remaining = deadline - time.monotonic()
                if status not in (429, 500, 502, 503, 504) or remaining <= 0:
                    raise
                retry_after = exc.response.headers.get("Retry-After", "")
                pause = max(delay, float(retry_after)) if retry_after.isdecimal() else delay
                pause = min(pause, remaining)
                print(f"DATASET_DOWNLOAD_RETRY status={status} seconds={pause}", flush=True)
                time.sleep(pause)
                delay = min(delay * 2, 60)
        temporary = Path(tempfile.mkdtemp(prefix="reverse-text-", dir=root))
        try:
            shutil.copyfile(source, temporary / "train.parquet")
            (temporary / "ready.json").write_text(
                json.dumps(
                    {
                        "revision": REVISION,
                        "bytes": (temporary / "train.parquet").stat().st_size,
                    }
                )
            )
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"DATASET_CACHE_MISS path={target}", flush=True)
        return target


if __name__ == "__main__":
    stage()
