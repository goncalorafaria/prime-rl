"""Copy dependency files concurrently off shared storage before compression."""

import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def copy_runtime(runtime: Path, base_python: Path, destination: Path):
    started = time.monotonic()
    count = 0
    pending = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        for source, target in ((runtime, destination / "runtime"), (base_python, destination / "python")):
            for directory, directories, files in os.walk(source):
                local = target / Path(directory).relative_to(source)
                local.mkdir(parents=True, exist_ok=True)
                for name in directories[:]:
                    entry = Path(directory) / name
                    if entry.is_symlink():
                        (local / name).symlink_to(os.readlink(entry))
                        directories.remove(name)
                for name in files:
                    pending.add(
                        executor.submit(shutil.copy2, Path(directory) / name, local / name, follow_symlinks=False)
                    )
                    count += 1
                    if len(pending) >= 64:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for result in done:
                            result.result()
                    if count % 2000 == 0:
                        print(f"RUNTIME_COPY files={count} seconds={time.monotonic() - started:.1f}", flush=True)
        for result in pending:
            result.result()
    print(f"RUNTIME_COPY_COMPLETE files={count} seconds={time.monotonic() - started:.1f}", flush=True)


if __name__ == "__main__":
    copy_runtime(*(Path(value) for value in sys.argv[1:]))
