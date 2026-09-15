"""Discard consumed toy-run artifacts behind a completed publication watermark."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def step_directories(root: Path):
    return {
        int(path.name[5:]): path
        for path in root.glob("step_*")
        if path.name[5:].isdigit() and path.is_dir() and not path.is_symlink()
    }


def prune_intermediates(training: Path, keep_rollout_steps: int, keep_adapter_steps: int):
    if min(keep_rollout_steps, keep_adapter_steps) < 1:
        raise ValueError("retention windows must be positive")
    for run in training.glob("run_*"):
        if not run.is_dir() or run.is_symlink():
            continue
        published = step_directories(run / "broadcasts")
        stable = [step for step, path in published.items() if (path / "STABLE").is_file()]
        if not stable:
            continue
        completed = max(stable)
        for folder, keep in (
            ("rollouts", keep_rollout_steps),
            ("token_exports", keep_rollout_steps),
            ("broadcasts", keep_adapter_steps),
        ):
            for step, path in step_directories(run / folder).items():
                if step <= completed - keep:
                    try:
                        shutil.rmtree(path)
                    except FileNotFoundError:
                        continue  # Trainer broadcast cleanup can remove the same directory.
                    logger.info("Discarded intermediate artifact %s", path)
