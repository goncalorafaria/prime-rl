"""Stage immutable rubric inputs on node-local storage."""
import fcntl
import shutil
from pathlib import Path


def stage(source: str, name: str) -> Path:
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    root = Path('/tmp/litecast-rubrics')
    root.mkdir(exist_ok=True)
    target = root / name
    with (root / (name + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root / (name + '.source')
        if marker.exists():
            if marker.read_text() != str(source.resolve()) or not target.is_dir():
                raise RuntimeError(f'Local cache identity mismatch: {target}')
            return target
        temporary = root / (name + '.partial')
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        temporary.rename(target)
        marker.write_text(str(source.resolve()))
    return target
