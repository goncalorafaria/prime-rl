"""Own one Quokka vLLM replica registered directly with the shared head."""
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from prime_rl.litecast.bootstrap import head_registry, wait


def interrupt(signum, frame):
    raise SystemExit(128 + signum)


def main():
    root = Path(__file__).resolve().parents[3]
    run = 'toy-' + os.environ.get('LITECAST_PARENT_EXPERIMENT_ID', os.environ['REXS_EXPERIMENT_ID'])
    head = head_registry(root / 'outputs' / run)
    registry = wait(head, 'redis', timeout=3600, healthcheck='redis')
    if head.startswith('sqlite:'):
        registry = 'head+' + head
    env = {
        **os.environ,
        'REGISTRY': registry,
        'MODEL_ID': os.environ['LITECAST_JUDGE_MODEL'],
        'MODEL_ROLE': 'judge',
        'TENSOR_PARALLEL_SIZE': '1',
        'BEAKER_NODE_HOSTNAME': socket.getfqdn(),
    }
    command = [
        'uv', 'run', '--no-project', '--python', sys.executable, 'python',
        '/gscratch/ark/graf/rexs/deploy/klone/browsecomp/model-service.py',
    ]
    # The model wrapper registers only after its own /v1/models is healthy.
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait()
        raise RuntimeError(f'Quokka model service exited with code {code}')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    main()
