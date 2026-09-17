"""Controlled multi-turn token replay through the PrimeRL LiteRegistry gateway."""
import asyncio
import hashlib
import json
import os
import random
import runpy
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
import tomli_w
import tomllib
import uvicorn
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from prime_rl.litecast.affinity import prompt_key
from prime_rl.litecast.gateway import CapacityConfig, create_app

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'outputs' / ('affinity-bench-' + os.environ['REXS_EXPERIMENT_ID'])
OUT.mkdir(parents=True, exist_ok=True)
MODEL_SOURCE = '/gscratch/ark/graf/data/batchrubrics/models/sloth2b-sft-step400'
PROCESSES = []


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def launch(args, name, env=None):
    log = (OUT / f'{name}.log').open('w')
    p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    PROCESSES.append(p)
    return p


def emit(value):
    print(json.dumps(value), flush=True)


async def healthy(url, process):
    async with httpx.AsyncClient(timeout=5) as client:
        async with asyncio.timeout(900):
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Backend exited {process.returncode}: {url}')
                try:
                    response = await client.get(url + '/health')
                    if response.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(2)


async def worker():
    rank = int(os.environ['BEAKER_REPLICA_RANK'])
    model = str(runpy.run_path(str(ROOT / 'examples/litecast-rubrics/stage.py'))['stage'](MODEL_SOURCE, 'sloth2b-sft-step400'))
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0]:
        raise RuntimeError(f'Expected one allocated A40, got {devices}')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total', '--format=csv'], text=True)
    (OUT / f'hardware-{rank}.csv').write_text(hardware)
    config = tomllib.loads((ROOT / 'examples/litecast-rubrics/inference.toml').read_text())
    config['enable_lora'] = False
    config['model']['name'] = model
    config['model']['max_model_len'] = 16384
    config['parallel'] = {'tp': 1, 'dp': 1}
    config['vllm_extra']['max_num_seqs'] = 16
    backend_port = port()
    config['server'].update(host='0.0.0.0', port=backend_port)
    path = OUT / f'inference-{rank}.toml'
    path.write_text(tomli_w.dumps(config))
    process = launch(['uv', 'run', '--no-sync', 'inference', '@', str(path), '--data-parallel-rpc-port', str(port())], f'engine-{rank}')
    await healthy(f'http://127.0.0.1:{backend_port}', process)
    record = {'url': f'http://{socket.getfqdn()}:{backend_port}', 'model': model,
              'rank': rank, 'job_id': os.environ.get('SLURM_JOB_ID'), 'devices': devices}
    pending = OUT / f'worker-{rank}.partial'
    pending.write_text(json.dumps(record))
    pending.rename(OUT / f'worker-{rank}.json')
    emit({'event': 'worker_ready', **record})
    while not (OUT / 'STOP').exists():
        if process.poll() is not None:
            raise RuntimeError(f'Engine {rank} exited {process.returncode}')
        await asyncio.sleep(2)


async def main():
    records = []
    async with asyncio.timeout(3600):
        while len(records) < 4:
            records = [json.loads(p.read_text()) for p in sorted(OUT.glob('worker-*.json'))]
            if len(records) < 4:
                emit({'event': 'waiting_for_workers', 'ready': len(records), 'expected': 4})
                await asyncio.sleep(10)
    assert len({record['model'] for record in records}) == 1
    model = records[0]['model']
    servers = [(record['url'], None) for record in records]
    async with httpx.AsyncClient(timeout=10) as client:
        for url, _ in servers:
            response = await client.get(url + '/health')
            response.raise_for_status()
    emit({'event': 'engines_ready', 'replicas': 4, 'tp': 1, 'workers': records})
    registry_path = f'file://{tempfile.mkdtemp(prefix="affinity-registry-")}'
    store = get_kvstore(registry_path, raise_on_error=True)
    registry = RegistryClient(store, cache_ttl=0)
    registrations = []
    for i, (url, _) in enumerate(servers):
        registration = ServerRegistry(store)
        await registration.register_server(url.rsplit(':', 1)[0], int(url.rsplit(':', 1)[1]),
            {'model_path': model, 'replica_id': f'bench-{i}', 'max_inflight_requests': 16})
        registrations.append(registration)
    app = create_app(registry, CapacityConfig(max_inflight_per_replica=16, request_timeout_seconds=600,
        queue_timeout_seconds=600, prompt_affinity_replicas=2))
    gateway_port = port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=gateway_port, log_level='warning'))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        if server_task.done():
            await server_task
        await asyncio.sleep(.05)
    async def heartbeat():
        while True:
            for registration, (url, _) in zip(registrations, servers):
                await registration.heartbeat(url.rsplit(':', 1)[0], int(url.rsplit(':', 1)[1]))
            await asyncio.sleep(5)
    heartbeat_task = asyncio.create_task(heartbeat())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, local_files_only=True)
    corpus = []
    for group in range(8):
        rng = random.Random(1700 + group)
        document = '\n'.join(f'Record {i}: item {rng.randrange(1000000)}, count {rng.randrange(10000)}, status verified.' for i in range(150))
        initial = [{'role': 'system', 'content': 'Analyze the supplied records and tool results. Give a concise summary.'},
                   {'role': 'user', 'content': f'Audit batch {group}.\n{document}'}]
        conversation = list(initial)
        turns = []
        for turn in range(4):
            # Fixed replay keeps request tokens identical in both routing modes.
            text = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in conversation)
            text += '<|im_start|>assistant\n<think>\n'
            tokens = tokenizer.encode(text, add_special_tokens=False)
            turns.append(tokens)
            conversation += [{'role': 'assistant', 'content': f'I will inspect the next records for audit batch {group}, stage {turn}.'},
                             {'role': 'tool', 'content': '\n'.join(f'Stage {turn}, match {j}: code {rng.randrange(1000000)} verified.' for j in range(35))}]
        corpus.append({'key': prompt_key(initial, initial=True), 'turns': turns})
    (OUT / 'corpus.json').write_text(json.dumps(corpus))
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{gateway_port}', timeout=600) as client:
        async def metrics():
            totals = {}
            for i, (url, _) in enumerate(servers):
                response = await client.get(url + '/metrics')
                response.raise_for_status()
                for family in text_string_to_metric_families(response.text):
                    for sample in family.samples:
                        if any(word in sample.name for word in ('prefix_cache', 'time_to_first_token_seconds', 'request_prefill_time_seconds')):
                            if not sample.name.endswith(('_bucket', '_created')):
                                totals[sample.name] = totals.get(sample.name, 0) + sample.value
            return totals
        async def request(tokens, salt, headers, output_tokens=96):
            response = await client.post('/inference/v1/generate', headers=headers, json={
                'model': model, 'token_ids': tokens, 'cache_salt': salt,
                'sampling_params': {'temperature': 0, 'seed': 17, 'max_tokens': output_tokens, 'min_tokens': output_tokens, 'ignore_eos': True, 'logprobs': 1}})
            response.raise_for_status()
            data = response.json()
            generated = data['choices'][0]['token_ids']
            if len(generated) != output_tokens:
                raise RuntimeError(f'Expected {output_tokens} tokens, got {len(generated)}')
            return data
        # Exercise prefill/decode kernels before collecting either condition.
        await asyncio.gather(*(request(corpus[i % 8]['turns'][0], 'warmup-' + uuid4().hex, {}, 16) for i in range(16)))
        results = []
        for concurrency in (8, 32):
            for repeat in range(2):
                modes = ('none', 'weak') if repeat == 0 else ('weak', 'none')
                for mode in modes:
                    salt = uuid4().hex
                    semaphore = asyncio.Semaphore(concurrency)
                    rows, episodes = [], []
                    before = await metrics()
                    stats_before = app.state.capacity_routing.snapshot()
                    start = time.perf_counter()
                    async def episode(index):
                        async with semaphore:
                            began = time.perf_counter()
                            sample = corpus[index // 4]
                            headers = {'X-Session-ID': sample['key']} if mode == 'weak' else {}
                            for turn, tokens in enumerate(sample['turns']):
                                tick = time.perf_counter()
                                data = await request(tokens, salt, headers)
                                rows.append({'episode': index, 'turn': turn, 'seconds': time.perf_counter() - tick,
                                    'prompt_tokens': len(tokens), 'output_tokens': len(data['choices'][0]['token_ids']), 'usage': data.get('usage')})
                                await asyncio.sleep(.05)
                            episodes.append(time.perf_counter() - began)
                    await asyncio.gather(*(episode(i) for i in range(32)))
                    wall = time.perf_counter() - start
                    after = await metrics()
                    delta = {k: after[k] - before.get(k, 0) for k in after}
                    stats_after = app.state.capacity_routing.snapshot()
                    result = {'mode': mode, 'concurrency': concurrency, 'repeat': repeat, 'wall_seconds': wall,
                        'episode_mean_seconds': statistics.mean(episodes), 'episode_p95_seconds': sorted(episodes)[int(.95 * (len(episodes)-1))],
                        'request_mean_seconds': statistics.mean(x['seconds'] for x in rows),
                        'output_tokens_per_second': sum(x['output_tokens'] for x in rows) / wall,
                        'metrics_delta': delta, 'routing_delta': {k: stats_after[k] - stats_before[k] for k in ('affinity_preferred', 'affinity_overflow')},
                        'requests': rows}
                    results.append(result)
                    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
                    emit({k: v for k, v in result.items() if k != 'requests'})
        (OUT / 'COMPLETE').write_text('All eight phases completed\n')
    heartbeat_task.cancel()
    await asyncio.gather(heartbeat_task, return_exceptions=True)
    server.should_exit = True
    await server_task
    for registration in registrations:
        await registration.deregister()
    await registry.close()


try:
    asyncio.run(worker() if len(sys.argv) > 1 and sys.argv[1] == 'worker' else main())
finally:
    if len(sys.argv) == 1 or sys.argv[1] != 'worker':
        (OUT / 'STOP').write_text('Benchmark coordinator exited\n')
    import signal
    for process in PROCESSES:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in PROCESSES:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
