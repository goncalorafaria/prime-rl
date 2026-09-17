"""TP4 retained-state reload acceptance check; generation uses LiteRegistry."""
import asyncio
import json
import os
from pathlib import Path
import runpy
import signal
import shutil
import socket
import subprocess
import tempfile
import time
import tomllib
import httpx
import tomli_w
import uvicorn
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry
from prime_rl.litecast.gateway import CapacityConfig, create_app

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'outputs' / ('hybrid-preflight-' + os.environ['REXS_EXPERIMENT_ID'])
OUT.mkdir(parents=True, exist_ok=True)

def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]

async def main():
    check_fence = runpy.run_path(str(ROOT/"examples/litecast-hybrid-lora/check_fence.py"))["check"]
    await check_fence()
    model = str(runpy.run_path(str(ROOT/'examples/litecast-rubrics/stage.py'))['stage'](
        '/gscratch/ark/graf/data/batchrubrics/models/sloth2b-sft-step400', 'sloth2b-sft-step400'))
    local = Path(tempfile.mkdtemp(prefix='hybrid-check-'))
    source=ROOT/'outputs/toy-a9f058f6c65e43a19ceefd21ff62d26a/training/run_default/broadcasts'
    for step in (2,3):
        shutil.copytree(source/f'step_{step}', local/f'step_{step}')
    config=tomllib.loads(Path(os.getenv('LITECAST_CHECK_INFERENCE_CONFIG', ROOT/'examples/litecast-rubrics/inference-tp4.toml')).read_text())
    backend_port=port()
    config['model']['name']=model
    config['server']['port']=backend_port
    config['vllm_extra']['async_scheduling']=False
    path=OUT/'inference.toml'
    path.write_text(tomli_w.dumps(config))
    with (OUT/'engine.log').open('w') as log:
        process=subprocess.Popen(['uv','run','--no-sync','inference','@',str(path),'--data-parallel-rpc-port',str(port())], stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    gateway=None
    try:
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{backend_port}',timeout=600) as admin:
            async with asyncio.timeout(1200):
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'engine exited {process.returncode}')
                    try:
                        response=await admin.get('/health',timeout=5)
                        if response.status_code==200: break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(2)
            for name in ('A','B'):
                response=await admin.post('/v1/load_lora_adapter',json={'lora_name':name,'lora_path':str(local/'step_2')})
                response.raise_for_status()
            registry_uri='file://'+str(local/'registry')
            store=get_kvstore(registry_uri,raise_on_error=True)
            registry=RegistryClient(store,cache_ttl=0)
            registrations=[]
            for name in ('A','B'):
                reg=ServerRegistry(store)
                await reg.register_server('http://127.0.0.1',backend_port,{'model_path':name,'replica_id':'tp4','max_inflight_requests':16})
                registrations.append(reg)
            gateway=uvicorn.Server(uvicorn.Config(create_app(registry,CapacityConfig(request_timeout_seconds=600)),host='127.0.0.1',port=port(),log_level='warning'))
            server_task=asyncio.create_task(gateway.serve())
            while not gateway.started:
                if server_task.done(): await server_task
                await asyncio.sleep(.1)
            url=f'http://127.0.0.1:{gateway.config.port}'
            async with httpx.AsyncClient(base_url=url,timeout=600) as client:
                async def generate(name, count=512):
                    response=await client.post('/inference/v1/generate',json={'model':name,'token_ids':[100]*(4353 if config['vllm_extra'].get('decode_context_parallel_size',1)>1 else 2048),'sampling_params':{'max_tokens':count,'min_tokens':count,'ignore_eos':True,'temperature':0,'logprobs':1},'cache_salt':'shared'})
                    response.raise_for_status()
                    value=response.json()
                    assert value.get('hybrid_lora_lineage'),value
                    return value
                warm=await generate('A',32)
                shared=await generate('A',32)
                active=asyncio.create_task(generate('A'))
                other=asyncio.create_task(generate('B'))
                async with asyncio.timeout(60):
                    while True:
                        status=(await admin.get('/litecast/v1/hybrid_status')).json()
                        if any(r['adapter']=='A' and r['output_tokens']>=16 for r in status['active']): break
                        if active.done(): raise RuntimeError('A completed before update')
                        await asyncio.sleep(.1)
                started=time.perf_counter()
                response=await admin.post('/litecast/v1/update_lora_inflight',json={'lora_name':'A','lora_path':str(local/'step_3'),'expected_version':str(local/'step_2'),'version':'A-v3'})
                response.raise_for_status()
                event=response.json()
                update_seconds=time.perf_counter()-started
                results=await asyncio.gather(active,other)
                a,b=[v['hybrid_lora_lineage'] for v in results]
                assert len(a['segments'])==2 and a['segments'][1]['version']=='A-v3',a
                assert 0<a['segments'][1]['output_start']<512,a
                assert len(b['segments'])==1,b
                fresh=await generate('A',32)
                reused=await generate('A',32)
                assert fresh['hybrid_lora_lineage']['segments'][0]['version']=='A-v3',fresh
                print('CACHE_CHECK '+json.dumps({'shared':shared.get('usage'),'fresh':fresh.get('usage'),'reused':reused.get('usage')}),flush=True)
                assert (reused['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) > 0, reused.get('usage')
                assert shared['usage']['prompt_tokens_details']['cached_tokens'] > 0, shared.get('usage')
                assert fresh['choices'][0]['logprobs'] != warm['choices'][0]['logprobs'], 'Reload did not change generation logprobs'
                result={'status':'passed','tp':config['parallel']['tp'],'dcp':config['vllm_extra'].get('decode_context_parallel_size',1),'update_seconds':update_seconds,'event':event,'A':a,'B':b,'old_cache_usage':shared.get('usage'),'new_cache_usage':reused.get('usage')}
                worker_check=runpy.run_path(str(ROOT/'examples/litecast-hybrid-lora/check_worker.py'))['check']
                result['worker']=await worker_check(admin,generate,local,registry_uri,model,url,backend_port,port)
                (OUT/'result.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(result),flush=True)
            gateway.should_exit=True
            await server_task
            for reg in registrations: await reg.deregister()
            await registry.close()
    finally:
        if gateway: gateway.should_exit=True
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()

asyncio.run(main())
