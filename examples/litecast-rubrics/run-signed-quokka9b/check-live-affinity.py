"""Run a bounded judge-affinity smoke on an existing Rex CPU allocation."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import httpx
from literegistry import get_kvstore
from literegistry.affinity import SoftAffinityBindingStore
from prime_rl.litecast.bootstrap import head_registry,wait

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = 'toy-' + os.environ['REXS_EXPERIMENT_ID']
OUT = ROOT / 'outputs' / RUN / ('affinity-smoke-' + uuid.uuid4().hex[:8])
OUT.mkdir(parents=True)
HEAD = head_registry(ROOT / 'outputs' / RUN)
settings = json.loads((HERE/'judge-settings.json').read_text())
service = 'quokka-affinity-smoke-' + uuid.uuid4().hex[:8]
with socket.socket() as sock:
    sock.bind(('',0)); port=sock.getsockname()[1]
gateway=f'http://127.0.0.1:{port}'
env={**os.environ,'JUDGE_SERVICE_OVERRIDE':service,'JUDGE_GATEWAY_PORT':str(port),'PYTHONUNBUFFERED':'1'}
log=(OUT/'judge-api.log').open('w')
process=subprocess.Popen(['uv','run','--no-project','--python',sys.executable,'python',str(HERE/'judge-api.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('SMOKE_OUTPUT='+str(OUT),flush=True)

async def run():
    registry=await asyncio.to_thread(wait,HEAD,'redis',timeout=120,healthcheck='redis')
    endpoint=await asyncio.to_thread(wait,HEAD,service,timeout=600,healthcheck='http')
    head_gateway=await asyncio.to_thread(wait,HEAD,'gateway',timeout=120,healthcheck='http')
    store=get_kvstore(registry)
    bindings=SoftAffinityBindingStore(store)
    model=settings['model']; key='jtc-chat:'+model
    evidence={'run':RUN,'registry':registry,'judge_api':endpoint,'head_gateway':head_gateway,'model':model,'chat_turns':[]}
    async with httpx.AsyncClient(timeout=300,trust_env=False) as client:
        models=(await client.get(gateway+'/v1/models')).json()
        assert any(m['id']==model for m in models['data'])
        session='affinity-live-'+uuid.uuid4().hex
        messages=[{'role':'user','content':'What is 6 times 7? Reply with just the number.'}]
        last=None
        for i in range(3):
            r=await client.post(gateway+'/v1/chat/completions',headers={'X-Session-ID':session},json={'model':model,'messages':messages,'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
            r.raise_for_status(); body=r.json(); binding=await bindings.resolve(key,session)
            assert binding is not None
            if last is not None: assert binding.server_uri==last, (last,binding.server_uri)
            last=binding.server_uri
            evidence['chat_turns'].append({'turn':i,'server':last,'response':body['choices'][0]['message']})
            messages += [{'role':'assistant','content':body['choices'][0]['message'].get('content') or ''},{'role':'user','content':'Repeat that number.'}]
        print('CHAT_REUSE_PASSED server='+last,flush=True)
        # Simulate a departed preferred replica without changing the live model pool.
        await bindings.handoff(key,session,'retired-smoke-replica','http://127.0.0.1:1')
        r=await client.post(gateway+'/v1/chat/completions',headers={'X-Session-ID':session},json={'model':model,'messages':messages,'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
        r.raise_for_status(); binding=await bindings.resolve(key,session)
        assert binding.server_uri!='http://127.0.0.1:1'
        evidence['missing_replica_handoff']=binding.server_uri
        print('MISSING_REPLICA_HANDOFF_PASSED',flush=True)
        cases=[('correct','42','The response gives 42 as the answer.','pass'),('incorrect','41','The response gives 42 as the answer.','fail'),('negative','41','The response incorrectly gives 41 as the answer. Mark pass if this error is present and fail if absent.','pass')]
        async def judge(case):
            name,output,rubric,expected=case
            response=await client.post(head_gateway+'/judge',json={'model_path':service,'model':model,'input':'What is 6 times 7?','output':output,'rubrics':[rubric]},timeout=1500)
            response.raise_for_status(); data=response.json()
            print('JUDGE_RESULT '+name+' '+str([j['label'] for j in data['judgments']]),flush=True)
            return {'case':name,'expected':expected,'result':data}
        evidence['judgments']=await asyncio.gather(*(judge(c) for c in cases))
        (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
        for result in evidence['judgments']:
            judgments=result['result']['judgments']
            assert len(judgments)==1 and judgments[0]['label']==result['expected'],result
    # Logs demonstrate that actual tool-using judge sessions reuse replicas.
    log.flush()
    lines=(OUT/'judge-api.log').read_text().splitlines()
    chat_session=hashlib.sha256(session.encode()).hexdigest()[:16]
    judge_reuses=[line for line in lines if 'CHAT_AFFINITY' in line and 'reused=True' in line and 'session='+chat_session not in line]
    assert judge_reuses,'No repeated affinity observed in actual judge workflow'
    assert any('POST /terminal HTTP/1.1" 200' in line for line in lines),'No successful terminal calls'
    evidence['judge_affinity_reuse_log_count']=len(judge_reuses)
    evidence['passed']=True
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
    print('LIVE_AFFINITY_AND_JUDGE_SMOKE_PASSED '+str(OUT/'evidence.json'),flush=True)

try:
    asyncio.run(run())
finally:
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    log.close()
