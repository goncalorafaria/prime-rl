"""Quokka terminal-enabled judge adapter using only its local model gateway."""
import json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

class GatewayOnlyRegistry:
    def __init__(self,model,gateway):self.model=model;self.gateway=gateway
    async def sample_servers(self,model,n=1):
        if model!=self.model:raise ValueError('Unconfigured judge model')
        return [(self.gateway,1.0)]

def free_port():
    with socket.socket() as s:s.bind(('',0));return s.getsockname()[1]

def main():
    jtc_source="/gscratch/ark/graf/jtc-runtime"
    sys.path.insert(0,"/gscratch/ark/graf/literegistry-core")
    sys.path.insert(0,jtc_source)
    os.environ["PYTHONPATH"]=jtc_source+":/gscratch/ark/graf/literegistry-core:"+os.environ.get("PYTHONPATH", "")
    from prime_rl.litecast.bootstrap import head_registry,wait,refresh_endpoint
    import uvicorn
    here=Path(__file__).resolve().parent;root=here.parents[2]
    run='toy-'+os.environ.get('LITECAST_PARENT_EXPERIMENT_ID',os.environ['REXS_EXPERIMENT_ID'])
    head=head_registry(root/'outputs'/run)
    settings=json.loads((here/'judge-settings.json').read_text())
    settings['service']=os.environ.get('JUDGE_SERVICE_OVERRIDE',settings['service'])
    registry=wait(head,'redis',timeout=3600,healthcheck='redis')
    if head.startswith('sqlite:'):registry='head+'+head
    port,api_port=int(os.environ.get('JUDGE_GATEWAY_PORT',0)) or free_port(),free_port();gateway=f'http://127.0.0.1:{port}'
    env={**os.environ,'REGISTRY_PATH':registry,'TIMEOUT':os.environ.get('TIMEOUT','600'),'MAX_RETRIES':os.environ.get('MAX_RETRIES','5')}
    process=subprocess.Popen(['uv','run','--no-project','--python',sys.executable,'python','-m','uvicorn','jtc.datadev.literegistry.gateway:create_app','--factory','--host','127.0.0.1','--port',str(port)],env=env,start_new_session=True)
    def stop(signum,frame):raise SystemExit(128+signum)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,stop)
    try:
        deadline=time.monotonic()+3600
        while True:
            if process.poll() is not None:raise RuntimeError('Judge gateway exited')
            try:
                with urlopen(gateway+'/v1/models',timeout=5) as r:models={m['id'] for m in json.load(r).get('data',[])}
                if settings['model'] in models and 'terminal' in models:break
            except (URLError,TimeoutError):pass
            if time.monotonic()>deadline:raise TimeoutError('Waiting for Quokka and terminal registrations')
            time.sleep(2)
        sys.path.insert(0,'/gscratch/ark/graf/rexs/deploy/klone/browsecomp/judge')
        from judge_server import JudgeServer,JudgeServerConfig
        config=JudgeServerConfig(host='0.0.0.0',port=api_port,registry=registry,model_path=settings['service'],tool_server_url=gateway,tools=('webterminal',),model_profiles_dir=str(here/'profiles'))
        server=JudgeServer(config)
        # The reusable judge workflow asks for a model endpoint. Supply only the
        # local gateway; model replicas are selected by that gateway's registry.
        server.model_registry=GatewayOnlyRegistry(settings['model'],gateway)
        import asyncio
        async def supervise():
            http=uvicorn.Server(uvicorn.Config(server.app,host='0.0.0.0',port=api_port))
            task=asyncio.create_task(http.serve())
            try:
                while not task.done():
                    if process.poll() is not None:raise RuntimeError('Judge gateway exited')
                    if http.started:
                        await asyncio.to_thread(refresh_endpoint,head,settings['service'],f'http://{socket.getfqdn()}:{api_port}',publisher_id=run+'-judge-api',ttl_seconds=30)
                    await asyncio.sleep(2)
                await task
            finally:
                http.should_exit=True
                await task
        asyncio.run(supervise())
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

if __name__=='__main__':main()
