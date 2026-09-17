"""Exercise the actual LiteCast publisher, sidecar, gateway and renderer together."""
import asyncio
import os
from types import SimpleNamespace
import uvicorn
from prime_rl.configs.shared import LitecastPoolConfig
from prime_rl.litecast.distribution import Publisher
from prime_rl.litecast.inflight import live_alias
from prime_rl.litecast.protocol import base_alias
from prime_rl.litecast.worker import Worker, create_app
from prime_rl.litecast.affinity import install_prompt_affinity
from prime_rl.litecast.lineage import install_lineage_tracking, validate_lineage


async def check(admin, generate, local, registry_uri, model, gateway_url, backend_port, port):
    os.environ['LITECAST_INFLIGHT_UPDATES']='1'
    config=LitecastPoolConfig(registry=registry_uri,run_id='check-live',origin_host='127.0.0.1',origin_port=port(),poll_seconds=.2)
    publisher=Publisher(config,model)
    args=SimpleNamespace(run_id=config.run_id,registry=registry_uri,base_model=model,
        backend_url=f'http://127.0.0.1:{backend_port}',request_timeout=600,lease_seconds=30,
        advertise_host='127.0.0.1',port=port(),shard_port=port(),cache_dir=local,
        transport='http',max_versions=8,max_adapter_bytes=512*1024*1024,shard_bytes=8*1024*1024,
        max_inflight_requests=4,poll_seconds=.2,require_middle=False)
    worker=Worker(args)
    server=uvicorn.Server(uvicorn.Config(create_app(worker),host='127.0.0.1',port=args.port,log_level='warning'))
    task=asyncio.create_task(server.serve())
    await publisher.start()
    try:
        old=await publisher.publish(local/'step_2',2)
        await publisher.wait_ready(live_alias(old),120)
        install_prompt_affinity()
        install_lineage_tracking()
        from verifiers.clients import renderer_client
        from verifiers.types import ClientConfig
        from renderers.configs import Qwen35RendererConfig
        rc=renderer_client.RendererClient(ClientConfig(api_base_url=gateway_url+'/v1',api_key_var='EMPTY',
            renderer_model_name=model,renderer_config=Qwen35RendererConfig(enable_thinking=False)))
        assert str(rc.client.base_url).rstrip('/') == gateway_url+'/v1'
        state={'prompt':[{'role':'user','content':'What is 1 plus 1?'}]}
        try:
            rendered=await rc.get_native_response(state['prompt'],live_alias(old),
                {'max_tokens':32,'temperature':0,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},state=state)
            assert state['info']['litecast_policy_lineage'],state
            validate_lineage(rendered['litecast_policy_lineage'],len(rendered['completion_ids']))
        finally:
            await rc.close()
        output_tokens=int(os.getenv('LITECAST_CHECK_OUTPUT_TOKENS','8192'))
        active=asyncio.create_task(generate(live_alias(old),output_tokens))
        slot=base_alias(config.run_id)+'-live'
        async with asyncio.timeout(60):
            while True:
                status=(await admin.get('/litecast/v1/hybrid_status')).json()
                if any(r['adapter']==slot and r['output_tokens']>=16 for r in status['active']): break
                if active.done(): raise RuntimeError('sidecar request finished before update')
                await asyncio.sleep(.1)
        new=await publisher.publish(local/'step_3',3)
        await publisher.wait_ready(live_alias(new),120)
        result=await active
        lineage=result['hybrid_lora_lineage']
        validate_lineage(lineage,output_tokens)
        assert [s['version'] for s in lineage['segments']]==[old.model_name,new.model_name],lineage
        newer=await generate(live_alias(old),32)
        assert newer['hybrid_lora_lineage']['segments'][0]['version']==new.model_name,newer
        return {'status':'passed','lineage':lineage,'renderer_metadata':state['info'],
                'transfer_metrics':await publisher.transfer_metrics(new)}
    finally:
        server.should_exit=True
        await task
        await publisher.close()
