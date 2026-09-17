"""Check the API fence without loading model weights."""
import asyncio
import os
import httpx
from fastapi import FastAPI
from prime_rl.inference.vllm.hybrid_lora import add_routes

async def check():
    previous=os.environ.get('PRIME_RL_HYBRID_LORA')
    os.environ['PRIME_RL_HYBRID_LORA']='1'
    try:
        for eager in (False,True):
            app=FastAPI()
            @app.get('/health')
            async def health():
                return {'status':'ok'}
            if eager:
                app.middleware_stack=app.build_middleware_stack()
            add_routes(app)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://check') as client:
                assert (await client.get('/health')).status_code==200
                app.state.hybrid_failed=True
                assert (await client.get('/health')).status_code==503
                assert (await client.post('/inference/v1/generate',json={})).status_code==503
        print('HYBRID_FENCE_CHECK_PASSED eager_and_lazy_stack=true',flush=True)
    finally:
        if previous is None:
            os.environ.pop('PRIME_RL_HYBRID_LORA',None)
        else:
            os.environ['PRIME_RL_HYBRID_LORA']=previous

if __name__=='__main__':
    asyncio.run(check())
