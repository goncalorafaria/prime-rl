"""Exercise the same msgspec conversion used by engine utility RPCs."""
import os
import msgspec
os.environ['PRIME_RL_HYBRID_LORA']='1'
from prime_rl.inference.vllm.hybrid_lora import install_engine_hooks
install_engine_hooks()
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.lora.request import LoRARequest
request=LoRARequest(lora_name='A',lora_int_id=1,lora_path='/tmp/adapter')
core=object.__new__(EngineCore)
converted=EngineCoreProc._convert_msgspec_args(core.add_lora,[msgspec.to_builtins(request)])
assert isinstance(converted[0],LoRARequest)
assert converted[0].lora_name=='A'
print('HYBRID_RPC_TYPE_CHECK_PASSED',flush=True)
