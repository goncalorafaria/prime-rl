"""Experimental in-place LoRA swaps with retained per-request state.

Opt in with PRIME_RL_HYBRID_LORA=1. This deliberately changes the policy during
requests. Consumers must inspect lineage; a model name alone is not a version.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse


def enabled():
    return os.getenv("PRIME_RL_HYBRID_LORA") == "1"


class HybridUpdate(BaseModel):
    lora_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    expected_version: str = Field(min_length=1)


def cache_namespace(original_salt, adapter, history):
    encoded = json.dumps([original_salt, adapter, history], sort_keys=True, separators=(",", ":")).encode()
    return "hybrid-lora:" + hashlib.sha256(encoded).hexdigest()


def rehash_request(request, version, *, switch=False, recompute=False):
    if not hasattr(request, "_hybrid_cache_history"):
        request._hybrid_original_salt = request.cache_salt
        request._hybrid_cache_history = []
    if recompute:
        request._hybrid_cache_history = []
        switch = False
    request._hybrid_cache_history.append({
        "version": version,
        "computed_tokens": request.num_computed_tokens if switch else 0,
        "output_tokens": request.num_output_tokens if switch else 0,
    })
    request.cache_salt = cache_namespace(request._hybrid_original_salt, request.lora_request.lora_name,
                                         request._hybrid_cache_history)
    request.block_hashes.clear()
    request.update_block_hashes()


def install_engine_hooks():
    if not enabled():
        return
    from vllm.v1.engine.core import EngineCore
    from vllm.lora.request import LoRARequest

    if getattr(EngineCore, "_prime_hybrid_lora", False):
        return
    from vllm.v1.request import Request as EngineRequest

    original_from_wire = EngineRequest.from_engine_core_request.__func__

    def from_wire(cls, request, block_hasher):
        value = original_from_wire(cls, request, block_hasher)
        value._hybrid_external_id = request.external_req_id or request.request_id
        return value

    EngineRequest.from_engine_core_request = classmethod(from_wire)
    original_add_request = EngineCore.add_request
    original_add_lora = EngineCore.add_lora
    original_abort_requests = EngineCore.abort_requests

    def state(core):
        if not hasattr(core, "_hybrid_state"):
            core._hybrid_state = {"versions": {}, "requests": {}, "pending": None, "failed": False}
        return core._hybrid_state

    def validate(core):
        config = core.vllm_config
        if config.max_concurrent_batches != 1:
            raise RuntimeError("Hybrid LoRA currently requires async_scheduling=false and PP=1")
        if config.parallel_config.data_parallel_size != 1:
            raise RuntimeError("Hybrid LoRA currently requires DP=1 per inference replica (TP is supported)")
        if config.speculative_config is not None:
            raise RuntimeError("Hybrid LoRA does not support speculative decoding")
        if state(core)["failed"]:
            raise RuntimeError("Hybrid update failed: restart this inference replica")

    def add_request(core, request, request_wave=0):
        validate(core)
        values = state(core)
        adapter = request.lora_request
        if adapter is not None:
            version = values["versions"].get(adapter.lora_name, adapter.lora_path)
            rehash_request(request, version)
            if len(values["requests"]) >= 4096:
                raise RuntimeError("Hybrid lineage storage is full; consume request lineage before adding requests")
            values["requests"][request.request_id] = {
                "request": request,
                "adapter": adapter.lora_name,
                "segments": [{"output_start": 0, "computed_tokens": 0,
                              "version": values["versions"].get(adapter.lora_name, adapter.lora_path)}],
            }
        try:
            return original_add_request(core, request, request_wave)
        except BaseException:
            values["requests"].pop(request.request_id, None)
            raise

    def abort_requests(core, request_ids):
        result = original_abort_requests(core, request_ids)
        for request_id in request_ids:
            state(core)["requests"].pop(request_id, None)
        return result

    def add_lora(core, request: LoRARequest) -> bool:
        validate(core)
        values = state(core)
        if request.load_inplace and values["pending"] is None:
            raise RuntimeError("Use /litecast/v1/update_lora_inflight for audited in-place updates")
        result = original_add_lora(core, request)
        if not request.load_inplace:
            values["versions"].setdefault(request.lora_name, request.lora_path)
        return result

    def hybrid_prepare(core, name, expected, version):
        validate(core)
        values = state(core)
        if not core.is_scheduler_paused() or core.batch_queue:
            raise RuntimeError("Hybrid update requires a completed keep-state pause and empty batch queue")
        if values["pending"] is not None:
            raise RuntimeError("Another hybrid update is pending")
        if values["versions"].get(name) != expected:
            raise ValueError(f"Adapter {name} is not at expected version {expected}")
        if version == expected:
            raise ValueError("New version must differ from expected version")
        affected = []
        for request in core.scheduler.requests.values():
            if request.lora_request is not None and request.lora_request.lora_name == name:
                affected.append({"request_id": request.request_id, "output_start": request.num_output_tokens,
                                 "computed_tokens": request.num_computed_tokens, "version": version})
        values["pending"] = {"adapter": name, "version": version, "affected": affected}
        return values["pending"]

    def hybrid_commit(core, name, version):
        values = state(core)
        pending = values["pending"]
        if pending is None or (pending["adapter"], pending["version"]) != (name, version):
            raise RuntimeError("Hybrid commit does not match prepared update")
        pending["affected"] = [
            {"request_id": request.request_id, "output_start": request.num_output_tokens,
             "computed_tokens": request.num_computed_tokens, "version": version}
            for request in core.scheduler.requests.values()
            if request.lora_request is not None and request.lora_request.lora_name == name
        ]
        for item in pending["affected"]:
            record = values["requests"].get(item["request_id"])
            if record is None:
                raise RuntimeError("Missing active request lineage")
            rehash_request(record["request"], version, switch=True,
                           recompute=record["request"].num_computed_tokens == 0)
            record["segments"].append({k: v for k, v in item.items() if k != "request_id"})
        values["versions"][name] = version
        values["pending"] = None
        return pending

    def hybrid_fail(core):
        state(core)["failed"] = True

    def hybrid_lineage(core, request_id, consume=True):
        values = state(core)
        matches = [(key, record) for key, record in values["requests"].items()
                   if getattr(record["request"], "_hybrid_external_id", key) == request_id]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("Ambiguous external request ID for hybrid lineage")
        internal_id, record = matches[0]
        request = record["request"]
        result = {"semantics": "retained_state", "adapter": record["adapter"],
                  "segments": record["segments"], "output_tokens": request.num_output_tokens,
                  "scheduler_preemptions": request.num_preemptions}
        if consume:
            del values["requests"][internal_id]
        return result

    from vllm.v1.core.sched.scheduler import Scheduler

    original_preempt = Scheduler._preempt_request

    def preempt(scheduler, request, timestamp):
        result = original_preempt(scheduler, request, timestamp)
        if hasattr(request, "_hybrid_cache_history"):
            version = request._hybrid_cache_history[-1]["version"]
            # Recomputed state is a pure current-version prefix, not retained hybrid state.
            rehash_request(request, version, recompute=True)
        return result

    Scheduler._preempt_request = preempt
    def hybrid_status(core):
        values = state(core)
        return {"versions": values["versions"], "failed": values["failed"],
                "active": [{"request_id": request.request_id, "adapter": request.lora_request.lora_name,
                            "output_tokens": request.num_output_tokens, "computed_tokens": request.num_computed_tokens}
                           for request in core.scheduler.requests.values() if request.lora_request is not None]}

    EngineCore.hybrid_status = hybrid_status
    EngineCore.abort_requests = abort_requests
    EngineCore.add_request = add_request
    EngineCore.add_lora = add_lora
    EngineCore.hybrid_prepare = hybrid_prepare
    EngineCore.hybrid_commit = hybrid_commit
    EngineCore.hybrid_fail = hybrid_fail
    EngineCore.hybrid_lineage = hybrid_lineage
    EngineCore._prime_hybrid_lora = True


class HybridFailureFence:
    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.state.hybrid_failed:
            response = JSONResponse({"error": "hybrid update failed; restart replica"}, status_code=503)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def add_routes(app):
    if not enabled():
        return
    app.state.hybrid_update_lock = asyncio.Lock()
    app.state.hybrid_failed = False

    # vLLM eagerly constructs its middleware stack before returning build_app.
    # Install the outer fence during app construction without rebuilding it.
    if app.middleware_stack is None:
        app.add_middleware(HybridFailureFence, state=app.state)
    else:
        app.middleware_stack = HybridFailureFence(app.middleware_stack, app.state)

    @app.get("/litecast/v1/hybrid_status")
    async def status(request: Request):
        return await request.app.state.engine_client.engine_core.call_utility_async("hybrid_status")

    @app.post("/litecast/v1/update_lora_inflight")
    async def update(body: HybridUpdate, request: Request):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest

        path = Path(body.lora_path)
        if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
            raise HTTPException(400, "LoRA payload must be staged and validated before swapping")
        engine = request.app.state.engine_client
        models = request.app.state.openai_serving_models
        async with app.state.hybrid_update_lock:
            if body.lora_name not in models.lora_requests:
                raise HTTPException(404, "Load the initial adapter before requesting a hybrid update")
            await engine.pause_generation(mode="keep", clear_cache=False)
            prepared = False
            try:
                await engine.engine_core.call_utility_async(
                    "hybrid_prepare", body.lora_name, body.expected_version, body.version
                )
                prepared = True
                result = await models.load_lora_adapter(LoadLoRAAdapterRequest(
                    lora_name=body.lora_name, lora_path=body.lora_path, load_inplace=True
                ))
                if isinstance(result, ErrorResponse):
                    raise RuntimeError(str(result))
                # Reload is a one-shot operation, not a per-decode-step instruction.
                models.lora_requests[body.lora_name].load_inplace = False
                pinned = await engine.pin_lora(models.lora_requests[body.lora_name].lora_int_id)
                if not pinned:
                    raise RuntimeError("Could not pin updated adapter against stale-path cache reload")
                event = await engine.engine_core.call_utility_async("hybrid_commit", body.lora_name, body.version)
            except BaseException:
                if prepared:
                    app.state.hybrid_failed = True
                    await engine.engine_core.call_utility_async("hybrid_fail")
                else:
                    await engine.resume_generation()
                raise
            await engine.resume_generation()
            return {"status": "updated", "retained_state": True, **event}
