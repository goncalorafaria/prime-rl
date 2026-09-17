"""Per-rollout weak affinity, with prompt keys retained for benchmark comparisons."""

from contextvars import ContextVar
import hashlib
import json

HEADER = "X-Session-ID"
STATE_KEY = "_prime_rl_serving_replica"
_served = ContextVar("litecast_serving_replica", default=None)


def prompt_key(messages, *, initial=False):
    selected = []
    for message in messages:
        value = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
        if not initial and value["role"] in ("assistant", "tool"):
            break
        if value["role"] in ("system", "developer", "user"):
            selected.append(value)
    if not selected or not any(message["role"] == "user" for message in selected):
        return None
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "prompt-sha256:" + hashlib.sha256(encoded).hexdigest()


def install_prompt_affinity():
    """Install per-rollout affinity in spawned env workers (legacy flag name)."""
    import renderers.client as rendering
    from verifiers.clients import renderer_client
    from verifiers.v1.clients import config as client_config

    if getattr(renderer_client.RendererClient, "_prime_rl_affinity", False):
        return
    original_parse = rendering.parse_generate_response
    original_generate = rendering.generate

    def parse(raw):
        value = original_parse(raw)
        _served.set(value.get("litecast_replica_id"))
        return value

    async def generate(*args, **kwargs):
        token = _served.set(None)
        try:
            value = await original_generate(*args, **kwargs)
            replica = _served.get()
            if replica:
                value["litecast_replica_id"] = replica
            return value
        finally:
            _served.reset(token)

    class RolloutRendererClient(renderer_client.RendererClient):
        _prime_rl_affinity = True

        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
            state = kwargs.get("state")
            replica = state.get(STATE_KEY) if state is not None else None
            headers = {k: v for k, v in (kwargs.get("extra_headers") or {}).items()
                       if k.lower() != HEADER.lower()}
            if replica:
                headers[HEADER] = "replica:" + replica
            kwargs["extra_headers"] = headers
            value = await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
            if state is not None:
                state[STATE_KEY] = value.get("litecast_replica_id")
            return value

    class RolloutTrainClient(client_config.TrainClient):
        async def get_response(self, dialect, body, model, sampling_args, session_id=None, turn=None, headers=None):
            state = turn.trace.info if turn is not None else None
            replica = state.get(STATE_KEY) if state is not None else None
            token = _served.set(None)
            try:
                # TrainClient forwards session_id but currently ignores headers.
                value = await super().get_response(
                    dialect, body, model, sampling_args,
                    session_id="replica:" + replica if replica else None, turn=turn, headers=headers)
                if state is not None:
                    state[STATE_KEY] = _served.get()
                return value
            finally:
                _served.reset(token)

    # The v1 client imports generate inside get_response; publish the serving
    # replica to its enclosing request context as well as the legacy return value.
    async def tracked_generate(*args, **kwargs):
        value = await generate(*args, **kwargs)
        _served.set(value.get("litecast_replica_id"))
        return value

    rendering.parse_generate_response = parse
    rendering.generate = tracked_generate
    renderer_client.generate = tracked_generate
    renderer_client.RendererClient = RolloutRendererClient
    client_config.TrainClient = RolloutTrainClient
