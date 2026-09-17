"""Preserve mixed-policy provenance while retaining sampled behavior logprobs."""
from contextvars import ContextVar
import json
import logging

logger = logging.getLogger(__name__)
_lineage = ContextVar('litecast_lineage', default=None)
_result = ContextVar('litecast_result', default=None)


def validate_lineage(lineage, token_count):
    if not lineage or lineage.get('semantics') != 'retained_state':
        raise ValueError('Live adapter response requires retained-state lineage')
    segments = lineage.get('segments', [])
    positions = [segment['output_start'] for segment in segments]
    if not positions or positions[0] != 0 or positions != sorted(positions):
        raise ValueError('Invalid policy transition offsets')
    if any(type(p) is not int or p < 0 or p > token_count for p in positions):
        raise ValueError('Policy transition exceeds generated token count')
    if lineage.get('output_tokens') != token_count or any(not s.get('version') for s in segments):
        raise ValueError('Policy lineage does not match generated tokens')


def install_lineage_tracking():
    import renderers.client as rendering
    from verifiers.clients import renderer_client
    from verifiers.v1.clients import train, config as client_config

    if getattr(rendering, '_litecast_lineage', False):
        return
    original_parse = rendering.parse_generate_response
    original_generate = rendering.generate

    def parse(content):
        value = original_parse(content)
        _lineage.set(value.get('hybrid_lora_lineage'))
        return value

    async def generate(*args, **kwargs):
        token = _lineage.set(None)
        try:
            value = await original_generate(*args, **kwargs)
            if kwargs.get('model', '').endswith('-live'):
                lineage = _lineage.get()
                validate_lineage(lineage, len(value['completion_ids']))
                if len(value['completion_logprobs']) != len(value['completion_ids']):
                    raise ValueError('Mixed-policy training requires sampled logprobs for every token')
                value['litecast_policy_lineage'] = lineage
                _result.set({'request_id':value['request_id'], **lineage})
                logger.info('LITECAST_ROLLOUT_POLICY request_id=%s %s',value['request_id'],json.dumps(lineage))
            return value
        finally:
            _lineage.reset(token)

    class LineageRendererClient(renderer_client.RendererClient):
        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
            value = await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
            lineage = value.get('litecast_policy_lineage')
            state = kwargs.get('state')
            if lineage is not None and state is not None:
                info = state.get('info') or {}
                state['info'] = {**info, 'litecast_policy_lineage': [
                    *info.get('litecast_policy_lineage', []), {'request_id':value['request_id'], **lineage}]}
            return value

    class LineageTrainClient(client_config.TrainClient):
        async def get_response(self, dialect, body, model, sampling_args, session_id=None, turn=None, headers=None):
            token = _result.set(None)
            try:
                response = await super().get_response(dialect, body, model, sampling_args,
                    session_id=session_id, turn=turn, headers=headers)
                lineage = _result.get()
                if lineage is not None and turn is not None:
                    turn.trace.info.setdefault('litecast_policy_lineage', []).append(lineage)
                return response
            finally:
                _result.reset(token)

    client_config.TrainClient = LineageTrainClient
    rendering.parse_generate_response = parse
    rendering.generate = generate
    renderer_client.generate = generate
    train.generate = generate
    renderer_client.RendererClient = LineageRendererClient
    rendering._litecast_lineage = True


def annotate_trace(trace):
    records = trace.info.get('litecast_policy_lineage', [])
    if trace.num_turns and not records:
        raise ValueError('Live-policy rollout lost inference version metadata')
    trace.metrics['litecast/mixed_policy_turns'] = float(sum(len(r['segments']) > 1 for r in records))
    trace.metrics['litecast/policy_transitions'] = float(sum(len(r['segments']) - 1 for r in records))
    trace.metrics['litecast/scheduler_preemptions'] = float(sum(r['scheduler_preemptions'] for r in records))
