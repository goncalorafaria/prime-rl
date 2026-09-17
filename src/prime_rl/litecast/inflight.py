"""Opt-in stable LoRA slots, with explicit minimum-version routing aliases."""
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from prime_rl.litecast.distribution import blocking, fetch_publication
from prime_rl.litecast.protocol import base_alias, peer_service, split_adapter

logger = logging.getLogger(__name__)


def enabled():
    return os.getenv('LITECAST_INFLIGHT_UPDATES') == '1'


def live_alias(publication):
    return publication.model_name + '-live'


async def reconcile(worker, publications, backend_models):
    slot = base_alias(worker.args.run_id) + '-live'
    current = getattr(worker, 'live_publication', None)
    if slot not in backend_models:
        await worker.withdraw()
        worker.live_publication = current = None
    if not hasattr(worker, 'live_versions'):
        worker.live_versions = {}
    if publications and (current is None or publications[-1].model_name != current.model_name):
        publication = publications[-1]
        started = time.perf_counter()
        transfer_details = {}
        payload = await fetch_publication(publication, worker.registry, worker.root, worker.args.transport,
            source_role='middle' if worker.args.require_middle else None, metrics=transfer_details)
        fetched = time.perf_counter()
        config, weights = split_adapter(payload)
        staging = Path(tempfile.mkdtemp(prefix='adapter-', dir=worker.root))
        (staging/'adapter_config.json').write_bytes(config)
        (staging/'adapter_model.safetensors').write_bytes(weights)
        # Stop new admissions while the engine crosses the update boundary.
        worker.last_healthy = float('-inf')
        await worker.withdraw()
        worker.live_versions[str(staging)] = publication.model_name
        if current is None:
            response = await worker.backend.post('/v1/load_lora_adapter', json={
                'lora_name':slot, 'lora_path':str(staging), 'load_inplace':False})
            expected = str(staging)
        else:
            response = await worker.backend.post('/litecast/v1/update_lora_inflight', json={
                'lora_name':slot, 'lora_path':str(staging),
                'expected_version':worker.live_engine_version, 'version':publication.model_name})
            expected = publication.model_name
        if response.is_error:
            logger.error("In-place adapter update failed; restarting replica: %s", response.text)
            raise SystemExit(1)
        worker.live_engine_version = expected
        worker.live_publication = current = publication
        worker.loaded[publication.model_name] = staging
        worker.transfer_metrics[publication.model_name] = {
            **transfer_details, 'fetch_seconds':fetched-started,'load_seconds':time.perf_counter()-fetched,
            'payload_bytes':publication.size}
        logger.info('LITECAST_TRANSFER model=%s inflight_updates=true metrics=%s',
            publication.model_name, worker.transfer_metrics[publication.model_name])
        version = await blocking(worker.origin.broadcast_buffer,payload,worker.args.shard_bytes,False)
        service = peer_service(worker.args.run_id,publication.digest)
        worker.peers[service] = (await worker.register(service,worker.origin.port,
            litecast_version=version,source_role='peer',digest=publication.digest),version)
    keep = {base_alias(worker.args.run_id)}
    if current:
        # These explicitly named aliases mean "at least this version", never
        # masquerading a mutable adapter as an immutable model identity.
        keep.update(live_alias(p) for p in publications if p.step <= current.step)
        keep.add(current.model_name)  # readiness/transfer acknowledgement only
    for name in list(worker.serving):
        if name not in keep:
            await worker.serving.pop(name).deregister()
    for name in keep:
        if name not in worker.serving:
            metadata = {'base_model':worker.args.base_model}
            if current and name != base_alias(worker.args.run_id):
                metadata.update(step=current.step,digest=current.digest,installed_model=current.model_name,
                    update_semantics='retained_state',litecast_transfer=worker.transfer_metrics[current.model_name])
            worker.serving[name] = await worker.register(name,worker.args.port,**metadata)
    for name in list(worker.loaded):
        if current and name != current.model_name and not worker.inflight.get(name+'-live',0):
            shutil.rmtree(worker.loaded.pop(name))
            worker.transfer_metrics.pop(name,None)
    worker.last_healthy = time.monotonic()


def record_response(worker, payload, requested):
    lineage = payload.get('hybrid_lora_lineage')
    if not lineage:
        raise RuntimeError('Retained-state generation omitted adapter lineage')
    for segment in lineage['segments']:
        segment['version'] = worker.live_versions.get(segment['version'],segment['version'])
    lineage.update(requested_model=requested,replica_id=worker.replica_id)
    # Small provenance records; never persist token arrays or generation states.
    record = {'request_id':payload.get('request_id'),**lineage}
    logger.info('LITECAST_ROLLOUT_POLICY %s',json.dumps(record,separators=(',',':')))
    directory = os.getenv('LITECAST_LINEAGE_DIR')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True,exist_ok=True)
        with (path/(worker.replica_id+'.jsonl')).open('a') as out:
            out.write(json.dumps(record,separators=(',',':'))+'\n')
    return payload
