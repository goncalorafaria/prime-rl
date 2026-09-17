# Klone LiteCast rubric training

`experiment.yaml` requests two H200 GPUs on one host using data parallelism
(CP=1), 16 independent TP=4 inference replicas (64 GPUs total; L40/L40S/A40 pool), eight CPU middles
(16 cores, 64 GB each), 16 restricted terminal workers, and a registry head.
All profiles target Klone; allocations have four-hour limits.

Training uses LoRA rank 32 (alpha 64), 128K packing, 64K rollout context, 32K completion
budget, batch 128, group 16, 128 in-flight episodes, lag 4, LR 2e-6, and KL 0.001.
The optimizer and offload settings are retained from the LoRA experiment.
P2P uses HTTP with source discovery in LiteRegistry, including inference peers.

Input source URIs and verified local copies are recorded in ASSETS.json. The
role stages the checkpoint and DatasetDict under /tmp/litecast-rubrics on each
compute host. The manifest verifies the dataset; checkpoint verification covers
object sizes and safetensors structure, not an upstream cryptographic manifest.

The local PrimeBeaker environment and LiteRegistry tool client are supplied on
PYTHONPATH; their source is not modified. Terminal workers expose the existing
restricted stdin pipeline service through the same LiteRegistry gateway.

Run `rexs validate experiment.yaml --strict` before submission. `preflight.yaml`
loads the dataset, constructs the rubric environment, and validates model/config
schemas on Slurm before the GPU allocation. A successful preflight is not an
end-to-end GPU training result.

Trainer exit owns experiment completion; other roles restart independently.
Intermediate training artifacts are pruned by the existing role supervisor.
`orchestrator.save_train_rollouts` is unsupported in this fork and is omitted.
