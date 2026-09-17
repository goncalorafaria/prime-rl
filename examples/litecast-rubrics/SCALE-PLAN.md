# Rubric scale deployment — pending inputs

Requested topology: eight L40 training nodes, 64 one-GPU inference replicas
allowed on L40/L40S/A40, eight CPU middle nodes, and one LiteRegistry/gateway
head. This is a preparation document, not a submitted Rex experiment.

Provisional interpretation: eight full L40 nodes, eight GPUs per node, 64 training
GPUs total. Confirm GPUs per training node before compiling the allocation.
Klone advertises ten L40 nodes, each with eight GPUs, plus L40S and A40 hosts.
Inventory does not imply immediately available quota or schedulable capacity.

`trainer-cp8.toml` contains a trainer-only candidate: CP=8, Ulysses, with the
existing LoRA and offload settings. CP ranks should stay within each eight-GPU
node; the other dimension spans eight data-parallel ranks. Validate actual rank
placement, Qwen3.5 hybrid attention boundaries, gradients, and memory on a small
Slurm run before the large submission. No dtype defaults are changed.

Each middle reserves 16 CPU cores and 64 GB RAM, on distinct CPU hosts where
possible. P2P transfers use LiteRegistry discovery and HTTP byte transport, with
workers eligible as sources; UCXX/RDMA has not been validated for this topology.
The terminal service capacity must be measured separately from inference.

Launch dependencies still unresolved:
- Reachable rubric dataset: the supplied prefix lists zero objects on Kopah.
- Reachable sloth2b-sft-step400 checkpoint (currently only a Delta local path).
- services.yaml, socket-env.sh and the terminal service runtime.
- Final training GPU count and LoRA versus full-parameter choice.

The current toy launcher only starts a single-node trainer and hardcodes model
and dataset staging. The production Rex launch needs multi-node rendezvous, a
single orchestrator, asset staging, and terminal-service lifecycle wiring. Do not
scale the toy replica count and treat it as distributed training. Trainer failure
must terminate the experiment; inference and middle replacement must preserve the
run namespace and versioned adapter discovery.
