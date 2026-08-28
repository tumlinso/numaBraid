# Architecture

## Boundary

numaBraid owns one narrow decision and data path: selecting and driving a validated GPU-to-GPU forwarding route for the dual-NUMA, four-V100 X topology when stock NCCL does not already provide the required behavior. It may own topology-specific policy, relay orchestration, minimal buffers, synchronization, and measurements needed to make that route fast.

CUDA owns devices, contexts, memory, streams, events, peer access, and the execution mechanisms used by any eventual implementation. NCCL owns its collective and point-to-point communication behavior. numaBraid begins with an NCCL-first policy: characterize and configure NCCL before duplicating communication machinery, and use it whenever it meets the required semantics near the physical bottleneck.

## Payload and control planes

MPI may start processes, assign ranks, exchange small topology or capability records, and coordinate control flow. Host memory may support explicit diagnostics or correctness fallbacks. Neither MPI nor host memory is the normal payload transport numaBraid exists to provide.

The principal invariant is:

> For a forwarding path that numaBraid claims to optimize, payload data must not traverse the CPU inter-socket link when the configured local-GPU -> NVLink route is available.

A future correctness or debug fallback may traverse a slower path, but it must be explicit, observable, and reported as non-optimized. It must never silently masquerade as the optimized route.

## Why the package remains small

The useful mechanism depends on a precise physical arrangement. General `Device`, `Endpoint`, `Fabric`, or `Router` hierarchies would encode hardware assumptions that have not been earned and would obscure the hot path. Version 1 therefore does not model arbitrary accelerator graphs, FPGA endpoints, PCIe switch fabrics, distributed nodes, or hypothetical hardware. Larger carrier counts, switch topologies, and FPGA endpoints remain future questions that actual hardware and measurements must force.

The public API is likewise deferred. The bootstrap exports only version identity to prove header and optional module consumption. `NB-01` through `NB-03` establish facts; `NB-05` decides whether numaBraid should configure stock NCCL, narrowly specialize NCCL behavior, or own a custom relay.
