# numaBraid agent contract

## Mission

numaBraid exists to make the existing X topology fast by forwarding cross-NUMA GPU traffic through a NUMA-local GPU and then through NVLink, avoiding the host inter-socket payload path.

## Current hardware mental model

Two NUMA nodes host two dual-V100 carriers. Each carrier contains one GPU attached to each NUMA side, and the two GPUs within a carrier have NVLink. This deliberate arrangement lets a carrier's NUMA-local GPU act as accelerator-side ingress for its partner across NVLink. Physical identities must be discovered; CUDA ordinals are only process-local labels.

The bootstrap machine snapshot in [docs/topology.md](docs/topology.md) does not yet confirm this mental model. Reconcile that discrepancy under `NB-01` before implementing communication behavior.

## Hardware setup precondition

The intended X configuration requires a physical cable/carrier change followed by a host reboot, which the user plans to perform before the next hardware-focused work session. At the start of `NB-01` or any CUDA, NCCL, topology, benchmark, or performance task, inspect the live topology and confirm that it matches the intended carrier-spanning X arrangement. If it does not, stop before topology-dependent implementation or measurement, return to the user, and remind them to perform the physical switch and reboot. Do not infer that the switch happened from this note, and do not attempt to change hardware configuration on the user's behalf.

## Golden rules

- Do not generalize numaBraid into a fabric runtime, arbitrary accelerator graph scheduler, or generic NUMA library.
- Do not design FPGA, PLX/Broadcom switch, generic endpoint, or future-hardware support yet.
- Do not replace or fork NCCL without measured evidence. If stock NCCL solves a case near physical-link speed, use it.
- Do not put MPI, host memory, or the CPU inter-socket link back into the optimized payload path. MPI may remain control plane.
- Benchmark before optimizing and benchmark after optimizing. Record topology and software versions with results.
- Do not infer topology from CUDA device ordinal. Validate PCI identity, NUMA locality, peer access, and NVLink relationships.
- Prefer a small fast implementation over a broad clever abstraction.
- Headers are the default public interface. C++20 modules are optional, opt-in, and public-boundary-only.
- Keep CUDA implementation details out of the public API where possible. Preserve `.hh`, `.cc`, `.cu`, `.cuh`, `.inl`, and `.cppm` conventions.
- Never silently introduce a fallback that violates the performance invariant. Correctness/debug fallback behavior must identify itself as non-optimized.
- Do not begin a custom forwarding primitive until `NB-05` selects it from `NB-01`, `NB-02`, and `NB-03` evidence.

## Agent workflow

1. Read this file and the relevant document under `docs/`.
2. Inspect workspace `numabraid` through Project Control; it is observational only.
3. Follow the currently installed coding/todo workflow skill. Use coding-workflow as the normal front door and todo-orchestrator as authority.
4. Claim one bounded task before editing, honor its declared paths and dependencies, and preserve unrelated work.
5. Use the current CUDA skill for serious GPU profiling, benchmarks, or optimization campaigns.
6. Record measured evidence for every performance decision; distinguish facts, hypotheses, and unvalidated topology assumptions.
7. Keep authoritative todo state current through the workflow. Do not hand-edit SQLite or generated todo projections.
8. Finish with required gates and a clear handoff containing results, limitations, and the next evidence needed.

The project boundary and payload invariant are authoritative in [docs/architecture.md](docs/architecture.md). Measurement rules are in [docs/performance.md](docs/performance.md).
