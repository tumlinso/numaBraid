# Performance contract

Performance is the reason numaBraid exists. No path is called optimized and no throughput claim is made without an empirical baseline on the relevant machine.

For a large transfer of size `S`, the intended path is:

```text
source GPU -> local PCIe P2P -> forwarding GPU -> NVLink -> destination GPU
```

A whole-buffer hop followed by a whole-buffer hop tends toward:

```text
T ~= S / B_pcie + S / B_nvlink
```

A correctly overlapped pipeline should aim toward:

```text
T ~= S / min(B_pcie, B_nvlink) + fill/drain/synchronization overhead
```

The eventual target is forwarded cross-NUMA throughput approaching the achieved throughput of the bottleneck physical leg, ideally ordinary local PCIe P2P throughput when PCIe is the bottleneck. The repository does not invent a GB/s target before measuring this machine.

## Measurement rules

- Record GPU PCI identities, NUMA affinity, peer-access and NVLink relationships, driver/runtime/toolkit versions, NCCL version/configuration, compiler/build settings, clocks or throttling, direction, and contention relevant to the result.
- Measure direct NVLink and same-NUMA PCIe P2P legs independently before interpreting an end-to-end route.
- Establish the existing cross-NUMA path, relevant MPI/host-mediated behavior, stock NCCL behavior, and any NCCL intermediate-GPU/NVB-like behavior before replacing anything.
- Distinguish theoretical signaling or link rate from achieved application payload throughput.
- Treat large-transfer steady-state throughput separately from latency-sensitive small transfers. Sweep several transfer sizes and report latency and throughput.
- Measure relevant unidirectional and bidirectional cases and account for synchronization and pipeline overhead.
- Require before/after measurements for hot-path changes and preserve enough raw context to reproduce comparisons.

Event/stream synchronization, intermediate-buffer placement, chunk size, buffer count, overlap strategy, and whether a custom relay is needed are measured questions for later tasks. They are not bootstrap assumptions. Serious benchmark and profiling campaigns must use the current installed CUDA skill so GPU leases, interference controls, evidence, and profiler practices remain consistent.
