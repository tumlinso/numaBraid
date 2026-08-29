

<!-- todo-orchestrator:v2-managed:start -->
# Todo Orchestrator v2 Projection

Project revision: `39`

## Workstreams
- `NB-00` | kind: task | status: done | parent: - | objective: Establish the compact repository, authoritative documentation, C++20 build skeleton, header compile smoke coverage, optional public module boundary, and todo-orchestrator foundation without implementing forwarding.
- `NB-01` | kind: workstream | status: done | parent: - | objective: Implement a small explicit probe for CUDA GPUs, PCI identities, NUMA locality, CUDA peer access, NVLink pairs, and the carrier relationships required by numaBraid, without building a generic graph framework.
- `NB-02` | kind: workstream | status: done | parent: - | objective: Build the minimal benchmark harness for local PCIe P2P, direct NVLink, cross-NUMA baseline, and relevant NCCL paths across transfer sizes and directions.
- `NB-03` | kind: workstream | status: done | parent: - | objective: Determine and measure how installed NCCL handles the X topology, including any intermediate-GPU or NVB-like forwarding, and answer whether it avoids host inter-socket payload traversal near bottleneck throughput.
- `NB-04` | kind: workstream | status: done | parent: - | objective: Refine the small header-first public boundary, optional module interface, library target, and consumer build integration without designing a forwarding API.
- `NB-05` | kind: validation_task | status: done | parent: - | objective: Use measured topology, transport, and NCCL evidence to select stock NCCL, narrow NCCL specialization, or a dedicated numaBraid relay.
- `NB-06` | kind: workstream | status: done | parent: - | objective: If and only if evidence selects a custom relay, implement the narrow X-path forwarding primitive for maximum throughput.
- `NB-07` | kind: workstream | status: done | parent: - | objective: Empirically tune chunking, staging, overlap, streams, events, and synchronization toward steady-state bottleneck-link throughput.
- `NB-08` | kind: workstream | status: done | parent: - | objective: Measure explicit NUMA-controlled host-staged GPU diagonals, isolated host inter-socket movement, and comparable numaBraid relay behavior through 12 GiB under single-flow and concurrent load without changing relay tuning or physical topology.
<!-- todo-orchestrator:v2-managed:end -->
