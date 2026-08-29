# Large-transfer host/inter-socket comparison — 2026-08-29

## Outcome

The custom-relay decision remains justified. At 1-12 GiB, the unchanged
numaBraid relay sustained 5.17 GB/s average, within about 0.7% of the measured
same-NUMA copy-engine reference. The explicit, route-controlled host pipeline
reached 3.38-3.44 GB/s. Stock NCCL remained faster for one flow at about
6.03 GB/s, but its already-proven `SHM/direct/direct` route uses host memory
across the sockets and does not satisfy numaBraid's payload invariant.

The isolated host NUMA link was not saturated by the tested GPU staging path.
One CPU `memcpy` flow reached about 8.7-8.9 GB/s and two reached about
16.5-16.9 GB/s. The explicit two-GPU host pipeline reached only 6.65 GB/s
aggregate at 12 GiB. Its limiting behavior is therefore in the GPU PCIe/copy
pipeline and its staging synchronization, not in the available bulk QPI
bandwidth measured by the host-only test.

Under the central two-flow NUMA-0-origin case, numaBraid reached 8.69 GB/s at
12 GiB versus 6.65 GB/s for explicit host staging, a 30.7% aggregate advantage.
This does not eliminate all contention: the two relay flows exchange traffic in
opposite directions across the same A0/B0 same-NUMA PCIe relationship, reducing
each flow to about 4.35 GB/s. Opposite diagonals use distinct PCIe sides and
carrier legs; at the largest safe 4 GiB size they retained 10.27 GB/s aggregate
versus 6.43 GB/s for host staging.

## Route definitions and timing

Sizes are binary (`1 GiB = 1024^3` bytes); bandwidth is decimal payload GB/s.
Every requested device payload allocation used the full requested size. Setup,
CUDA context creation, allocation, first touch, registration, initialization,
and final validation were outside the timed interval. Cases below 1 GiB used
two warmups and five timed trials; 1 GiB and larger used one warmup and three
trials. Tables report medians or averages of direction medians.

- **Same-NUMA:** `cudaMemcpyPeerAsync` between A0/B0 or A1/B1, with the
  destination-side control thread pinned to the corresponding NUMA node.
- **Explicit host-staged:** full-size source and destination GPU allocations;
  two 64 MiB pinned buffers on the source NUMA node and two on the destination
  node; D2H into source-local memory, destination-pinned CPU `memcpy` across
  NUMA nodes, then H2D from destination-local memory. The two-slot pipeline
  overlaps the next D2H with CPU copy/H2D work. It is not a serialized
  whole-buffer path, and the 64 MiB chunk was fixed for every comparison.
- **numaBraid:** the existing 12 MiB/two-buffer relay, without retuning:
  source to same-NUMA GPU over PCIe P2P, then relay to destination over NVLink.
- **Host-only:** `numa_alloc_onnode`, affinity-controlled first touch, and one
  destination-local glibc `memcpy` thread per flow.

All host placements were verified with `move_pages`: 18,688 sampled pages were
on their requested node and none were unknown or misplaced. The host-staged
path used 256 MiB pinned memory per flow, or 512 MiB for two flows.

## Single-flow size response

The same-NUMA, explicit-host, and relay columns average the two required
directions. NCCL is the average of A0->B1 and B0->A1 where measured.

| Size | Same-NUMA PCIe | Explicit host-staged | numaBraid relay | Stock NCCL SHM |
| ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 4.289 | 1.901 | 3.823 | — |
| 8 MiB | 5.013 | 2.297 | 4.827 | — |
| 64 MiB | 5.187 | 2.294 | 5.140 | — |
| 256 MiB | 5.204 | 3.353 | 5.162 | 6.009 |
| 1 GiB | 5.208 | 3.380 | 5.169 | 6.023 |
| 2 GiB | 5.209 | 3.391 | 5.171 | — |
| 4 GiB | 5.209 | 3.431 | 5.171 | 6.027 |
| 8 GiB | 5.194 | 3.430 | 5.171 | 6.027 |
| 12 GiB | 5.189 | 3.440 | 5.171 | 6.033 |

The large-transfer paths reached stable plateaus. From 1 to 12 GiB the relay
changed by less than 0.1%, same-NUMA PCIe by less than 0.4%, NCCL by about 0.2%,
and explicit host staging improved by about 1.8%. There is no degradation or
instability near 12 GiB.

## Isolated host NUMA movement

| Size | NUMA0->1 | NUMA1->0 | Two 0->1 aggregate | Opposite aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 256 MiB | 8.763 | 8.722 | 16.492 | 16.632 |
| 1 GiB | 8.696 | 8.707 | 16.779 | 16.734 |
| 2 GiB | 8.830 | 8.593 | 16.632 | 16.655 |
| 4 GiB | 8.877 | 8.829 | 16.727 | 16.750 |
| 8 GiB | 8.768 | 8.718 | 16.865 | 16.909 |
| 12 GiB | 8.726 | 8.840 | 16.891 | 16.881 |

The single-flow plateau is about 8.78 GB/s at 12 GiB. Two-flow scaling at
12 GiB was 96.8% for two NUMA0->1 flows and 96.1% for opposite directions.
The platform therefore has substantially more host inter-socket copy capacity
than the explicit GPU staging pipeline consumed.

## Two simultaneous GPU diagonals

### Two NUMA-0 origins: A0->B1 plus B0->A1

| Size each | Host per flow | Host aggregate | Host scaling | Relay per flow | Relay aggregate | Relay scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 MiB | 3.15 | 6.28 | 93.6% | 4.35 | 8.68 | 84.0% |
| 1 GiB | 3.14 | 6.27 | 92.8% | 4.36 | 8.72 | 84.4% |
| 4 GiB | 3.15 | 6.30 | 91.8% | 4.35 | 8.70 | 84.1% |
| 8 GiB | 3.29-3.30 | 6.57 | 95.8% | 4.35 | 8.69 | 84.0% |
| 12 GiB | 3.33 | 6.65 | 96.7% | 4.35 | 8.69 | 84.0% |

Scaling efficiency is aggregate concurrent throughput divided by the sum of
the measured direction-matched single-flow throughputs. Host staging scales
more efficiently as a percentage, but from a much lower baseline. numaBraid
remains 30.7% faster in aggregate at 12 GiB. Its loss identifies shared A0/B0
same-NUMA PCIe traffic as the remaining contention point; NVLink is not the
bottleneck.

### Opposite diagonals: A0->B1 plus B1->A0

| Size each | Host aggregate | Host scaling | Relay aggregate | Relay scaling |
| ---: | ---: | ---: | ---: | ---: |
| 256 MiB | 6.60 | 98.1% | 10.21 | 98.9% |
| 1 GiB | 6.63 | 97.7% | 10.26 | 99.2% |
| 4 GiB | 6.43 | 94.7% | 10.27 | 99.4% |

Four GiB was the largest safe case for this pattern. Each endpoint GPU is both
source and destination and therefore holds two full payload allocations;
8 GiB would require at least 16 GiB per GPU before CUDA context and relay
staging overhead, exceeding the safely available 15.8 GiB. No smaller size was
substituted and 8/12 GiB were not attempted.

## CPU, memory, and counters

Each host-staged flow occupied approximately one CPU-equivalent core during the
timed interval, principally for the NUMA `memcpy`; two flows occupied about two.
The CUDA and NCCL cases also reported roughly one CPU-equivalent core per
control thread because CUDA/NCCL synchronization actively waits on this
configuration. The evidence therefore does **not** support a claim of lower
process CPU occupancy for numaBraid. Its proven distinction is removal of
payload bytes from host memory and the inter-socket fabric, not zero control
CPU activity.

The kernel exposes QPI and IMC uncore PMUs, but `perf stat` was denied with
`perf_event_paranoid=1` and no effective `CAP_PERFMON`. No sysctl, capability,
driver, or system setting was changed. Direct QPI byte counters are therefore
unavailable in this campaign. Route semantics are explicit and placement is
verified, but the amount of freed QPI traffic cannot be claimed from a hardware
counter.

The largest host-only dual-flow case allocated 48 GiB. Host-staged GPU tests
used at most 512 MiB pinned memory, while full payloads remained on the GPUs.
All allocations succeeded. Available host memory was about 60 GiB before and
59 GiB after; swap was disabled and unused. Per-node free memory recovered to
about 28.0/29.0 GiB, all GPUs recovered to 16,145 MiB free, and no foreign GPU
process was observed. There was no pinned-memory failure, paging, silent size
reduction, or topology change.

## Interpretation

1. The isolated inter-socket host plateau is about 8.7-8.9 GB/s per flow and
   16.5-16.9 GB/s for two flows.
2. Explicit staged GPU diagonals plateau at about 3.44 GB/s single and
   6.65 GB/s for two NUMA0-origin flows.
3. Because the isolated QPI result exceeds both the single and dual staged GPU
   results, PCIe/copy-engine staging and synchronization dominate; QPI capacity
   is not saturated at two flows.
4. The host path is numerically competitive only when represented by NCCL's
   tuned SHM implementation. The explicit reproducible pipeline is slower than
   numaBraid even for one large transfer.
5. numaBraid's benefit is a combination of route determinism, host-fabric
   isolation, higher explicit-route bandwidth, and stronger absolute
   multi-flow throughput. It is not merely a 256 MiB or concurrency-only win.
6. The benefit is not unlimited scaling: two relay paths sharing the A0/B0
   same-NUMA exchange reach 84% scaling. Disjoint opposite paths reach about
   99%.
7. Stock NCCL remains about 16.7% faster than numaBraid for one large transfer,
   but uses the forbidden SHM route. This does not change the custom-relay
   decision.

## Correctness, limits, and provenance

All 138 measured flow cases passed deterministic exact-byte validation,
including the largest successful size for every route/concurrency pattern.
The requested byte count is range-checked against 12 GiB and passed directly
to every allocation and transfer; allocation errors include the exact device
and requested byte count.

Maximum successful sizes:

- Same-NUMA, explicit host-staged, numaBraid, and NCCL single flow: 12 GiB.
- Two independent NUMA0-origin host-staged and numaBraid flows: 12 GiB each.
- Host-only single and two-flow cases: 12 GiB each.
- Opposite GPU diagonals: 4 GiB each, limited prospectively by V100 memory.

CUDA-controller evidence IDs and the before/after environment are recorded in
`2026-08-29-qpi-comparison-large-system.json`. Raw machine-readable results are
the sibling `large-gpu-single.jsonl`, `large-gpu-contention.jsonl`,
`large-gpu-reverse-single.jsonl`, and `large-host-numa.jsonl` files.
