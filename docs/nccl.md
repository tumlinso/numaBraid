# Stock NCCL characterization

## Conclusion

Stock NCCL 2.29.2 recognizes the validated X topology and its NVB graph model
finds the intended intermediate-GPU diagonal. It does **not** instantiate that
path for the measured `ncclSend`/`ncclRecv` transfers. The actual cross-diagonal
transport is `SHM/direct/direct` with NVB enabled, with NVB disabled, and with
`NCCL_P2P_LEVEL=PHB`. Because SHM is host shared-memory transport and the two
endpoint GPUs attach to opposite NUMA nodes, this path does not satisfy
numaBraid's no-host-inter-socket-payload invariant.

The evidence supports a `custom-relay` NB-05 decision. A narrow NCCL setting was
tested before reaching that conclusion; it changed neither the selected
transport nor steady-state performance.

## Machine and software

The run used four Tesla V100-SXM2-16GB GPUs, driver 580.173.02, CUDA toolkit
12.9 (nvcc 12.9.86), NCCL 2.29.2+cuda12.9, and GCC 13.3.0. Each GPU negotiated
PCIe Gen3 x8. The physical roles and stable PCI addresses were:

| Role | CUDA ordinal in this run | PCI identity | NUMA node | Carrier peer |
| --- | ---: | --- | ---: | --- |
| A0 | 0 | `0000:02:00.0` | 0 | A1 over NV6 |
| B0 | 1 | `0000:03:00.0` | 0 | B1 over NV6 |
| A1 | 2 | `0000:83:00.0` | 1 | A0 over NV6 |
| B1 | 3 | `0000:84:00.0` | 1 | B0 over NV6 |

The roles were resolved by PCI identity rather than assumed ordinal order. The
NB-01 observation contains UUIDs, peer-access results, sysfs locality, and the
complete `nvidia-smi topo -m` output.

## Method

`bench/nccl/nccl_sendrecv_benchmark.cu` creates one four-rank communicator and
measures each relevant directed pair. It uses five warm-up copies, seven timed
trials, sizes from 4 KiB through 256 MiB, and exact byte validation after every
case. Bandwidth is one-way payload bytes per elapsed second (decimal GB/s), not
bidirectional aggregate or theoretical signaling rate.

Four isolated CUDA-controller campaigns were run:

| Campaign | Settings | Purpose | Evidence id |
| --- | --- | --- | --- |
| Default | `NCCL_NVB_DISABLE=0` | Full direct, same-NUMA, and diagonal baseline | `20b094fc-11b0-4b64-9c4d-2bc4489ac14c` |
| NVB disabled | `NCCL_NVB_DISABLE=1` | Diagonal control | `663c0ca2-7cd3-4c61-961a-ee71e6ca04e4` |
| PHB + NVB | `NCCL_NVB_DISABLE=0`, `NCCL_P2P_LEVEL=PHB` | Test narrow route specialization | `479e25a8-4cd1-45de-99a9-6f754f82fc7b` |
| PHB, no NVB | `NCCL_NVB_DISABLE=1`, `NCCL_P2P_LEVEL=PHB` | Specialization control | `7db2190e-094f-4fb3-8285-ed4b2477ba7c` |

The harness scoped `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,P2P,SHM`,
and `NCCL_TOPO_DUMP_FILE` to its own process. It did not persist environment
changes in the user's shell. Campaign JSON, machine-readable measurements,
NCCL topology XML, and condensed connection evidence are under
`bench/nccl/results/`.

## Route evidence

NCCL's graph with NVB enabled reports the physical A0-to-B1 diagonal as
`GPU/0-84000 (3/6.0/PHB)`. Disabling NVB changes precisely that graph entry to
`(3/6.0/SYS)`. This proves the installed library sees an intermediate-GPU NVB
candidate rather than misunderstanding the hardware.

Connection setup tells a different story. Carrier NVLink pairs select
`P2P/direct pointer`, the positive control. Same-NUMA PCIe pairs and both
cross-NUMA diagonals select `SHM/direct/direct`. The same SHM selection appears
with NVB disabled and in both `NCCL_P2P_LEVEL=PHB` campaigns. Thus graph path
classification does not imply that the point-to-point transport used the
intermediate GPU.

The endpoint GPUs for each diagonal belong to different NUMA nodes and lack
CUDA peer access. A host SHM payload path between them necessarily violates the
optimized-path invariant: it is not the required source-to-local-relay PCIe P2P
followed by carrier NVLink.

## Results

Large-message results below average the four directed pairs in each route class
at 256 MiB. Ranges show the minimum and maximum directed-pair median, avoiding
precision beyond what the run supports.

| Route / setting | Mean payload GB/s | Directed-pair range GB/s |
| --- | ---: | ---: |
| NCCL direct carrier NVLink | 40.10 | 39.83-40.26 |
| NCCL same-NUMA pair | 6.13 | 6.09-6.15 |
| Cross diagonal, NVB enabled | 5.99 | 5.98-6.01 |
| Cross diagonal, NVB disabled | 5.97 | 5.93-5.99 |
| Cross diagonal, PHB + NVB | 6.00 | 5.98-6.01 |
| Cross diagonal, PHB, no NVB | 5.98 | 5.98-5.99 |

The independent NB-02 physical-leg probe measured 136.88 GB/s average for the
direct NVLink SM-pull legs and 5.18 GB/s for same-NUMA PCIe P2P. Its current
cross-NUMA CUDA fallback measured 5.81 GB/s but did not establish a compliant
route. NCCL's 5.99 GB/s cross result is therefore not evidence of NVLink relay;
the connection log establishes SHM, and the on/off controls are effectively
unchanged. Relative to the measured 5.18 GB/s same-NUMA P2P leg, NCCL reports
116% payload throughput because its host-staged transport can pipeline distinct
PCIe transfers; that number is not a compliant forwarding efficiency.

At 256 MiB, the NVB-enabled and disabled diagonal means differ by only 0.28%,
and the PHB enabled/control means differ by 0.21%. Those differences are too
small to attribute to a routing change and the connection logs prove there was
none.

## Answers required by NB-03

1. **Does NCCL recognize the topology correctly?** Yes. It records the correct
   PCI hierarchy, two NV6 carrier pairs, and—when NVB is enabled—three-hop PHB
   diagonal candidates.
2. **Does it use intermediate-GPU/NVLink forwarding for the required pairs?**
   No, not for these `ncclSend`/`ncclRecv` measurements. It selects SHM.
3. **Does the payload avoid the CPU inter-socket link?** No. The selected host
   SHM transport crosses between GPUs attached to opposite NUMA nodes.
4. **What bandwidth does it achieve?** About 5.99 GB/s one-way at 256 MiB with
   NVB enabled; small-message latency is about 10.7 microseconds at 4 KiB.
5. **How close is it to the measured bottleneck leg?** The numerical ratio is
   116% of the NB-02 5.18 GB/s SM-pull PCIe leg, but it is an invalid comparison
   for route efficiency because NCCL used a different, host-staged pipeline.
6. **Where is the remaining loss?** For the required route, the primary defect
   is route selection itself: the relay route is absent. NCCL's direct NVLink
   send/receive result is also only 29% of the 136.88 GB/s physical probe, but
   that protocol-specific gap is not the limiting leg of the intended relay.
7. **Is the loss large enough to justify numaBraid-specific implementation?**
   Yes. The issue is correctness of the performance path, not merely a small
   throughput delta. The stock and narrowly configured paths violate the core
   payload invariant.

## Limitations

This conclusion applies to NCCL 2.29.2's point-to-point send/receive path on
this host and topology. It does not claim that all NCCL collectives or future
versions behave identically. NCCL's own connection log is direct route evidence,
so an additional traffic profiler was not needed to distinguish SHM from P2P in
this campaign.
