# Forwarding strategy decision

## Decision: custom-relay

numaBraid will implement a narrow CUDA relay for the validated four-V100 X
topology. The decision is based on measured route semantics, not a preference
for owning more code.

## Evidence

NB-01 validated the physical arrangement by stable PCI identity. Each carrier
has six NVLinks between its NUMA-0 and NUMA-1 GPU, while each same-side pair
shares a PCI host bridge. CUDA peer access is available for the carrier and
same-NUMA pairs and unavailable for the two cross-NUMA diagonals.

NB-02 measured the relevant physical legs at 256 MiB:

| Leg | Mean one-way payload GB/s |
| --- | ---: |
| Direct carrier NVLink, SM pull | 136.88 |
| Same-NUMA PCIe P2P, SM pull | 5.18 |
| Existing cross-NUMA CUDA fallback | 5.81 |

The PCIe P2P leg is the measured ceiling for a compliant steady-state relay.
The cross-NUMA fallback number is retained only as a baseline; it has no route
proof and cannot be treated as satisfying the payload invariant.

NB-03 established that NCCL 2.29.2 understands the hardware graph. With NVB
enabled it changes each diagonal from a three-hop `SYS` path to a three-hop
`PHB` intermediate-GPU candidate. Actual `ncclSend`/`ncclRecv` connection setup,
however, selects `SHM/direct/direct` for every required diagonal. Disabling NVB
does not change that selection or performance.

The narrow specialization experiment `NCCL_P2P_LEVEL=PHB` also retains SHM for
both diagonals, whether NVB is enabled or disabled. At 256 MiB the four-direction
means were 6.00 and 5.98 GB/s respectively, an immaterial difference consistent
with the identical route logs. Stock NCCL therefore sends payload through host
shared memory between opposite NUMA domains rather than through the intended
same-NUMA GPU relay and carrier NVLink.

## Alternatives

### stock-nccl: rejected

Stock NCCL is numerically competitive with the current baseline at about 5.99
GB/s, but its route is wrong for numaBraid's defining invariant. Throughput
alone cannot make a host-SHM path acceptable.

### specialize-nccl: rejected for the tested controls

The installed NCCL contains NVB graph machinery, so a configuration-only route
was tested first. `NCCL_NVB_DISABLE=0` and `NCCL_P2P_LEVEL=PHB` expose the
candidate but do not instantiate it for point-to-point traffic. No supported
configuration or narrow plugin seam was found that replaces this connection's
SHM transport without modifying NCCL itself. Forking NCCL would be broader and
less deterministic than the fixed-topology primitive required here.

### custom-relay: selected

A dedicated relay can explicitly issue the two known GPU-accessible legs:

```text
source -> same-NUMA relay over PCIe P2P -> destination over carrier NVLink
```

This is warranted because the available stock and narrowly configured NCCL
paths fail the route invariant. NB-06 remains intentionally narrow: the relay
is a known physical role, not a general router, endpoint graph, or collective
framework. It must fail explicitly when the validated X relationships are not
present and must not host-stage the optimized path.

## Success criterion

Correctness comes first: exact bytes, awkward sizes, asynchronous safety, and
all four directed diagonals. The initial whole-buffer relay establishes a
correct route. NB-07 then measures chunked overlap against the 5.18 GB/s
same-NUMA PCIe P2P ceiling. Complexity is justified only while it produces a
repeatable improvement under the recorded hardware conditions.
