# Topology

## Intended physical model

Logical names describe physical roles, never CUDA device ordinals:

```text
                         carrier A
NUMA 0                                                     NUMA 1
  PCIe                                                       PCIe
    |                                                          |
   A0 ======================= NVLink ========================== A1

   B0 ======================= NVLink ========================== B1
    |                                                          |
  PCIe                                                       PCIe
NUMA 0                                                     NUMA 1
                         carrier B
```

Each carrier spans the two NUMA roots. GPUs on different carriers but the same NUMA side can use local PCIe P2P. Together those carrier and same-NUMA legs provide the routes called the X topology here:

```text
NUMA 0 side                 carrier NVLink                NUMA 1 side

   A0 ========================== NVLink ======================== A1
    |                                                            |
    | same-NUMA PCIe                              same-NUMA PCIe |
    |                                                            |
   B0 ========================== NVLink ======================== B1
```

The desired logical routes are:

| Source -> destination | Intended route |
| --- | --- |
| `A0 -> A1` | direct carrier-A NVLink |
| `B0 -> B1` | direct carrier-B NVLink |
| `A0 -> B0` | same-NUMA-0 PCIe P2P |
| `A1 -> B1` | same-NUMA-1 PCIe P2P |
| `A0 -> B1` | `A0 -> B0 -> B1` |
| `B0 -> A1` | `B0 -> A0 -> A1` |
| `A1 -> B0` | `A1 -> B1 -> B0` |
| `B1 -> A0` | `B1 -> A1 -> A0` |

Symmetric reverse directions have the same physical legs. Peer-access directionality and bidirectional contention must still be tested.

## Identity and runtime validation

CUDA ordinals can change with visibility settings, enumeration, and process environment. Future detection must correlate CUDA devices with stable PCI identities and validate NUMA locality, CUDA peer access, and NVLink relationships through appropriate CUDA/NVML/system data. It must reject or clearly report a machine that does not provide the expected relationships; it must not infer a carrier solely from an ordinal pair.

## Bootstrap observation requiring reconciliation

On 2026-08-28, the local bootstrap environment reported four Tesla V100-SXM2-16GB devices. `nvidia-smi topo -m` grouped the NVLink pairs as GPU 0/1 with NUMA affinity 0 and GPU 2/3 with NUMA affinity 1; every pair across those groups was reported as `SYS`. That observation does **not** establish the intended carrier-spanning X model above and appears to contradict it if the reported affinities are taken at face value.

This is a recorded observation, not a revised architecture and not a permanent ordinal mapping. `NB-01` must reconcile PCI identities, physical carrier membership, NUMA attachment, peer access, NVLink state, and the intended deployment before any forwarding implementation or topology-dependent performance claim.
