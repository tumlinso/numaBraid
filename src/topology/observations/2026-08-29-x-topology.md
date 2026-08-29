# X-topology observation — 2026-08-29

The live machine satisfies numaBraid's intended four-V100 X arrangement. This
observation was made after the physical carrier switch and reboot, without any
software or hardware reconfiguration.

| Role | CUDA ordinal at observation | PCI identity | UUID | NUMA | Carrier peer |
| --- | ---: | --- | --- | ---: | --- |
| A0 | 0 | `0000:02:00.0` | `GPU-21131915-1488-23af-38dd-1743ae1f5cc8` | 0 | A1 over NV6 |
| A1 | 2 | `0000:83:00.0` | `GPU-d745da9c-7649-8334-41f1-483afa6f3206` | 1 | A0 over NV6 |
| B0 | 1 | `0000:03:00.0` | `GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505` | 0 | B1 over NV6 |
| B1 | 3 | `0000:84:00.0` | `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a` | 1 | B0 over NV6 |

`nvidia-smi topo -m` reported A0/B0 and A1/B1 as `PHB`, carrier pairs as
`NV6`, and A0/B1 plus B0/A1 as `SYS`. Linux sysfs independently reported two
GPUs on NUMA node 0 and two on NUMA node 1. Every GPU negotiated PCIe Gen3 x8
at the time of observation (maximum capability Gen3 x16).

The CUDA P2P matrix is narrower than the physical visibility matrix: CUDA P2P
is available in both directions on the two NVLink pairs and the two same-NUMA
PHB pairs, but unavailable directly on the two `SYS` diagonals. This is the
expected motivation for investigating intermediate-GPU forwarding; the probe
does not treat those diagonals as required direct P2P legs.

Environment: Tesla V100-SXM2-16GB, NVIDIA driver 580.173.02, CUDA runtime
12.9, CUDA toolkit 12.9.86, and GCC 13.3.0. The machine-readable companion is
`2026-08-29-x-topology.json`.
