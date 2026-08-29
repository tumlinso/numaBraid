# Nsight Systems overlap evidence

CUDA-controller evidence `4c924431-9475-4182-a154-077f11d3cea2` captured the
selected 12 MiB/two-buffer configuration with Nsight Systems 2025.3.1. The raw
`report.nsys-rep` and exported SQLite database remain in the controller artifact.

The generic CUDA-skill summary marked the timeline for rerun because it expects
CUDA kernel rows. This application is intentionally copy-only, so an empty
kernel table is expected. The trace contains CUDA runtime, synchronization, and
`CUPTI_ACTIVITY_KIND_MEMCPY` rows and is usable for copy-pipeline analysis; the
throughput claims still come from the repeated benchmark campaigns.

For every physical leg, the trace recorded 176 peer copies and 2,147,483,648
payload bytes. Mean 12 MiB copy durations were:

| Leg | Mean duration per chunk |
| --- | ---: |
| PCIe ingress `0 -> 1` | 2345.1 us |
| PCIe ingress `1 -> 0` | 2370.9 us |
| PCIe ingress `2 -> 3` | 2344.2 us |
| PCIe ingress `3 -> 2` | 2372.7 us |
| NVLink egress `1 -> 3` | 88.6 us |
| NVLink egress `3 -> 1` | 88.5 us |
| NVLink egress `0 -> 2` | 87.5 us |
| NVLink egress `2 -> 0` | 88.1 us |

Joining ingress and egress intervals by the four explicit route leg pairs found
that 98.28-98.29% of total NVLink-egress copy time overlaps a PCIe-ingress copy.
This trace evidence agrees with the one-buffer versus two-buffer performance
control: the ring is overlapping the intended device-to-device legs rather than
merely changing host timing.
