# Transport benchmark

`transport_benchmark.cu` measures the twelve directed physical and diagonal
pairs of the validated X topology. Physical roles are supplied as PCI identities
and resolved at runtime; CUDA ordinals are never treated as stable identities.

The benchmark emits JSON Lines. Each measurement validates exact payload bytes,
warms up the path, records seven batch trials by default, and reports per-copy
latency statistics plus one-way decimal payload GB/s. P2P-capable legs use a
vectorized destination-SM pull kernel, which is relevant to NCCL's GPU-driven
transport and avoids assuming that the copy engine is the only useful path.
`cross_numa_runtime` uses CUDA's synchronous `cudaMemcpyPeer` behavior when
direct peer access is absent; it is a baseline observation, not an optimized
numaBraid route.

Example build and validation for the 2026-08-29 observation:

```sh
nvcc -std=c++20 -arch=sm_70 bench/transport/transport_benchmark.cu -o transport_benchmark
./transport_benchmark --a0 0000:02:00.0 --a1 0000:83:00.0 \
  --b0 0000:03:00.0 --b1 0000:84:00.0 --validate-only
```

The benchmark is intentionally separate from ordinary unit/CTest execution.
