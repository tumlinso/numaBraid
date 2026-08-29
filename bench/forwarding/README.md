# Custom relay benchmark campaign

`relay_benchmark.cu` measures the four directed X-topology diagonals using
`XRelay`. It reports JSON Lines, one-way decimal payload GB/s, host-observed
enqueue-to-completion latency, repeated-run statistics, and exact-byte status.
The benchmark uses five warm-ups and keeps performance binaries outside CTest.

The 2026-08-29 campaign established an immutable whole-buffer baseline, swept
chunk size and ring depth, refined the optimum, confirmed the selected setting
across 4 KiB through 256 MiB, and measured simultaneous reverse directions.
Campaign specifications and raw JSONL are retained beside the source. CUDA
controller evidence IDs are listed in `results/2026-08-29-summary.md`.

The selected implementation default is a 12 MiB chunk and two staging buffers.
This is a measured default for the validated V100 X machine, not a universal
CUDA constant. Callers conducting a controlled experiment can pass
`RelayOptions` explicitly.

Build directly on the measured host with:

```sh
nvcc -std=c++20 -arch=sm_70 -O3 -Isrc/forwarding \
  src/forwarding/route_validation.cc src/forwarding/x_relay.cu \
  bench/forwarding/relay_benchmark.cu -lnvidia-ml \
  -o relay_benchmark
```
