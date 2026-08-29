#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
benchmark="$root/build-nb08/large_transfer_benchmark"

echo '{"type":"campaign","schema":"numabraid-large-gpu-contention-v1","maximum_bytes":12884901888,"size_convention":"binary","host_chunk_bytes":67108864,"relay_retuned":false}'
for bytes in 268435456 1073741824 4294967296 8589934592 12884901888; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --path host-staged --pattern a0-b1+b0-a1 --size-bytes "$bytes" --chunk-bytes 67108864 --warmups "$warmups" --trials "$trials"
  "$benchmark" --path relay --pattern a0-b1+b0-a1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done

# Opposite directions require two full-size allocations on each endpoint GPU.
# Four GiB is the largest safe requested size on these 16 GiB devices.
for bytes in 268435456 1073741824 4294967296; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --path host-staged --pattern a0-b1+b1-a0 --size-bytes "$bytes" --chunk-bytes 67108864 --warmups "$warmups" --trials "$trials"
  "$benchmark" --path relay --pattern a0-b1+b1-a0 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done
