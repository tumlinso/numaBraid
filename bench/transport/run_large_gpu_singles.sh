#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
benchmark="$root/build-nb08/large_transfer_benchmark"
nccl_benchmark="$root/build-nb08/large_nccl_benchmark"

sizes=(1048576 8388608 67108864 268435456 1073741824 2147483648 4294967296 8589934592 12884901888)

echo '{"type":"campaign","schema":"numabraid-large-gpu-singles-v1","maximum_bytes":12884901888,"size_convention":"binary","host_chunk_bytes":67108864,"relay_retuned":false}'
for bytes in "${sizes[@]}"; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --path same-numa --pattern a0-b0 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
  "$benchmark" --path same-numa --pattern a1-b1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
  "$benchmark" --path host-staged --pattern a0-b1 --size-bytes "$bytes" --chunk-bytes 67108864 --warmups "$warmups" --trials "$trials"
  "$benchmark" --path host-staged --pattern b0-a1 --size-bytes "$bytes" --chunk-bytes 67108864 --warmups "$warmups" --trials "$trials"
  "$benchmark" --path relay --pattern a0-b1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
  "$benchmark" --path relay --pattern b0-a1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done

for bytes in 268435456 1073741824 4294967296 8589934592 12884901888; do
  "$nccl_benchmark" --pattern a0-b1 --size-bytes "$bytes" --warmups 1 --trials 3
  "$nccl_benchmark" --pattern b0-a1 --size-bytes "$bytes" --warmups 1 --trials 3
done
