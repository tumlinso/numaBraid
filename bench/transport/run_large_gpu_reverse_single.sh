#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
benchmark="$root/build-nb08/large_transfer_benchmark"

echo '{"type":"campaign","schema":"numabraid-large-gpu-reverse-single-v1","maximum_bytes":4294967296,"host_chunk_bytes":67108864,"relay_retuned":false}'
for bytes in 268435456 1073741824 4294967296; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --path host-staged --pattern b1-a0 --size-bytes "$bytes" --chunk-bytes 67108864 --warmups "$warmups" --trials "$trials"
  "$benchmark" --path relay --pattern b1-a0 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done
