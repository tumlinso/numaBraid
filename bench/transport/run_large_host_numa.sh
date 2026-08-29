#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
benchmark="$root/build-nb08/numa_host_benchmark"
sizes=(268435456 1073741824 2147483648 4294967296 8589934592 12884901888)

echo '{"type":"campaign","schema":"numabraid-large-host-numa-v1","maximum_bytes":12884901888,"size_convention":"binary","implementation":"glibc memcpy with one pinned destination-local thread per flow"}'
for bytes in "${sizes[@]}"; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --pattern 0-1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
  "$benchmark" --pattern 1-0 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done

for bytes in "${sizes[@]}"; do
  warmups=2
  trials=5
  if (( bytes >= 1073741824 )); then
    warmups=1
    trials=3
  fi
  "$benchmark" --pattern 0-1+0-1 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
  "$benchmark" --pattern 0-1+1-0 --size-bytes "$bytes" --warmups "$warmups" --trials "$trials"
done
