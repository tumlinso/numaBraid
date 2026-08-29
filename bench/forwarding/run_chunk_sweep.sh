#!/bin/sh
set -eu

for chunk_bytes in 262144 1048576 4194304 16777216 67108864; do
  for buffers in 2 3; do
    /tmp/numabraid-relay-benchmark \
      --trials 5 \
      --chunk-bytes "$chunk_bytes" \
      --buffers "$buffers" \
      --size-bytes 268435456
  done
done
