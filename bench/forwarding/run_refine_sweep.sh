#!/bin/sh
set -eu

for chunk_bytes in 8388608 12582912 16777216 25165824 33554432; do
  for buffers in 2 3; do
    /tmp/numabraid-relay-benchmark \
      --trials 7 \
      --chunk-bytes "$chunk_bytes" \
      --buffers "$buffers" \
      --size-bytes 268435456
  done
done

for buffers in 1 4; do
  /tmp/numabraid-relay-benchmark \
    --trials 7 \
    --chunk-bytes 16777216 \
    --buffers "$buffers" \
    --size-bytes 268435456
done
