# numaBraid

numaBraid is a small C++20/CUDA package for one communication problem: avoid CPU inter-socket payload traversal in a dual-NUMA, four-GPU X topology. Each dual-GPU carrier spans the NUMA domains and provides NVLink between its GPUs. For a cross path such as `A0 -> B1`, the desired route is `A0 -> B0` over same-NUMA PCIe P2P, followed by `B0 -> B1` over the destination carrier's NVLink.

The alternative path through host PCIe, the CPU inter-socket link, and the remote PCIe root wastes the accelerator-side route that the carrier topology provides. numaBraid therefore treats the NUMA-local peer GPU as a forwarding extension of the destination GPU.

numaBraid is NCCL-adjacent, not an NCCL replacement. The first engineering work is to validate the physical topology and measure what the installed NCCL already does. A custom relay is justified only if evidence shows a semantic or performance gap.

The scope is intentionally narrow: NVIDIA V100 GPUs, CUDA, NCCL, GPU P2P, NVLink, PCIe, NUMA locality, and this X-shaped arrangement. Version 1 is not a generic fabric runtime, FPGA runtime, PCIe router or switch manager, distributed multi-node system, CellShard component, generic NUMA library, or abstraction for hypothetical hardware.

## Build

The bootstrap is header-only and does not require CUDA or NCCL to compile:

```sh
cmake -S . -B build -DNUMABRAID_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA and NCCL discovery seams are opt-in until an implementation needs them:

```sh
cmake -S . -B build-cuda -DNUMABRAID_ENABLE_CUDA=ON -DNUMABRAID_ENABLE_NCCL=ON
```

The default public boundary is:

```cpp
#include <numabraid/numabraid.hh>
```

An optional C++20 module boundary is represented by `import numabraid;`. Enable it with `-DNUMABRAID_BUILD_MODULES=ON` using a CMake generator and compiler combination that supports C++20 module dependency scanning. It remains off by default and does not change header consumers.

## Status

This repository is an engineering foundation only. It contains no forwarding primitive and makes no throughput claim. Current work is topology validation, transport baselines, stock NCCL characterization, and stabilization of the minimal public/build boundary. See [architecture](docs/architecture.md), [topology](docs/topology.md), [performance](docs/performance.md), and [development](docs/development.md).
