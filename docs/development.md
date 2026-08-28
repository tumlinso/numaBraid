# Development

## Build model

numaBraid requires a C++20 compiler and uses ordinary out-of-tree CMake builds. The default target is a header-facing interface library, `numabraid::numabraid`. Tests use CTest and are controlled by `BUILD_TESTING` together with `NUMABRAID_BUILD_TESTS`. Hardware benchmarks are separately controlled by `NUMABRAID_BUILD_BENCHMARKS` and do not exist yet.

```sh
cmake -S . -B build -DNUMABRAID_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The source conventions are `.hh` for ordinary headers, `.cc` for ordinary C++, `.cu` for CUDA translation units, `.cuh` only for genuinely CUDA-specific headers, `.inl` when it materially clarifies inline/template structure, and `.cppm` for the optional public module. Do not introduce `.hpp` or `.cpp` arbitrarily, and do not force C++ modules into CUDA internals.

## CUDA and NCCL discovery

CUDA is enabled only with `NUMABRAID_ENABLE_CUDA=ON`; CMake then discovers and enables its installed CUDA compiler and requires CUDA C++20. NCCL is never vendored or forked. `NUMABRAID_ENABLE_NCCL=ON` searches for the installed `nccl.h` and `libnccl`, with `NCCL_ROOT` as an explicit hint, and provides the internal integration target `numabraid::nccl`. No bootstrap source links it because no implementation exists.

The environment observed during bootstrap on 2026-08-28 provided CMake 3.28.3, GCC 13.3.0, NVIDIA driver 580.173.02, CUDA compiler 12.9.86, NCCL 2.29.2, and four compute-capability 7.0 V100 GPUs. These are observations, not dependency pins. Re-detect the active environment for future builds and measurements.

## Public interfaces

Headers are the default and maximally compatible interface. `NUMABRAID_BUILD_MODULES=ON` adds the opt-in `numabraid::module` target from `modules/numabraid.cppm`. CMake 3.28 or newer and a generator/compiler pair with C++20 module dependency-scanning support are required for that target. Module machinery stays at the public boundary and must not force CUDA translation units or downstream consumers to use modules. Compile tests keep the header and module surfaces aligned as the interface grows.

During bootstrap, GCC 13.3 compiled the module interface and its importing smoke test directly with Modules TS support. The local CMake `Unix Makefiles` generator cannot scan C++ module dependencies, and Ninja was not installed, so the CMake module target could not be validated locally. This is a generator limitation, not a claim that all GCC/CMake module combinations are supported.

## Validation and performance workflow

Ordinary compile/unit tests must run without GPUs. Hardware topology tests belong under a distinct topology category and must clearly skip or report unmet hardware rather than making general CI depend on four V100s. Benchmarks are not tests and must record reproducibility context described in [performance.md](performance.md).

Before work, read `AGENTS.md`, inspect Project Control workspace `numabraid`, and claim a bounded todo through the current coding workflow. Use the current CUDA skill for serious GPU profiling or benchmark campaigns. A performance change requires a pre-change baseline, a post-change measurement under comparable conditions, correctness evidence, and a handoff that identifies variance and validation limits.
