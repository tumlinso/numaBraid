#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct RoleDevice {
  std::string role;
  std::string pci_bus_id;
  int device = -1;
};

struct Pair {
  int source_role = -1;
  int destination_role = -1;
  std::string route;
};

struct Options {
  std::array<RoleDevice, 4> roles{{{"A0", "", -1},
                                   {"A1", "", -1},
                                   {"B0", "", -1},
                                   {"B1", "", -1}}};
  bool validate_only = false;
  int trials = 7;
};

void check(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::string canonical_pci_id(std::string value) {
  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  if (std::sscanf(value.c_str(), "%x:%x:%x.%x", &domain, &bus, &device,
                  &function) != 4) {
    throw std::runtime_error("invalid PCI identity: " + value);
  }
  char result[32]{};
  std::snprintf(result, sizeof(result), "%04x:%02x:%02x.%x", domain, bus,
                device, function);
  return result;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--validate-only") {
      options.validate_only = true;
      continue;
    }
    if (argument == "--trials" && i + 1 < argc) {
      options.trials = std::stoi(argv[++i]);
      continue;
    }
    bool matched = false;
    for (auto& role : options.roles) {
      std::string flag = "--" + role.role;
      std::transform(flag.begin() + 2, flag.end(), flag.begin() + 2,
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (argument == flag && i + 1 < argc) {
        role.pci_bus_id = canonical_pci_id(argv[++i]);
        matched = true;
        break;
      }
    }
    if (!matched) {
      throw std::runtime_error("unknown or incomplete argument: " +
                               std::string(argument));
    }
  }
  if (options.trials < 3) {
    throw std::runtime_error("--trials must be at least 3");
  }
  for (const auto& role : options.roles) {
    if (role.pci_bus_id.empty()) {
      throw std::runtime_error("all of --a0, --a1, --b0, and --b1 are required");
    }
  }
  return options;
}

void resolve_devices(Options& options) {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  for (int device = 0; device < count; ++device) {
    char bus_id[32]{};
    check(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device),
          "cudaDeviceGetPCIBusId");
    const std::string canonical = canonical_pci_id(bus_id);
    for (auto& role : options.roles) {
      if (role.pci_bus_id == canonical) role.device = device;
    }
  }
  for (const auto& role : options.roles) {
    if (role.device < 0) {
      throw std::runtime_error(role.role + " PCI device " + role.pci_bus_id +
                               " is not CUDA-visible");
    }
  }
}

void enable_peer_if_supported(int source, int destination) {
  int accessible = 0;
  check(cudaDeviceCanAccessPeer(&accessible, source, destination),
        "cudaDeviceCanAccessPeer");
  if (!accessible) return;
  check(cudaSetDevice(source), "cudaSetDevice(peer enable)");
  const cudaError_t status = cudaDeviceEnablePeerAccess(destination, 0);
  if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
    check(status, "cudaDeviceEnablePeerAccess");
  }
  if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
}

std::vector<std::uint8_t> make_pattern(std::size_t bytes, int source,
                                       int destination) {
  std::vector<std::uint8_t> pattern(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    pattern[i] = static_cast<std::uint8_t>(
        (i * 131U + bytes * 17U + source * 29U + destination * 43U) & 0xffU);
  }
  return pattern;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::string json_string(std::string_view value) {
  std::string result = "\"";
  for (const char c : value) {
    if (c == '"' || c == '\\') result.push_back('\\');
    result.push_back(c);
  }
  result.push_back('"');
  return result;
}

__global__ void copy_u32_kernel(std::uint32_t* destination,
                                const std::uint32_t* source,
                                std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    destination[index] = source[index];
  }
}

int copy_blocks(std::size_t bytes, int destination) {
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, destination),
        "cudaGetDeviceProperties(copy kernel)");
  const std::size_t words = bytes / sizeof(std::uint32_t);
  return static_cast<int>(std::min<std::size_t>(
      (words + 511) / 512,
      static_cast<std::size_t>(properties.multiProcessorCount * 8)));
}

void launch_sm_copy(void* destination_data, const void* source_data,
                    std::size_t bytes, int blocks, cudaStream_t stream) {
  copy_u32_kernel<<<blocks, 512, 0, stream>>>(
      static_cast<std::uint32_t*>(destination_data),
      static_cast<const std::uint32_t*>(source_data),
      bytes / sizeof(std::uint32_t));
  check(cudaGetLastError(), "copy_u32_kernel launch");
}

std::vector<double> measure_sm_copy(int destination, void* source_data,
                                    void* destination_data, std::size_t bytes,
                                    int copies_per_trial, int trials,
                                    cudaStream_t stream) {
  check(cudaSetDevice(destination), "cudaSetDevice(SM copy)");
  const int blocks = copy_blocks(bytes, destination);
  cudaEvent_t begin{};
  cudaEvent_t end{};
  check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
  check(cudaEventCreate(&end), "cudaEventCreate(end)");
  for (int i = 0; i < 5; ++i) {
    launch_sm_copy(destination_data, source_data, bytes, blocks, stream);
  }
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SM warmup)");

  std::vector<double> microseconds;
  microseconds.reserve(static_cast<std::size_t>(trials));
  for (int trial = 0; trial < trials; ++trial) {
    check(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
    for (int copy = 0; copy < copies_per_trial; ++copy) {
      launch_sm_copy(destination_data, source_data, bytes, blocks, stream);
    }
    check(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end),
          "cudaEventElapsedTime");
    microseconds.push_back(static_cast<double>(milliseconds) * 1000.0 /
                           copies_per_trial);
  }
  check(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
  check(cudaEventDestroy(end), "cudaEventDestroy(end)");
  return microseconds;
}

std::vector<double> measure_peer_copy(int source, int destination, void* source_data,
                                      void* destination_data, std::size_t bytes,
                                      int copies_per_trial, int trials) {
  for (int i = 0; i < 5; ++i) {
    check(cudaMemcpyPeer(destination_data, destination, source_data, source, bytes),
          "cudaMemcpyPeer(warmup)");
  }
  check(cudaSetDevice(destination), "cudaSetDevice(runtime warmup destination)");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(runtime warmup destination)");
  check(cudaSetDevice(source), "cudaSetDevice(runtime warmup source)");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(runtime warmup source)");

  std::vector<double> microseconds;
  microseconds.reserve(static_cast<std::size_t>(trials));
  for (int trial = 0; trial < trials; ++trial) {
    const auto begin = std::chrono::steady_clock::now();
    for (int copy = 0; copy < copies_per_trial; ++copy) {
      check(cudaMemcpyPeer(destination_data, destination, source_data, source, bytes),
            "cudaMemcpyPeer(measure)");
    }
    check(cudaSetDevice(destination), "cudaSetDevice(runtime trial destination)");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(runtime trial destination)");
    check(cudaSetDevice(source), "cudaSetDevice(runtime trial source)");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(runtime trial source)");
    const auto end = std::chrono::steady_clock::now();
    microseconds.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count() /
        copies_per_trial);
  }
  return microseconds;
}

void verify_copy(int source, int destination, void* source_data,
                 void* destination_data, const std::vector<std::uint8_t>& expected,
                 bool p2p, cudaStream_t stream,
                 std::string_view source_role,
                 std::string_view destination_role) {
  if (p2p) {
    check(cudaSetDevice(destination), "cudaSetDevice(SM validation)");
    launch_sm_copy(destination_data, source_data, expected.size(),
                   copy_blocks(expected.size(), destination), stream);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SM validation)");
  } else {
    check(cudaMemcpyPeer(destination_data, destination, source_data, source,
                         expected.size()),
          "cudaMemcpyPeer(validation fallback)");
  }
  std::vector<std::uint8_t> actual(expected.size());
  check(cudaSetDevice(destination), "cudaSetDevice(validation)");
  check(cudaMemcpy(actual.data(), destination_data, actual.size(),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(validation result)");
  if (actual != expected) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    throw std::runtime_error(
        "payload validation failed for " + std::string(source_role) + "->" +
        std::string(destination_role) + " at byte " +
        std::to_string(static_cast<std::size_t>(mismatch.first - actual.begin())) +
        " of " + std::to_string(expected.size()) + " (actual=" +
        std::to_string(static_cast<unsigned int>(*mismatch.first)) +
        ", expected=" +
        std::to_string(static_cast<unsigned int>(*mismatch.second)) + ")");
  }
}

void run_pair(const Options& options, const Pair& pair, std::size_t bytes) {
  const auto& source_role = options.roles[pair.source_role];
  const auto& destination_role = options.roles[pair.destination_role];
  const int source = source_role.device;
  const int destination = destination_role.device;
  int p2p = 0;
  check(cudaDeviceCanAccessPeer(&p2p, source, destination),
        "cudaDeviceCanAccessPeer(benchmark)");

  check(cudaSetDevice(source), "cudaSetDevice(source allocation)");
  void* source_data = nullptr;
  check(cudaMalloc(&source_data, bytes), "cudaMalloc(source)");
  check(cudaSetDevice(destination), "cudaSetDevice(destination allocation)");
  void* destination_data = nullptr;
  check(cudaMalloc(&destination_data, bytes), "cudaMalloc(destination)");
  cudaStream_t stream = nullptr;

  const auto expected = make_pattern(bytes, source, destination);
  check(cudaSetDevice(source), "cudaSetDevice(source initialize)");
  check(cudaMemcpy(source_data, expected.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(source initialize)");
  // Work launched on the destination device cannot order an H2D transfer in
  // the source device's context. Make the peer-readable source explicit.
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(source initialize)");
  if (options.validate_only) {
    std::vector<std::uint8_t> source_roundtrip(bytes);
    check(cudaMemcpy(source_roundtrip.data(), source_data, bytes,
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy(source roundtrip)");
    if (source_roundtrip != expected) {
      throw std::runtime_error("source initialization validation failed");
    }
  }
  check(cudaSetDevice(destination), "cudaSetDevice(destination initialize)");
  check(cudaMemset(destination_data, 0, bytes), "cudaMemset(destination)");

  try {
    verify_copy(source, destination, source_data, destination_data, expected,
                p2p != 0, stream, source_role.role, destination_role.role);
    if (!options.validate_only) {
      const std::size_t target_bytes = 512ULL * 1024ULL * 1024ULL;
      int copies = std::clamp<int>(
          static_cast<int>(target_bytes / bytes), 4, 5000);
      std::vector<double> samples;
      std::string engine;
      if (p2p) {
        samples = measure_sm_copy(destination, source_data, destination_data,
                                  bytes, copies, options.trials, stream);
        engine = "sm_pull";
      } else {
        copies = std::min(copies, 50);
        samples = measure_peer_copy(source, destination, source_data,
                                    destination_data, bytes, copies,
                                    options.trials);
        engine = "cuda_runtime_no_p2p";
      }
      const double mean =
          std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
      double squared_error = 0.0;
      for (const double sample : samples) {
        squared_error += (sample - mean) * (sample - mean);
      }
      const double standard_deviation =
          std::sqrt(squared_error / static_cast<double>(samples.size()));
      const double median_us = median(samples);
      const double gbps = static_cast<double>(bytes) / median_us / 1000.0;
      std::cout << std::fixed << std::setprecision(6)
                << "{\"type\":\"measurement\",\"source_role\":"
                << json_string(source_role.role) << ",\"destination_role\":"
                << json_string(destination_role.role) << ",\"source_pci\":"
                << json_string(source_role.pci_bus_id)
                << ",\"destination_pci\":"
                << json_string(destination_role.pci_bus_id) << ",\"route\":"
                << json_string(pair.route) << ",\"cuda_p2p\":"
                << (p2p ? "true" : "false") << ",\"engine\":"
                << json_string(engine) << ",\"bytes\":" << bytes
                << ",\"copies_per_trial\":" << copies
                << ",\"trials\":" << options.trials
                << ",\"median_us\":" << median_us << ",\"mean_us\":"
                << mean << ",\"min_us\":"
                << *std::min_element(samples.begin(), samples.end())
                << ",\"max_us\":"
                << *std::max_element(samples.begin(), samples.end())
                << ",\"stddev_us\":" << standard_deviation
                << ",\"payload_gbps\":" << gbps << "}\n";
    }
  } catch (...) {
    check(cudaSetDevice(destination), "cudaSetDevice(destination free error)");
    check(cudaFree(destination_data), "cudaFree(destination error)");
    check(cudaSetDevice(source), "cudaSetDevice(source free error)");
    check(cudaFree(source_data), "cudaFree(source error)");
    throw;
  }

  check(cudaSetDevice(destination), "cudaSetDevice(destination free)");
  check(cudaFree(destination_data), "cudaFree(destination)");
  check(cudaSetDevice(source), "cudaSetDevice(source free)");
  check(cudaFree(source_data), "cudaFree(source)");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto options = parse_options(argc, argv);
    resolve_devices(options);
    for (const auto& source : options.roles) {
      for (const auto& destination : options.roles) {
        if (source.device != destination.device) {
          enable_peer_if_supported(source.device, destination.device);
        }
      }
    }

    const std::array<Pair, 12> pairs{{
        {0, 1, "direct_nvlink"}, {1, 0, "direct_nvlink"},
        {2, 3, "direct_nvlink"}, {3, 2, "direct_nvlink"},
        {0, 2, "same_numa_pcie"}, {2, 0, "same_numa_pcie"},
        {1, 3, "same_numa_pcie"}, {3, 1, "same_numa_pcie"},
        {0, 3, "cross_numa_runtime"}, {3, 0, "cross_numa_runtime"},
        {2, 1, "cross_numa_runtime"}, {1, 2, "cross_numa_runtime"},
    }};
    const std::array<std::size_t, 5> sizes{{4ULL * 1024ULL, 64ULL * 1024ULL,
                                           1ULL * 1024ULL * 1024ULL,
                                           16ULL * 1024ULL * 1024ULL,
                                           256ULL * 1024ULL * 1024ULL}};

    std::cout << "{\"type\":\"metadata\",\"schema\":"
                 "\"numabraid-transport-v1\",\"timing\":"
                 "\"cuda_event_sm_pull_or_steady_clock_runtime_fallback\",\"bandwidth_units\":\"GB/s\","
                 "\"bandwidth_definition\":\"one_way_payload_bytes_per_second\"}\n";
    for (const auto& pair : pairs) {
      for (const std::size_t bytes : sizes) run_pair(options, pair, bytes);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "transport benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
