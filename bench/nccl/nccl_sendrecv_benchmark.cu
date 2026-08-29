#include <cuda_runtime_api.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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
  int source_rank = -1;
  int destination_rank = -1;
  std::string route;
};

struct Options {
  std::array<RoleDevice, 4> roles{{{"A0", "", -1},
                                   {"A1", "", -1},
                                   {"B0", "", -1},
                                   {"B1", "", -1}}};
  bool validate_only = false;
  bool cross_only = false;
  int trials = 7;
  int nvb_disable = 0;
  std::string p2p_level;
  std::string topology_dump;
};

void cuda_check(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void nccl_check(ncclResult_t status, std::string_view operation) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             ncclGetErrorString(status));
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
    if (argument == "--cross-only") {
      options.cross_only = true;
      continue;
    }
    if (argument == "--trials" && i + 1 < argc) {
      options.trials = std::stoi(argv[++i]);
      continue;
    }
    if (argument == "--nvb-disable" && i + 1 < argc) {
      options.nvb_disable = std::stoi(argv[++i]);
      continue;
    }
    if (argument == "--p2p-level" && i + 1 < argc) {
      options.p2p_level = argv[++i];
      continue;
    }
    if (argument == "--topology-dump" && i + 1 < argc) {
      options.topology_dump = argv[++i];
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
  if (options.trials < 3) throw std::runtime_error("--trials must be at least 3");
  if (options.nvb_disable != 0 && options.nvb_disable != 1) {
    throw std::runtime_error("--nvb-disable must be 0 or 1");
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
  cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  for (int device = 0; device < count; ++device) {
    char bus_id[32]{};
    cuda_check(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device),
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

void synchronize_pair(const Options& options, const Pair& pair) {
  cuda_check(cudaSetDevice(options.roles[pair.source_rank].device),
             "cudaSetDevice(sync source)");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(source)");
  cuda_check(cudaSetDevice(options.roles[pair.destination_rank].device),
             "cudaSetDevice(sync destination)");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(destination)");
}

void enqueue_sendrecv(const Pair& pair, std::size_t bytes, void* source_data,
                      void* destination_data,
                      const std::array<ncclComm_t, 4>& communicators,
                      const std::array<cudaStream_t, 4>& streams) {
  nccl_check(ncclGroupStart(), "ncclGroupStart");
  nccl_check(ncclSend(source_data, bytes, ncclUint8, pair.destination_rank,
                      communicators[pair.source_rank], streams[pair.source_rank]),
             "ncclSend");
  nccl_check(ncclRecv(destination_data, bytes, ncclUint8, pair.source_rank,
                      communicators[pair.destination_rank],
                      streams[pair.destination_rank]),
             "ncclRecv");
  nccl_check(ncclGroupEnd(), "ncclGroupEnd");
}

void run_pair(const Options& options, const Pair& pair, std::size_t bytes,
              const std::array<ncclComm_t, 4>& communicators,
              const std::array<cudaStream_t, 4>& streams) {
  const auto& source_role = options.roles[pair.source_rank];
  const auto& destination_role = options.roles[pair.destination_rank];
  cuda_check(cudaSetDevice(source_role.device), "cudaSetDevice(source allocation)");
  void* source_data = nullptr;
  cuda_check(cudaMalloc(&source_data, bytes), "cudaMalloc(source)");
  cuda_check(cudaSetDevice(destination_role.device),
             "cudaSetDevice(destination allocation)");
  void* destination_data = nullptr;
  cuda_check(cudaMalloc(&destination_data, bytes), "cudaMalloc(destination)");

  const auto expected = make_pattern(bytes, pair.source_rank, pair.destination_rank);
  cuda_check(cudaSetDevice(source_role.device), "cudaSetDevice(source initialize)");
  cuda_check(cudaMemcpy(source_data, expected.data(), bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(source initialize)");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(source initialize)");
  cuda_check(cudaSetDevice(destination_role.device),
             "cudaSetDevice(destination initialize)");
  cuda_check(cudaMemset(destination_data, 0, bytes), "cudaMemset(destination)");
  cuda_check(cudaDeviceSynchronize(),
             "cudaDeviceSynchronize(destination initialize)");

  try {
    enqueue_sendrecv(pair, bytes, source_data, destination_data, communicators,
                     streams);
    synchronize_pair(options, pair);
    std::vector<std::uint8_t> actual(bytes);
    cuda_check(cudaSetDevice(destination_role.device),
               "cudaSetDevice(validation result)");
    cuda_check(cudaMemcpy(actual.data(), destination_data, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(validation result)");
    if (actual != expected) {
      const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
      throw std::runtime_error(
          "NCCL payload validation failed for " + source_role.role + "->" +
          destination_role.role + " at byte " +
          std::to_string(static_cast<std::size_t>(mismatch.first - actual.begin())));
    }

    if (!options.validate_only) {
      for (int warmup = 0; warmup < 5; ++warmup) {
        enqueue_sendrecv(pair, bytes, source_data, destination_data, communicators,
                         streams);
      }
      synchronize_pair(options, pair);
      const std::size_t target_bytes = 512ULL * 1024ULL * 1024ULL;
      const int copies = std::clamp<int>(
          static_cast<int>(target_bytes / bytes), 4, 1000);
      std::vector<double> samples;
      samples.reserve(static_cast<std::size_t>(options.trials));
      for (int trial = 0; trial < options.trials; ++trial) {
        const auto begin = std::chrono::steady_clock::now();
        for (int copy = 0; copy < copies; ++copy) {
          enqueue_sendrecv(pair, bytes, source_data, destination_data,
                           communicators, streams);
        }
        synchronize_pair(options, pair);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count() /
            copies);
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
                << json_string(destination_role.role) << ",\"route\":"
                << json_string(pair.route) << ",\"bytes\":" << bytes
                << ",\"nvb_disable\":" << options.nvb_disable
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
    cuda_check(cudaSetDevice(destination_role.device),
               "cudaSetDevice(destination free error)");
    (void)cudaFree(destination_data);
    cuda_check(cudaSetDevice(source_role.device), "cudaSetDevice(source free error)");
    (void)cudaFree(source_data);
    throw;
  }

  cuda_check(cudaSetDevice(destination_role.device),
             "cudaSetDevice(destination free)");
  cuda_check(cudaFree(destination_data), "cudaFree(destination)");
  cuda_check(cudaSetDevice(source_role.device), "cudaSetDevice(source free)");
  cuda_check(cudaFree(source_data), "cudaFree(source)");
}

}  // namespace

int main(int argc, char** argv) {
  std::array<ncclComm_t, 4> communicators{};
  std::array<cudaStream_t, 4> streams{};
  bool initialized = false;
  try {
    auto options = parse_options(argc, argv);
    resolve_devices(options);
    setenv("NCCL_NVB_DISABLE", options.nvb_disable == 0 ? "0" : "1", 1);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,P2P,ENV", 1);
    if (!options.p2p_level.empty()) {
      setenv("NCCL_P2P_LEVEL", options.p2p_level.c_str(), 1);
    }
    if (!options.topology_dump.empty()) {
      setenv("NCCL_TOPO_DUMP_FILE", options.topology_dump.c_str(), 1);
    }

    std::array<int, 4> devices{};
    for (std::size_t rank = 0; rank < options.roles.size(); ++rank) {
      devices[rank] = options.roles[rank].device;
      cuda_check(cudaSetDevice(devices[rank]), "cudaSetDevice(stream create)");
      cuda_check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");
    }
    nccl_check(ncclCommInitAll(communicators.data(), 4, devices.data()),
               "ncclCommInitAll");
    initialized = true;
    int nccl_version = 0;
    nccl_check(ncclGetVersion(&nccl_version), "ncclGetVersion");
    std::cout << "{\"type\":\"metadata\",\"schema\":"
                 "\"numabraid-nccl-sendrecv-v1\",\"nccl_version\":"
              << nccl_version << ",\"nvb_disable\":" << options.nvb_disable
              << ",\"p2p_level\":" << json_string(options.p2p_level)
              << ",\"bandwidth_units\":\"GB/s\","
                 "\"bandwidth_definition\":\"one_way_payload_bytes_per_second\"}\n";

    const std::array<Pair, 12> all_pairs{{
        {0, 1, "direct_nvlink"}, {1, 0, "direct_nvlink"},
        {2, 3, "direct_nvlink"}, {3, 2, "direct_nvlink"},
        {0, 2, "same_numa_pcie"}, {2, 0, "same_numa_pcie"},
        {1, 3, "same_numa_pcie"}, {3, 1, "same_numa_pcie"},
        {0, 3, "cross_numa_nvb_candidate"},
        {3, 0, "cross_numa_nvb_candidate"},
        {2, 1, "cross_numa_nvb_candidate"},
        {1, 2, "cross_numa_nvb_candidate"},
    }};
    const std::array<std::size_t, 5> sizes{{4ULL * 1024ULL, 64ULL * 1024ULL,
                                           1ULL * 1024ULL * 1024ULL,
                                           16ULL * 1024ULL * 1024ULL,
                                           256ULL * 1024ULL * 1024ULL}};
    for (const auto& pair : all_pairs) {
      if (options.cross_only && pair.route != "cross_numa_nvb_candidate") continue;
      for (const std::size_t bytes : sizes) {
        run_pair(options, pair, bytes, communicators, streams);
      }
    }

    for (std::size_t rank = 0; rank < communicators.size(); ++rank) {
      nccl_check(ncclCommDestroy(communicators[rank]), "ncclCommDestroy");
      cuda_check(cudaSetDevice(devices[rank]), "cudaSetDevice(stream destroy)");
      cuda_check(cudaStreamDestroy(streams[rank]), "cudaStreamDestroy");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NCCL send/recv benchmark failed: " << error.what() << '\n';
    if (initialized) {
      for (auto communicator : communicators) (void)ncclCommAbort(communicator);
    }
    return 1;
  }
}
