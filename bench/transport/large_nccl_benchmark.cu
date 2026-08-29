#include <cuda_runtime_api.h>
#include <nccl.h>
#include <pthread.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

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

struct Options {
  std::string pattern;
  std::size_t bytes{};
  int warmups{1};
  int trials{3};
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("options require values");
    const std::string option = argv[index];
    const std::string value = argv[index + 1];
    if (option == "--pattern") {
      options.pattern = value;
    } else if (option == "--size-bytes") {
      options.bytes = std::stoull(value);
    } else if (option == "--warmups") {
      options.warmups = std::stoi(value);
    } else if (option == "--trials") {
      options.trials = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (options.pattern != "a0-b1" && options.pattern != "b0-a1") {
    throw std::invalid_argument("--pattern must be a0-b1 or b0-a1");
  }
  if (options.bytes == 0 || options.bytes > 12ULL * kGiB) {
    throw std::invalid_argument("--size-bytes must be in [1, 12 GiB]");
  }
  if (options.warmups < 1 || options.trials < 3) {
    throw std::invalid_argument("at least one warmup and three trials required");
  }
  return options;
}

__device__ unsigned char pattern_byte(std::size_t index, unsigned char salt) {
  return static_cast<unsigned char>((index * 131ULL + (index >> 7U) * 17ULL +
                                     static_cast<std::size_t>(salt)) &
                                    0xffULL);
}

__global__ void initialize_pattern(unsigned char* data, std::size_t bytes,
                                   unsigned char salt) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bytes; index += blockDim.x * gridDim.x) {
    data[index] = pattern_byte(index, salt);
  }
}

__global__ void count_mismatches(const unsigned char* data, std::size_t bytes,
                                 unsigned char salt,
                                 unsigned long long* mismatches) {
  unsigned long long local = 0;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bytes; index += blockDim.x * gridDim.x) {
    local += data[index] != pattern_byte(index, salt);
  }
  if (local != 0) atomicAdd(mismatches, local);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values) {
  const double average = mean(values);
  double total = 0.0;
  for (const double value : values) {
    const double difference = value - average;
    total += difference * difference;
  }
  return std::sqrt(total / static_cast<double>(values.size()));
}

double process_cpu_seconds() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage failed");
  }
  return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1.0e6 +
         usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1.0e6;
}

}  // namespace

int main(int argc, char** argv) {
  std::array<ncclComm_t, 4> communicators{};
  std::array<cudaStream_t, 4> streams{};
  std::array<int, 4> devices{{0, 1, 2, 3}};
  bool initialized = false;
  void* source = nullptr;
  void* destination = nullptr;
  try {
    const Options options = parse_options(argc, argv);
    const int source_rank = options.pattern == "a0-b1" ? 0 : 1;
    const int destination_rank = options.pattern == "a0-b1" ? 3 : 2;
    constexpr unsigned char salt = 0x7d;
    setenv("NCCL_NVB_DISABLE", "0", 1);

    for (int rank = 0; rank < 4; ++rank) {
      cuda_check(cudaSetDevice(devices[rank]), "cudaSetDevice stream setup");
      cuda_check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking),
                 "cudaStreamCreate");
    }
    nccl_check(ncclCommInitAll(communicators.data(), 4, devices.data()),
               "ncclCommInitAll");
    initialized = true;

    cuda_check(cudaSetDevice(source_rank), "cudaSetDevice source allocate");
    auto status = cudaMalloc(&source, options.bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error("cudaMalloc source bytes " +
                               std::to_string(options.bytes) + ": " +
                               cudaGetErrorString(status));
    }
    initialize_pattern<<<1280, 256>>>(static_cast<unsigned char*>(source),
                                      options.bytes, salt);
    cuda_check(cudaGetLastError(), "initialize_pattern");
    cuda_check(cudaDeviceSynchronize(), "initialize source synchronize");
    cuda_check(cudaSetDevice(destination_rank),
               "cudaSetDevice destination allocate");
    status = cudaMalloc(&destination, options.bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error("cudaMalloc destination bytes " +
                               std::to_string(options.bytes) + ": " +
                               cudaGetErrorString(status));
    }
    cuda_check(cudaMemset(destination, 0, options.bytes),
               "cudaMemset destination");

    auto transfer = [&] {
      nccl_check(ncclGroupStart(), "ncclGroupStart");
      nccl_check(ncclSend(source, options.bytes, ncclUint8, destination_rank,
                          communicators[source_rank], streams[source_rank]),
                 "ncclSend");
      nccl_check(ncclRecv(destination, options.bytes, ncclUint8, source_rank,
                          communicators[destination_rank],
                          streams[destination_rank]),
                 "ncclRecv");
      nccl_check(ncclGroupEnd(), "ncclGroupEnd");
      cuda_check(cudaSetDevice(source_rank), "cudaSetDevice send synchronize");
      cuda_check(cudaStreamSynchronize(streams[source_rank]),
                 "cudaStreamSynchronize send");
      cuda_check(cudaSetDevice(destination_rank),
                 "cudaSetDevice receive synchronize");
      cuda_check(cudaStreamSynchronize(streams[destination_rank]),
                 "cudaStreamSynchronize receive");
    };
    for (int warmup = 0; warmup < options.warmups; ++warmup) transfer();
    std::vector<double> elapsed;
    const double cpu_begin = process_cpu_seconds();
    for (int trial = 0; trial < options.trials; ++trial) {
      const auto begin = Clock::now();
      transfer();
      const auto end = Clock::now();
      elapsed.push_back(std::chrono::duration<double>(end - begin).count());
    }
    const double cpu_used = process_cpu_seconds() - cpu_begin;

    cuda_check(cudaSetDevice(destination_rank), "cudaSetDevice validate");
    unsigned long long* device_mismatches = nullptr;
    cuda_check(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)),
               "cudaMalloc mismatch counter");
    cuda_check(cudaMemset(device_mismatches, 0, sizeof(*device_mismatches)),
               "cudaMemset mismatch counter");
    count_mismatches<<<1280, 256>>>(
        static_cast<const unsigned char*>(destination), options.bytes, salt,
        device_mismatches);
    cuda_check(cudaGetLastError(), "count_mismatches");
    unsigned long long mismatches = 0;
    cuda_check(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy mismatch counter");
    cuda_check(cudaFree(device_mismatches), "cudaFree mismatch counter");
    if (mismatches != 0) {
      throw std::runtime_error("exact-byte validation failed with " +
                               std::to_string(mismatches) + " mismatches");
    }

    int nccl_version = 0;
    nccl_check(ncclGetVersion(&nccl_version), "ncclGetVersion");
    const double med = median(elapsed);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"type\":\"flow_measurement\",\"schema\":\"numabraid-large-nccl-v1\""
              << ",\"path\":\"stock-nccl-shm\",\"pattern\":\""
              << options.pattern << "\",\"bytes\":" << options.bytes
              << ",\"source_device\":" << source_rank
              << ",\"destination_device\":" << destination_rank
              << ",\"warmups\":" << options.warmups
              << ",\"trials\":" << options.trials
              << ",\"median_seconds\":" << med
              << ",\"mean_seconds\":" << mean(elapsed)
              << ",\"min_seconds\":"
              << *std::min_element(elapsed.begin(), elapsed.end())
              << ",\"max_seconds\":"
              << *std::max_element(elapsed.begin(), elapsed.end())
              << ",\"stddev_seconds\":" << stddev(elapsed)
              << ",\"payload_gbps\":"
              << static_cast<double>(options.bytes) / med / 1.0e9
              << ",\"timed_process_cpu_seconds\":" << cpu_used
              << ",\"cpu_equivalent_cores\":"
              << cpu_used / std::accumulate(elapsed.begin(), elapsed.end(), 0.0)
              << ",\"nccl_version\":" << nccl_version
              << ",\"nvb_disable\":0,\"known_transport\":\"SHM/direct/direct\""
              << ",\"exact_bytes\":true}\n";

    cuda_check(cudaSetDevice(destination_rank), "cudaSetDevice free destination");
    cuda_check(cudaFree(destination), "cudaFree destination");
    destination = nullptr;
    cuda_check(cudaSetDevice(source_rank), "cudaSetDevice free source");
    cuda_check(cudaFree(source), "cudaFree source");
    source = nullptr;
    for (int rank = 0; rank < 4; ++rank) {
      nccl_check(ncclCommDestroy(communicators[rank]), "ncclCommDestroy");
      cuda_check(cudaSetDevice(rank), "cudaSetDevice stream destroy");
      cuda_check(cudaStreamDestroy(streams[rank]), "cudaStreamDestroy");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "large NCCL benchmark failed: " << error.what() << '\n';
    if (destination != nullptr) {
      static_cast<void>(cudaFree(destination));
    }
    if (source != nullptr) {
      static_cast<void>(cudaFree(source));
    }
    if (initialized) {
      for (auto communicator : communicators) {
        if (communicator != nullptr) static_cast<void>(ncclCommAbort(communicator));
      }
    }
    return 1;
  }
}
