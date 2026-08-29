#include "x_relay.hh"

#include <cuda_runtime_api.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumBytes = 12ULL * kGiB;

void cuda_check(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct FlowSpec {
  std::string name;
  int source_device;
  int destination_device;
  int source_numa;
  int destination_numa;
  int affinity_cpu;
  unsigned char salt;
  numabraid::forwarding::PhysicalRoute relay_route;
};

struct Options {
  std::string path;
  std::string pattern;
  std::size_t bytes{};
  std::size_t chunk_bytes{64ULL * 1024ULL * 1024ULL};
  int warmups{2};
  int trials{5};
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("all benchmark options require values");
    }
    const std::string option = argv[index];
    const std::string value = argv[index + 1];
    if (option == "--path") {
      options.path = value;
    } else if (option == "--pattern") {
      options.pattern = value;
    } else if (option == "--size-bytes") {
      options.bytes = std::stoull(value);
    } else if (option == "--chunk-bytes") {
      options.chunk_bytes = std::stoull(value);
    } else if (option == "--warmups") {
      options.warmups = std::stoi(value);
    } else if (option == "--trials") {
      options.trials = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown benchmark option: " + option);
    }
  }
  if (options.path != "same-numa" && options.path != "host-staged" &&
      options.path != "relay") {
    throw std::invalid_argument(
        "--path must be same-numa, host-staged, or relay");
  }
  if (options.bytes == 0 || options.bytes > kMaximumBytes) {
    throw std::invalid_argument("--size-bytes must be in [1, 12 GiB]");
  }
  if (options.chunk_bytes == 0) {
    throw std::invalid_argument("--chunk-bytes must be nonzero");
  }
  if (options.warmups < 1 || options.trials < 3) {
    throw std::invalid_argument("at least one warmup and three trials required");
  }
  return options;
}

std::vector<FlowSpec> flows_for(std::string_view pattern) {
  const FlowSpec a0_b0{"A0->B0", 0, 1, 0, 0, 0, 0x19, {}};
  const FlowSpec a1_b1{"A1->B1", 2, 3, 1, 1, 20, 0x2b, {}};
  const FlowSpec a0_b1{"A0->B1", 0, 3, 0, 1, 20, 0x3d,
                       {"0000:02:00.0", "0000:03:00.0", "0000:84:00.0"}};
  const FlowSpec b0_a1{"B0->A1", 1, 2, 0, 1, 21, 0x4f,
                       {"0000:03:00.0", "0000:02:00.0", "0000:83:00.0"}};
  const FlowSpec b1_a0{"B1->A0", 3, 0, 1, 0, 0, 0x61,
                       {"0000:84:00.0", "0000:83:00.0", "0000:02:00.0"}};
  if (pattern == "a0-b0") return {a0_b0};
  if (pattern == "a1-b1") return {a1_b1};
  if (pattern == "a0-b1") return {a0_b1};
  if (pattern == "b0-a1") return {b0_a1};
  if (pattern == "b1-a0") return {b1_a0};
  if (pattern == "a0-b1+b0-a1") return {a0_b1, b0_a1};
  if (pattern == "a0-b1+b1-a0") return {a0_b1, b1_a0};
  throw std::invalid_argument("unknown --pattern: " + std::string(pattern));
}

void pin_to_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (status != 0) {
    throw std::runtime_error("pthread_setaffinity_np: " +
                             std::string(std::strerror(status)));
  }
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

int kernel_blocks(int device) {
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  return properties.multiProcessorCount * 16;
}

struct DeviceAllocation {
  int source_device{-1};
  int destination_device{-1};
  void* source{};
  void* destination{};

  DeviceAllocation(const FlowSpec& flow, std::size_t bytes)
      : source_device(flow.source_device),
        destination_device(flow.destination_device) {
    try {
      cuda_check(cudaSetDevice(source_device), "cudaSetDevice(source allocate)");
      auto status = cudaMalloc(&source, bytes);
      if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc source device " +
                                 std::to_string(source_device) + " bytes " +
                                 std::to_string(bytes) + ": " +
                                 cudaGetErrorString(status));
      }
      initialize_pattern<<<kernel_blocks(source_device), 256>>>(
          static_cast<unsigned char*>(source), bytes, flow.salt);
      cuda_check(cudaGetLastError(), "initialize_pattern source");
      cuda_check(cudaDeviceSynchronize(), "initialize source synchronize");

      cuda_check(cudaSetDevice(destination_device),
                 "cudaSetDevice(destination allocate)");
      status = cudaMalloc(&destination, bytes);
      if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc destination device " +
                                 std::to_string(destination_device) + " bytes " +
                                 std::to_string(bytes) + ": " +
                                 cudaGetErrorString(status));
      }
      cuda_check(cudaMemset(destination, 0, bytes), "cudaMemset destination");
    } catch (...) {
      release();
      throw;
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  ~DeviceAllocation() { release(); }

  void validate(std::size_t bytes, unsigned char salt) const {
    cuda_check(cudaSetDevice(destination_device),
               "cudaSetDevice(destination validate)");
    unsigned long long* device_mismatches = nullptr;
    cuda_check(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)),
               "cudaMalloc mismatch counter");
    cuda_check(cudaMemset(device_mismatches, 0, sizeof(*device_mismatches)),
               "cudaMemset mismatch counter");
    count_mismatches<<<kernel_blocks(destination_device), 256>>>(
        static_cast<const unsigned char*>(destination), bytes, salt,
        device_mismatches);
    cuda_check(cudaGetLastError(), "count_mismatches destination");
    unsigned long long mismatches = 0;
    cuda_check(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy mismatch counter");
    cuda_check(cudaFree(device_mismatches), "cudaFree mismatch counter");
    if (mismatches != 0) {
      throw std::runtime_error("exact-byte validation failed with " +
                               std::to_string(mismatches) + " mismatches");
    }
  }

 private:
  void release() noexcept {
    if (source != nullptr) {
      static_cast<void>(cudaSetDevice(source_device));
      static_cast<void>(cudaFree(source));
      source = nullptr;
    }
    if (destination != nullptr) {
      static_cast<void>(cudaSetDevice(destination_device));
      static_cast<void>(cudaFree(destination));
      destination = nullptr;
    }
  }
};

struct Placement {
  int requested_node{-1};
  std::size_t sampled_pages{};
  std::size_t pages_on_requested_node{};
  std::size_t unknown_pages{};
};

Placement query_placement(void* pointer, std::size_t bytes, int node) {
  constexpr std::size_t samples = 64;
  const long page_size = sysconf(_SC_PAGESIZE);
  std::vector<void*> pages;
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t offset =
        ((bytes - 1) * sample / (samples - 1)) /
        static_cast<std::size_t>(page_size) * static_cast<std::size_t>(page_size);
    pages.push_back(static_cast<unsigned char*>(pointer) + offset);
  }
  std::vector<int> status(pages.size(), -1);
  const long result = move_pages(0, pages.size(), pages.data(), nullptr,
                                 status.data(), 0);
  Placement placement{.requested_node = node, .sampled_pages = pages.size()};
  if (result < 0) {
    placement.unknown_pages = pages.size();
    return placement;
  }
  for (const int page_node : status) {
    if (page_node == node) {
      ++placement.pages_on_requested_node;
    } else if (page_node < 0) {
      ++placement.unknown_pages;
    }
  }
  return placement;
}

class PinnedNumaBuffer {
 public:
  PinnedNumaBuffer(std::size_t bytes, int node, int touch_cpu)
      : bytes_(bytes), node_(node) {
    if (numa_available() < 0) throw std::runtime_error("libnuma unavailable");
    pointer_ = numa_alloc_onnode(bytes_, node_);
    if (pointer_ == nullptr) {
      throw std::runtime_error("numa_alloc_onnode failed for " +
                               std::to_string(bytes_) + " bytes on node " +
                               std::to_string(node_));
    }
    try {
      std::thread toucher([&] {
        pin_to_cpu(touch_cpu);
        auto* bytes_data = static_cast<unsigned char*>(pointer_);
        const long page_size = sysconf(_SC_PAGESIZE);
        for (std::size_t offset = 0; offset < bytes_;
             offset += static_cast<std::size_t>(page_size)) {
          bytes_data[offset] = 0;
        }
        bytes_data[bytes_ - 1] = 0;
      });
      toucher.join();
      placement_ = query_placement(pointer_, bytes_, node_);
      cuda_check(cudaHostRegister(pointer_, bytes_, cudaHostRegisterPortable),
                 "cudaHostRegister NUMA staging buffer");
      registered_ = true;
    } catch (...) {
      release();
      throw;
    }
  }

  PinnedNumaBuffer(const PinnedNumaBuffer&) = delete;
  PinnedNumaBuffer& operator=(const PinnedNumaBuffer&) = delete;
  ~PinnedNumaBuffer() { release(); }

  void* data() const noexcept { return pointer_; }
  const Placement& placement() const noexcept { return placement_; }

 private:
  void release() noexcept {
    if (registered_) {
      static_cast<void>(cudaHostUnregister(pointer_));
      registered_ = false;
    }
    if (pointer_ != nullptr) {
      numa_free(pointer_, bytes_);
      pointer_ = nullptr;
    }
  }

  std::size_t bytes_{};
  int node_{};
  void* pointer_{};
  bool registered_{false};
  Placement placement_{};
};

class FlowRunner {
 public:
  virtual ~FlowRunner() = default;
  virtual void transfer() = 0;
  virtual void validate() = 0;
  virtual std::vector<Placement> placements() const { return {}; }
};

class SameNumaRunner final : public FlowRunner {
 public:
  SameNumaRunner(FlowSpec flow, std::size_t bytes)
      : flow_(std::move(flow)), bytes_(bytes), memory_(flow_, bytes_) {
    int can_access = 0;
    cuda_check(cudaDeviceCanAccessPeer(&can_access, flow_.destination_device,
                                       flow_.source_device),
               "cudaDeviceCanAccessPeer same-NUMA");
    if (!can_access) throw std::runtime_error("same-NUMA peer access unavailable");
    cuda_check(cudaSetDevice(flow_.destination_device),
               "cudaSetDevice same-NUMA setup");
    const auto peer = cudaDeviceEnablePeerAccess(flow_.source_device, 0);
    if (peer == cudaSuccess) {
      enabled_here_ = true;
    } else if (peer == cudaErrorPeerAccessAlreadyEnabled) {
      static_cast<void>(cudaGetLastError());
    } else {
      cuda_check(peer, "cudaDeviceEnablePeerAccess same-NUMA");
    }
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreate same-NUMA");
  }

  ~SameNumaRunner() override {
    static_cast<void>(cudaSetDevice(flow_.destination_device));
    if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
    if (enabled_here_) {
      static_cast<void>(cudaDeviceDisablePeerAccess(flow_.source_device));
    }
  }

  void transfer() override {
    pin_to_cpu(flow_.affinity_cpu);
    cuda_check(cudaSetDevice(flow_.destination_device),
               "cudaSetDevice same-NUMA transfer");
    cuda_check(cudaMemcpyPeerAsync(memory_.destination, flow_.destination_device,
                                   memory_.source, flow_.source_device, bytes_,
                                   stream_),
               "cudaMemcpyPeerAsync same-NUMA");
    cuda_check(cudaStreamSynchronize(stream_),
               "cudaStreamSynchronize same-NUMA");
  }

  void validate() override { memory_.validate(bytes_, flow_.salt); }

 private:
  FlowSpec flow_;
  std::size_t bytes_{};
  DeviceAllocation memory_;
  cudaStream_t stream_{};
  bool enabled_here_{false};
};

class RelayRunner final : public FlowRunner {
 public:
  RelayRunner(FlowSpec flow, std::size_t bytes)
      : flow_(std::move(flow)),
        bytes_(bytes),
        memory_(flow_, bytes_),
        relay_(flow_.relay_route,
               {bytes_, numabraid::forwarding::default_chunk_bytes,
                numabraid::forwarding::default_staging_buffers}) {}

  void transfer() override {
    pin_to_cpu(flow_.affinity_cpu);
    relay_.enqueue(memory_.source, memory_.destination, bytes_);
    relay_.synchronize();
  }

  void validate() override { memory_.validate(bytes_, flow_.salt); }

 private:
  FlowSpec flow_;
  std::size_t bytes_{};
  DeviceAllocation memory_;
  numabraid::forwarding::XRelay relay_;
};

class HostStagedRunner final : public FlowRunner {
 public:
  HostStagedRunner(FlowSpec flow, std::size_t bytes, std::size_t chunk_bytes)
      : flow_(std::move(flow)),
        bytes_(bytes),
        chunk_(std::min(bytes_, chunk_bytes)),
        memory_(flow_, bytes_) {
    constexpr unsigned int buffers = 2;
    for (unsigned int slot = 0; slot < buffers; ++slot) {
      local_.push_back(std::make_unique<PinnedNumaBuffer>(
          chunk_, flow_.source_numa, flow_.source_numa == 0 ? slot : 20 + slot));
      remote_.push_back(std::make_unique<PinnedNumaBuffer>(
          chunk_, flow_.destination_numa,
          flow_.destination_numa == 0 ? slot : 20 + slot));
    }
    try {
      cuda_check(cudaSetDevice(flow_.source_device),
                 "cudaSetDevice host-staged source setup");
      cuda_check(cudaStreamCreateWithFlags(&d2h_stream_, cudaStreamNonBlocking),
                 "cudaStreamCreate host-staged D2H");
      for (auto& event : d2h_done_) {
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreate host-staged D2H");
      }
      cuda_check(cudaSetDevice(flow_.destination_device),
                 "cudaSetDevice host-staged destination setup");
      cuda_check(cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking),
                 "cudaStreamCreate host-staged H2D");
      for (auto& event : h2d_done_) {
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreate host-staged H2D");
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~HostStagedRunner() override { cleanup(); }

  void transfer() override {
    pin_to_cpu(flow_.affinity_cpu);
    const std::size_t chunks = (bytes_ + chunk_ - 1) / chunk_;
    const std::size_t primed = std::min<std::size_t>(2, chunks);
    for (std::size_t sequence = 0; sequence < primed; ++sequence) {
      enqueue_d2h(sequence);
    }
    for (std::size_t sequence = 0; sequence < chunks; ++sequence) {
      const auto slot = sequence % 2;
      cuda_check(cudaSetDevice(flow_.source_device),
                 "cudaSetDevice host-staged D2H wait");
      cuda_check(cudaEventSynchronize(d2h_done_[slot]),
                 "cudaEventSynchronize host-staged D2H");
      const std::size_t current =
          std::min(chunk_, bytes_ - sequence * chunk_);
      std::memcpy(remote_[slot]->data(), local_[slot]->data(), current);

      cuda_check(cudaSetDevice(flow_.destination_device),
                 "cudaSetDevice host-staged H2D enqueue");
      cuda_check(cudaMemcpyAsync(
                     static_cast<unsigned char*>(memory_.destination) +
                         sequence * chunk_,
                     remote_[slot]->data(), current, cudaMemcpyHostToDevice,
                     h2d_stream_),
                 "cudaMemcpyAsync host-staged H2D");
      cuda_check(cudaEventRecord(h2d_done_[slot], h2d_stream_),
                 "cudaEventRecord host-staged H2D");

      const std::size_t next = sequence + 2;
      if (next < chunks) {
        cuda_check(cudaEventSynchronize(h2d_done_[slot]),
                   "cudaEventSynchronize host-staged slot reuse");
        enqueue_d2h(next);
      }
    }
    cuda_check(cudaSetDevice(flow_.destination_device),
               "cudaSetDevice host-staged completion");
    cuda_check(cudaStreamSynchronize(h2d_stream_),
               "cudaStreamSynchronize host-staged completion");
  }

  void validate() override { memory_.validate(bytes_, flow_.salt); }

  std::vector<Placement> placements() const override {
    std::vector<Placement> result;
    for (const auto& buffer : local_) result.push_back(buffer->placement());
    for (const auto& buffer : remote_) result.push_back(buffer->placement());
    return result;
  }

 private:
  void enqueue_d2h(std::size_t sequence) {
    const auto slot = sequence % 2;
    const std::size_t current =
        std::min(chunk_, bytes_ - sequence * chunk_);
    cuda_check(cudaSetDevice(flow_.source_device),
               "cudaSetDevice host-staged D2H enqueue");
    cuda_check(cudaMemcpyAsync(
                   local_[slot]->data(),
                   static_cast<const unsigned char*>(memory_.source) +
                       sequence * chunk_,
                   current, cudaMemcpyDeviceToHost, d2h_stream_),
               "cudaMemcpyAsync host-staged D2H");
    cuda_check(cudaEventRecord(d2h_done_[slot], d2h_stream_),
               "cudaEventRecord host-staged D2H");
  }

  void cleanup() noexcept {
    static_cast<void>(cudaSetDevice(flow_.destination_device));
    if (h2d_stream_ != nullptr) static_cast<void>(cudaStreamSynchronize(h2d_stream_));
    for (auto& event : h2d_done_) {
      if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
      event = nullptr;
    }
    if (h2d_stream_ != nullptr) static_cast<void>(cudaStreamDestroy(h2d_stream_));
    h2d_stream_ = nullptr;
    static_cast<void>(cudaSetDevice(flow_.source_device));
    if (d2h_stream_ != nullptr) static_cast<void>(cudaStreamSynchronize(d2h_stream_));
    for (auto& event : d2h_done_) {
      if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
      event = nullptr;
    }
    if (d2h_stream_ != nullptr) static_cast<void>(cudaStreamDestroy(d2h_stream_));
    d2h_stream_ = nullptr;
  }

  FlowSpec flow_;
  std::size_t bytes_{};
  std::size_t chunk_{};
  DeviceAllocation memory_;
  std::vector<std::unique_ptr<PinnedNumaBuffer>> local_;
  std::vector<std::unique_ptr<PinnedNumaBuffer>> remote_;
  cudaStream_t d2h_stream_{};
  cudaStream_t h2d_stream_{};
  std::array<cudaEvent_t, 2> d2h_done_{};
  std::array<cudaEvent_t, 2> h2d_done_{};
};

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

std::unique_ptr<FlowRunner> make_runner(const Options& options,
                                        const FlowSpec& flow) {
  if (options.path == "same-numa") {
    if (flow.source_numa != flow.destination_numa) {
      throw std::invalid_argument("same-numa path requires same-NUMA pattern");
    }
    return std::make_unique<SameNumaRunner>(flow, options.bytes);
  }
  if (flow.source_numa == flow.destination_numa) {
    throw std::invalid_argument("diagonal path requires cross-NUMA pattern");
  }
  if (options.path == "host-staged") {
    return std::make_unique<HostStagedRunner>(flow, options.bytes,
                                              options.chunk_bytes);
  }
  return std::make_unique<RelayRunner>(flow, options.bytes);
}

struct GroupResult {
  std::vector<std::vector<double>> flow_seconds;
  std::vector<double> group_seconds;
  double cpu_seconds{};
};

void execute_group(std::vector<std::unique_ptr<FlowRunner>>& runners,
                   std::vector<double>* elapsed) {
  std::barrier start(static_cast<std::ptrdiff_t>(runners.size() + 1));
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> threads;
  threads.reserve(runners.size());
  for (std::size_t index = 0; index < runners.size(); ++index) {
    threads.emplace_back([&, index] {
      try {
        start.arrive_and_wait();
        const auto begin = Clock::now();
        runners[index]->transfer();
        const auto end = Clock::now();
        if (elapsed != nullptr) {
          (*elapsed)[index] = std::chrono::duration<double>(end - begin).count();
        }
      } catch (...) {
        std::scoped_lock lock(failure_mutex);
        if (!failure) failure = std::current_exception();
      }
    });
  }
  start.arrive_and_wait();
  for (auto& thread : threads) thread.join();
  if (failure) std::rethrow_exception(failure);
}

GroupResult measure_group(std::vector<std::unique_ptr<FlowRunner>>& runners,
                          const Options& options) {
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    execute_group(runners, nullptr);
  }
  GroupResult result;
  result.flow_seconds.resize(runners.size());
  const double cpu_begin = process_cpu_seconds();
  for (int trial = 0; trial < options.trials; ++trial) {
    std::vector<double> elapsed(runners.size());
    const auto begin = Clock::now();
    execute_group(runners, &elapsed);
    const auto end = Clock::now();
    result.group_seconds.push_back(
        std::chrono::duration<double>(end - begin).count());
    for (std::size_t index = 0; index < elapsed.size(); ++index) {
      result.flow_seconds[index].push_back(elapsed[index]);
    }
  }
  result.cpu_seconds = process_cpu_seconds() - cpu_begin;
  for (auto& runner : runners) runner->validate();
  return result;
}

void print_placement(std::string_view flow_name, const Placement& placement,
                     std::size_t index) {
  std::cout << "{\"type\":\"placement\",\"flow\":\"" << flow_name
            << "\",\"buffer_index\":" << index
            << ",\"requested_node\":" << placement.requested_node
            << ",\"sampled_pages\":" << placement.sampled_pages
            << ",\"pages_on_requested_node\":"
            << placement.pages_on_requested_node
            << ",\"unknown_pages\":" << placement.unknown_pages << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto flows = flows_for(options.pattern);
    std::vector<std::unique_ptr<FlowRunner>> runners;
    runners.reserve(flows.size());
    for (const auto& flow : flows) runners.push_back(make_runner(options, flow));

    std::cout << "{\"type\":\"metadata\",\"schema\":\"numabraid-large-transfer-v1\""
              << ",\"path\":\"" << options.path << "\",\"pattern\":\""
              << options.pattern << "\",\"bytes\":" << options.bytes
              << ",\"binary_size_units\":true,\"maximum_bytes\":"
              << kMaximumBytes << ",\"flow_count\":" << flows.size()
              << ",\"warmups\":" << options.warmups
              << ",\"trials\":" << options.trials
              << ",\"host_chunk_bytes\":" << options.chunk_bytes
              << ",\"host_staging_buffers_per_node_per_flow\":2"
              << ",\"relay_chunk_bytes\":"
              << numabraid::forwarding::default_chunk_bytes
              << ",\"relay_staging_buffers\":"
              << numabraid::forwarding::default_staging_buffers
              << ",\"allocation_and_initialization_timed\":false}\n";
    for (std::size_t flow = 0; flow < runners.size(); ++flow) {
      const auto placements = runners[flow]->placements();
      for (std::size_t index = 0; index < placements.size(); ++index) {
        print_placement(flows[flow].name, placements[index], index);
      }
    }

    const GroupResult result = measure_group(runners, options);
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t index = 0; index < flows.size(); ++index) {
      const auto& values = result.flow_seconds[index];
      const double med = median(values);
      std::cout << "{\"type\":\"flow_measurement\",\"path\":\""
                << options.path << "\",\"pattern\":\"" << options.pattern
                << "\",\"flow\":\"" << flows[index].name
                << "\",\"bytes\":" << options.bytes
                << ",\"source_device\":" << flows[index].source_device
                << ",\"destination_device\":"
                << flows[index].destination_device
                << ",\"source_numa\":" << flows[index].source_numa
                << ",\"destination_numa\":" << flows[index].destination_numa
                << ",\"affinity_cpu\":" << flows[index].affinity_cpu
                << ",\"median_seconds\":" << med
                << ",\"mean_seconds\":" << mean(values)
                << ",\"min_seconds\":"
                << *std::min_element(values.begin(), values.end())
                << ",\"max_seconds\":"
                << *std::max_element(values.begin(), values.end())
                << ",\"stddev_seconds\":" << stddev(values)
                << ",\"payload_gbps\":"
                << static_cast<double>(options.bytes) / med / 1.0e9
                << ",\"exact_bytes\":true}\n";
    }
    const double group_median = median(result.group_seconds);
    const double total_payload =
        static_cast<double>(options.bytes) * static_cast<double>(flows.size());
    std::cout << "{\"type\":\"group_measurement\",\"path\":\""
              << options.path << "\",\"pattern\":\"" << options.pattern
              << "\",\"bytes_each\":" << options.bytes
              << ",\"flow_count\":" << flows.size()
              << ",\"median_seconds\":" << group_median
              << ",\"mean_seconds\":" << mean(result.group_seconds)
              << ",\"min_seconds\":"
              << *std::min_element(result.group_seconds.begin(),
                                   result.group_seconds.end())
              << ",\"max_seconds\":"
              << *std::max_element(result.group_seconds.begin(),
                                   result.group_seconds.end())
              << ",\"aggregate_payload_gbps\":"
              << total_payload / group_median / 1.0e9
              << ",\"timed_process_cpu_seconds\":" << result.cpu_seconds
              << ",\"cpu_equivalent_cores\":"
              << result.cpu_seconds /
                     std::accumulate(result.group_seconds.begin(),
                                     result.group_seconds.end(), 0.0)
              << ",\"exact_bytes\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "large transfer benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
