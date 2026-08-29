#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumBytes = 12ULL * kGiB;

struct Options {
  std::string pattern;
  std::size_t bytes{};
  int warmups{2};
  int trials{5};
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
  if (options.pattern != "0-1" && options.pattern != "1-0" &&
      options.pattern != "0-1+0-1" && options.pattern != "0-1+1-0") {
    throw std::invalid_argument(
        "--pattern must be 0-1, 1-0, 0-1+0-1, or 0-1+1-0");
  }
  if (options.bytes == 0 || options.bytes > kMaximumBytes) {
    throw std::invalid_argument("--size-bytes must be in [1, 12 GiB]");
  }
  if (options.warmups < 1 || options.trials < 3) {
    throw std::invalid_argument("at least one warmup and three trials required");
  }
  return options;
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

struct Placement {
  int requested_node{-1};
  std::size_t sampled_pages{};
  std::size_t pages_on_requested_node{};
  std::size_t unknown_pages{};
};

Placement query_placement(void* pointer, std::size_t bytes, int node) {
  constexpr std::size_t samples = 128;
  const long page_size = sysconf(_SC_PAGESIZE);
  std::vector<void*> pages;
  pages.reserve(samples);
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

class NumaBuffer {
 public:
  NumaBuffer(std::size_t bytes, int node, int touch_cpu, unsigned char value)
      : bytes_(bytes), node_(node) {
    pointer_ = numa_alloc_onnode(bytes_, node_);
    if (pointer_ == nullptr) {
      throw std::runtime_error("numa_alloc_onnode failed for " +
                               std::to_string(bytes_) + " bytes on node " +
                               std::to_string(node_));
    }
    try {
      std::thread toucher([&] {
        pin_to_cpu(touch_cpu);
        std::memset(pointer_, value, bytes_);
      });
      toucher.join();
      placement_ = query_placement(pointer_, bytes_, node_);
    } catch (...) {
      numa_free(pointer_, bytes_);
      pointer_ = nullptr;
      throw;
    }
  }

  NumaBuffer(const NumaBuffer&) = delete;
  NumaBuffer& operator=(const NumaBuffer&) = delete;
  ~NumaBuffer() {
    if (pointer_ != nullptr) numa_free(pointer_, bytes_);
  }

  void* data() const noexcept { return pointer_; }
  const Placement& placement() const noexcept { return placement_; }

 private:
  std::size_t bytes_{};
  int node_{};
  void* pointer_{};
  Placement placement_{};
};

struct FlowSpec {
  std::string name;
  int source_node;
  int destination_node;
  int source_cpu;
  int worker_cpu;
  unsigned char value;
};

std::vector<FlowSpec> flows_for(const std::string& pattern) {
  const FlowSpec zero_one{"NUMA0->NUMA1", 0, 1, 0, 20, 0x5a};
  const FlowSpec one_zero{"NUMA1->NUMA0", 1, 0, 20, 0, 0xa5};
  const FlowSpec zero_one_second{"NUMA0->NUMA1#2", 0, 1, 1, 21, 0xc3};
  if (pattern == "0-1") return {zero_one};
  if (pattern == "1-0") return {one_zero};
  if (pattern == "0-1+0-1") return {zero_one, zero_one_second};
  return {zero_one, one_zero};
}

class Flow {
 public:
  Flow(FlowSpec spec, std::size_t bytes)
      : spec_(std::move(spec)),
        bytes_(bytes),
        source_(bytes_, spec_.source_node, spec_.source_cpu, spec_.value),
        destination_(bytes_, spec_.destination_node, spec_.worker_cpu, 0) {}

  void copy() {
    pin_to_cpu(spec_.worker_cpu);
    std::memcpy(destination_.data(), source_.data(), bytes_);
  }

  void validate() const {
    const auto* bytes = static_cast<const unsigned char*>(destination_.data());
    for (std::size_t index = 0; index < bytes_; ++index) {
      if (bytes[index] != spec_.value) {
        throw std::runtime_error("exact-byte validation failed at byte " +
                                 std::to_string(index));
      }
    }
  }

  const FlowSpec& spec() const noexcept { return spec_; }
  const Placement& source_placement() const noexcept {
    return source_.placement();
  }
  const Placement& destination_placement() const noexcept {
    return destination_.placement();
  }

 private:
  FlowSpec spec_;
  std::size_t bytes_{};
  NumaBuffer source_;
  NumaBuffer destination_;
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

void execute_group(std::vector<std::unique_ptr<Flow>>& flows,
                   std::vector<double>* elapsed) {
  std::barrier start(static_cast<std::ptrdiff_t>(flows.size() + 1));
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> threads;
  for (std::size_t index = 0; index < flows.size(); ++index) {
    threads.emplace_back([&, index] {
      try {
        start.arrive_and_wait();
        const auto begin = Clock::now();
        flows[index]->copy();
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

void print_placement(const std::string& flow, const char* buffer,
                     const Placement& placement) {
  std::cout << "{\"type\":\"placement\",\"flow\":\"" << flow
            << "\",\"buffer\":\"" << buffer
            << "\",\"requested_node\":" << placement.requested_node
            << ",\"sampled_pages\":" << placement.sampled_pages
            << ",\"pages_on_requested_node\":"
            << placement.pages_on_requested_node
            << ",\"unknown_pages\":" << placement.unknown_pages << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (numa_available() < 0) throw std::runtime_error("libnuma unavailable");
    const Options options = parse_options(argc, argv);
    const auto specs = flows_for(options.pattern);
    std::vector<std::unique_ptr<Flow>> flows;
    for (const auto& spec : specs) {
      flows.push_back(std::make_unique<Flow>(spec, options.bytes));
    }
    std::cout << "{\"type\":\"metadata\",\"schema\":\"numabraid-host-numa-v1\""
              << ",\"pattern\":\"" << options.pattern << "\",\"bytes\":"
              << options.bytes << ",\"binary_size_units\":true"
              << ",\"maximum_bytes\":" << kMaximumBytes
              << ",\"flow_count\":" << flows.size()
              << ",\"warmups\":" << options.warmups
              << ",\"trials\":" << options.trials
              << ",\"allocation_and_initialization_timed\":false"
              << ",\"copy_implementation\":\"glibc_memcpy_single_pinned_thread_per_flow\"}\n";
    for (const auto& flow : flows) {
      print_placement(flow->spec().name, "source", flow->source_placement());
      print_placement(flow->spec().name, "destination",
                      flow->destination_placement());
    }
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
      execute_group(flows, nullptr);
    }
    std::vector<std::vector<double>> flow_seconds(flows.size());
    std::vector<double> group_seconds;
    const double cpu_begin = process_cpu_seconds();
    for (int trial = 0; trial < options.trials; ++trial) {
      std::vector<double> elapsed(flows.size());
      const auto begin = Clock::now();
      execute_group(flows, &elapsed);
      const auto end = Clock::now();
      group_seconds.push_back(std::chrono::duration<double>(end - begin).count());
      for (std::size_t index = 0; index < elapsed.size(); ++index) {
        flow_seconds[index].push_back(elapsed[index]);
      }
    }
    const double cpu_used = process_cpu_seconds() - cpu_begin;
    for (const auto& flow : flows) flow->validate();

    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t index = 0; index < flows.size(); ++index) {
      const auto& values = flow_seconds[index];
      const double med = median(values);
      const auto& spec = flows[index]->spec();
      std::cout << "{\"type\":\"flow_measurement\",\"path\":\"host-numa\""
                << ",\"pattern\":\"" << options.pattern << "\",\"flow\":\""
                << spec.name << "\",\"bytes\":" << options.bytes
                << ",\"source_numa\":" << spec.source_node
                << ",\"destination_numa\":" << spec.destination_node
                << ",\"affinity_cpu\":" << spec.worker_cpu
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
    const double group_median = median(group_seconds);
    const double total_bytes =
        static_cast<double>(options.bytes) * static_cast<double>(flows.size());
    std::cout << "{\"type\":\"group_measurement\",\"path\":\"host-numa\""
              << ",\"pattern\":\"" << options.pattern
              << "\",\"bytes_each\":" << options.bytes
              << ",\"flow_count\":" << flows.size()
              << ",\"median_seconds\":" << group_median
              << ",\"mean_seconds\":" << mean(group_seconds)
              << ",\"min_seconds\":"
              << *std::min_element(group_seconds.begin(), group_seconds.end())
              << ",\"max_seconds\":"
              << *std::max_element(group_seconds.begin(), group_seconds.end())
              << ",\"aggregate_payload_gbps\":"
              << total_bytes / group_median / 1.0e9
              << ",\"timed_process_cpu_seconds\":" << cpu_used
              << ",\"cpu_equivalent_cores\":"
              << cpu_used /
                     std::accumulate(group_seconds.begin(), group_seconds.end(),
                                     0.0)
              << ",\"exact_bytes\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NUMA host benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
