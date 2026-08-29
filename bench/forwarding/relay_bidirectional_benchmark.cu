#include "x_relay.hh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct Direction {
  const char* name;
  numabraid::forwarding::PhysicalRoute route;
};

struct Allocation {
  void* source{};
  void* destination{};
  int source_device{-1};
  int destination_device{-1};
};

Allocation allocate(const numabraid::forwarding::XRelay& relay,
                    std::size_t bytes, unsigned char value) {
  const auto& route = relay.route();
  Allocation result{.source_device = route.source_device,
                    .destination_device = route.destination_device};
  check(cudaSetDevice(result.source_device), "cudaSetDevice(source)");
  check(cudaMalloc(&result.source, bytes), "cudaMalloc(source)");
  check(cudaMemset(result.source, value, bytes), "cudaMemset(source)");
  check(cudaSetDevice(result.destination_device), "cudaSetDevice(destination)");
  check(cudaMalloc(&result.destination, bytes), "cudaMalloc(destination)");
  check(cudaMemset(result.destination, 0, bytes), "cudaMemset(destination)");
  return result;
}

void release(const Allocation& allocation) {
  check(cudaSetDevice(allocation.source_device), "cudaSetDevice(free source)");
  check(cudaFree(allocation.source), "cudaFree(source)");
  check(cudaSetDevice(allocation.destination_device),
        "cudaSetDevice(free destination)");
  check(cudaFree(allocation.destination), "cudaFree(destination)");
}

void validate(const Allocation& allocation, std::size_t bytes,
              unsigned char expected) {
  std::vector<unsigned char> data(bytes);
  check(cudaSetDevice(allocation.destination_device),
        "cudaSetDevice(validate destination)");
  check(cudaMemcpy(data.data(), allocation.destination, bytes,
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(validate destination)");
  for (const auto value : data) {
    if (value != expected) throw std::runtime_error("exact-byte validation failed");
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run_pair(const Direction& first, const Direction& second) {
  constexpr std::size_t bytes = 268435456;
  constexpr std::size_t chunk = 12582912;
  constexpr unsigned int buffers = 2;
  constexpr int copies = 3;
  constexpr int trials = 7;
  numabraid::forwarding::XRelay first_relay(first.route,
                                             {bytes, chunk, buffers});
  numabraid::forwarding::XRelay second_relay(second.route,
                                              {bytes, chunk, buffers});
  const auto first_memory = allocate(first_relay, bytes, 0x3c);
  const auto second_memory = allocate(second_relay, bytes, 0xc3);

  auto execute = [&] {
    std::barrier start(3);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto worker = [&](numabraid::forwarding::XRelay& relay,
                      const Allocation& memory) {
      try {
        start.arrive_and_wait();
        for (int copy = 0; copy < copies; ++copy) {
          relay.enqueue(memory.source, memory.destination, bytes);
          relay.synchronize();
        }
      } catch (...) {
        std::scoped_lock lock(failure_mutex);
        failure = std::current_exception();
      }
    };
    std::thread one(worker, std::ref(first_relay), std::cref(first_memory));
    std::thread two(worker, std::ref(second_relay), std::cref(second_memory));
    start.arrive_and_wait();
    const auto begin = Clock::now();
    one.join();
    two.join();
    const auto end = Clock::now();
    if (failure) std::rethrow_exception(failure);
    return std::chrono::duration<double, std::micro>(end - begin).count() /
           static_cast<double>(copies);
  };

  for (int warmup = 0; warmup < 3; ++warmup) static_cast<void>(execute());
  std::vector<double> times;
  for (int trial = 0; trial < trials; ++trial) times.push_back(execute());
  validate(first_memory, bytes, 0x3c);
  validate(second_memory, bytes, 0xc3);

  const double med = median(times);
  const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
                      static_cast<double>(times.size());
  std::cout << std::fixed << std::setprecision(6)
            << "{\"type\":\"measurement\",\"first\":\"" << first.name
            << "\",\"second\":\"" << second.name << "\",\"bytes_each\":"
            << bytes << ",\"chunk_bytes\":" << chunk << ",\"buffers\":"
            << buffers << ",\"copies_per_trial\":" << copies
            << ",\"trials\":" << trials << ",\"median_us\":" << med
            << ",\"mean_us\":" << mean
            << ",\"aggregate_payload_gbps\":"
            << (2.0 * static_cast<double>(bytes) / (med * 1000.0))
            << ",\"per_direction_effective_gbps\":"
            << (static_cast<double>(bytes) / (med * 1000.0))
            << ",\"exact_bytes\":true}\n";
  release(first_memory);
  release(second_memory);
}

}  // namespace

int main() {
  try {
    std::cout << "{\"type\":\"metadata\",\"schema\":\"numabraid-relay-bidirectional-v1\","
                 "\"bandwidth_definition\":\"sum_of_two_one_way_payloads\"}\n";
    run_pair({"A0->B1", {"0000:02:00.0", "0000:03:00.0", "0000:84:00.0"}},
             {"B1->A0", {"0000:84:00.0", "0000:83:00.0", "0000:02:00.0"}});
    run_pair({"B0->A1", {"0000:03:00.0", "0000:02:00.0", "0000:83:00.0"}},
             {"A1->B0", {"0000:83:00.0", "0000:84:00.0", "0000:03:00.0"}});
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bidirectional relay benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
