#include "x_relay.hh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct RouteCase {
  const char* source;
  const char* relay;
  const char* destination;
  numabraid::forwarding::PhysicalRoute route;
};

int copies_for(std::size_t bytes) {
  if (bytes <= 65536) return 1000;
  if (bytes <= 1048576) return 256;
  if (bytes <= 16777216) return 16;
  return 3;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double stddev(const std::vector<double>& values, double mean) {
  double sum = 0.0;
  for (const double value : values) {
    const double difference = value - mean;
    sum += difference * difference;
  }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

std::vector<std::byte> make_pattern(std::size_t bytes, unsigned int salt) {
  std::vector<std::byte> result(bytes);
  for (std::size_t index = 0; index < bytes; ++index) {
    result[index] = static_cast<std::byte>(
        (index * 131U + (index >> 7U) * 17U + salt) & 0xffU);
  }
  return result;
}

void measure(const RouteCase& route, std::size_t bytes, int trials,
             std::size_t chunk_bytes, unsigned int buffers) {
  const std::size_t capacity = 268435456;
  numabraid::forwarding::XRelay relay(
      route.route, {capacity, chunk_bytes, buffers});
  const auto expected = make_pattern(bytes, static_cast<unsigned int>(bytes));
  std::vector<std::byte> actual(bytes);
  void* source = nullptr;
  void* destination = nullptr;
  const auto& observed = relay.route();

  check(cudaSetDevice(observed.source_device), "cudaSetDevice(source)");
  check(cudaMalloc(&source, bytes), "cudaMalloc(source)");
  check(cudaMemcpy(source, expected.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(source initialize)");
  check(cudaSetDevice(observed.destination_device),
        "cudaSetDevice(destination)");
  check(cudaMalloc(&destination, bytes), "cudaMalloc(destination)");

  for (int warmup = 0; warmup < 5; ++warmup) {
    relay.enqueue(source, destination, bytes);
    relay.synchronize();
  }

  const int copies = copies_for(bytes);
  std::vector<double> times_us;
  times_us.reserve(static_cast<std::size_t>(trials));
  for (int trial = 0; trial < trials; ++trial) {
    const auto start = Clock::now();
    for (int copy = 0; copy < copies; ++copy) {
      relay.enqueue(source, destination, bytes);
      relay.synchronize();
    }
    const auto stop = Clock::now();
    const double elapsed =
        std::chrono::duration<double, std::micro>(stop - start).count() /
        static_cast<double>(copies);
    times_us.push_back(elapsed);
  }

  check(cudaMemcpy(actual.data(), destination, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(destination validate)");
  if (actual != expected) throw std::runtime_error("exact-byte validation failed");

  const double mean =
      std::accumulate(times_us.begin(), times_us.end(), 0.0) /
      static_cast<double>(times_us.size());
  const double med = median(times_us);
  const double payload_gbps = static_cast<double>(bytes) / (med * 1000.0);
  std::cout << std::fixed << std::setprecision(6)
            << "{\"type\":\"measurement\",\"source_role\":\"" << route.source
            << "\",\"relay_role\":\"" << route.relay
            << "\",\"destination_role\":\"" << route.destination
            << "\",\"bytes\":" << bytes << ",\"chunk_bytes\":" << chunk_bytes
            << ",\"buffers\":" << buffers << ",\"copies_per_trial\":" << copies
            << ",\"trials\":" << trials << ",\"median_us\":" << med
            << ",\"mean_us\":" << mean << ",\"min_us\":"
            << *std::min_element(times_us.begin(), times_us.end())
            << ",\"max_us\":" << *std::max_element(times_us.begin(), times_us.end())
            << ",\"stddev_us\":" << stddev(times_us, mean)
            << ",\"payload_gbps\":" << payload_gbps
            << ",\"exact_bytes\":true}\n";

  check(cudaSetDevice(observed.source_device), "cudaSetDevice(free source)");
  check(cudaFree(source), "cudaFree(source)");
  check(cudaSetDevice(observed.destination_device),
        "cudaSetDevice(free destination)");
  check(cudaFree(destination), "cudaFree(destination)");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int trials = 7;
    std::size_t chunk_bytes = 268435456;
    unsigned int buffers = 1;
    std::size_t only_size = 0;
    for (int index = 1; index < argc; index += 2) {
      if (index + 1 >= argc) {
        throw std::invalid_argument("benchmark options require values");
      }
      const std::string option = argv[index];
      if (option == "--trials") {
        trials = std::stoi(argv[index + 1]);
      } else if (option == "--chunk-bytes") {
        chunk_bytes = std::stoull(argv[index + 1]);
      } else if (option == "--buffers") {
        buffers = static_cast<unsigned int>(std::stoul(argv[index + 1]));
      } else if (option == "--size-bytes") {
        only_size = std::stoull(argv[index + 1]);
      } else {
        throw std::invalid_argument("unknown benchmark option: " + option);
      }
    }
    const std::array routes{
        RouteCase{"A0", "B0", "B1", {"0000:02:00.0", "0000:03:00.0", "0000:84:00.0"}},
        RouteCase{"B1", "A1", "A0", {"0000:84:00.0", "0000:83:00.0", "0000:02:00.0"}},
        RouteCase{"B0", "A0", "A1", {"0000:03:00.0", "0000:02:00.0", "0000:83:00.0"}},
        RouteCase{"A1", "B1", "B0", {"0000:83:00.0", "0000:84:00.0", "0000:03:00.0"}},
    };
    const std::array<std::size_t, 5> all_sizes{4096, 65536, 1048576, 16777216,
                                               268435456};
    const std::vector<std::size_t> sizes =
        only_size == 0
            ? std::vector<std::size_t>(all_sizes.begin(), all_sizes.end())
            : std::vector<std::size_t>{only_size};
    std::cout << "{\"type\":\"metadata\",\"schema\":\"numabraid-relay-v1\","
                 "\"mode\":\"chunked_ring\",\"chunk_bytes\":"
              << chunk_bytes << ",\"buffers\":" << buffers
              << ",\"bandwidth_units\":\"GB/s\","
                 "\"bandwidth_definition\":\"one_way_payload_bytes_per_second\"}\n";
    for (const auto& route : routes) {
      for (const auto bytes : sizes) {
        measure(route, bytes, trials, chunk_bytes, buffers);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "relay benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
