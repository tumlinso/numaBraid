#include "x_relay.hh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::vector<std::byte> pattern(std::size_t bytes, unsigned int salt) {
  std::vector<std::byte> data(bytes);
  for (std::size_t index = 0; index < bytes; ++index) {
    data[index] = static_cast<std::byte>(
        (index * 131U + (index >> 7U) * 17U + salt) & 0xffU);
  }
  return data;
}

struct RouteCase {
  const char* name;
  numabraid::forwarding::PhysicalRoute route;
};

void run_case(const RouteCase& test, std::size_t bytes, unsigned int salt) {
  const std::size_t capacity = bytes == 0 ? 1 : bytes;
  numabraid::forwarding::XRelay relay(
      test.route,
      {capacity, std::min<std::size_t>(capacity, 1048576), 3});
  const auto expected = pattern(bytes, salt);
  std::vector<std::byte> actual(bytes);
  void* source = nullptr;
  void* destination = nullptr;
  const auto& observed = relay.route();

  check(cudaSetDevice(observed.source_device), "cudaSetDevice(source)");
  check(cudaMalloc(&source, bytes == 0 ? 1 : bytes), "cudaMalloc(source)");
  if (bytes != 0) {
    check(cudaMemcpy(source, expected.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy(source initialize)");
  }
  check(cudaSetDevice(observed.destination_device),
        "cudaSetDevice(destination)");
  check(cudaMalloc(&destination, bytes == 0 ? 1 : bytes),
        "cudaMalloc(destination)");
  check(cudaMemset(destination, 0xa5, bytes == 0 ? 1 : bytes),
        "cudaMemset(destination)");

  relay.enqueue(source, destination, bytes);
  relay.synchronize();
  if (!relay.ready()) throw std::runtime_error("completed relay is not ready");

  if (bytes != 0) {
    check(cudaMemcpy(actual.data(), destination, bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(destination validate)");
    if (actual != expected) {
      throw std::runtime_error(std::string(test.name) +
                               " exact-byte validation failed");
    }
  }

  check(cudaSetDevice(observed.source_device), "cudaSetDevice(free source)");
  check(cudaFree(source), "cudaFree(source)");
  check(cudaSetDevice(observed.destination_device),
        "cudaSetDevice(free destination)");
  check(cudaFree(destination), "cudaFree(destination)");
  std::cout << test.name << " bytes=" << bytes << " exact=true\n";
}

}  // namespace

int main() {
  try {
    const std::array routes{
        RouteCase{"A0-B0-B1", {"0000:02:00.0", "0000:03:00.0", "0000:84:00.0"}},
        RouteCase{"B1-A1-A0", {"0000:84:00.0", "0000:83:00.0", "0000:02:00.0"}},
        RouteCase{"B0-A0-A1", {"0000:03:00.0", "0000:02:00.0", "0000:83:00.0"}},
        RouteCase{"A1-B1-B0", {"0000:83:00.0", "0000:84:00.0", "0000:03:00.0"}},
    };
    const std::array<std::size_t, 10> sizes{0, 1, 31, 4095, 4096, 4097,
                                            65521, 1048573, 16777219, 67108871};
    unsigned int salt = 1;
    for (const auto& route : routes) {
      for (const auto bytes : sizes) run_case(route, bytes, salt++);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "relay hardware test failed: " << error.what() << '\n';
    return 1;
  }
}
