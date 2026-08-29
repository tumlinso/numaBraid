#include "x_topology.hh"

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using numabraid::topology::DeviceObservation;
using numabraid::topology::ValidationResult;

namespace {

struct LiveDevice {
  DeviceObservation topology;
  std::string name;
  std::string uuid;
  std::string pcie_speed;
  std::string pcie_width;
};

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void nvml_check(nvmlReturn_t status, const char* operation) {
  if (status != NVML_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             nvmlErrorString(status));
  }
}

std::string trim(std::string value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  return value;
}

std::string read_file(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    return "unavailable";
  }
  std::string value;
  std::getline(input, value);
  return trim(value);
}

std::string canonical_pci_id(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  if (std::sscanf(value.c_str(), "%x:%x:%x.%x", &domain, &bus, &device,
                  &function) == 4) {
    char canonical[32]{};
    std::snprintf(canonical, sizeof(canonical), "%04x:%02x:%02x.%x", domain,
                  bus, device, function);
    return canonical;
  }
  return value;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  out << '"';
  return out.str();
}

std::vector<LiveDevice> observe_devices() {
  int count = 0;
  cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  std::vector<LiveDevice> devices(static_cast<std::size_t>(count));

  for (int device = 0; device < count; ++device) {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    char pci_id[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), device),
               "cudaDeviceGetPCIBusId");

    auto& observed = devices[static_cast<std::size_t>(device)];
    observed.topology.cuda_device = device;
    observed.topology.pci_bus_id = canonical_pci_id(pci_id);
    observed.name = properties.name;
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (const unsigned char byte : properties.uuid.bytes) {
      uuid << std::setw(2) << static_cast<unsigned int>(byte);
    }
    observed.uuid = uuid.str();

    const fs::path sysfs = fs::path("/sys/bus/pci/devices") /
                           observed.topology.pci_bus_id;
    const std::string numa = read_file(sysfs / "numa_node");
    if (numa != "unavailable") {
      observed.topology.numa_node = std::stoi(numa);
    }
    observed.pcie_speed = read_file(sysfs / "current_link_speed");
    observed.pcie_width = read_file(sysfs / "current_link_width");
    observed.topology.peer_access.resize(static_cast<std::size_t>(count));
    observed.topology.nvlink_counts.resize(static_cast<std::size_t>(count));
  }

  for (int source = 0; source < count; ++source) {
    for (int destination = 0; destination < count; ++destination) {
      if (source == destination) {
        continue;
      }
      int accessible = 0;
      cuda_check(cudaDeviceCanAccessPeer(&accessible, source, destination),
                 "cudaDeviceCanAccessPeer");
      devices[static_cast<std::size_t>(source)]
          .topology.peer_access[static_cast<std::size_t>(destination)] =
          accessible != 0;
    }
  }

  nvml_check(nvmlInit_v2(), "nvmlInit_v2");
  try {
    for (std::size_t i = 0; i < devices.size(); ++i) {
      nvmlDevice_t handle{};
      nvml_check(nvmlDeviceGetHandleByPciBusId_v2(
                     devices[i].topology.pci_bus_id.c_str(), &handle),
                 "nvmlDeviceGetHandleByPciBusId_v2");
      for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
        nvmlEnableState_t active = NVML_FEATURE_DISABLED;
        const nvmlReturn_t state = nvmlDeviceGetNvLinkState(handle, link, &active);
        if (state == NVML_ERROR_NOT_SUPPORTED || state == NVML_ERROR_INVALID_ARGUMENT) {
          continue;
        }
        nvml_check(state, "nvmlDeviceGetNvLinkState");
        if (active != NVML_FEATURE_ENABLED) {
          continue;
        }
        nvmlPciInfo_t remote{};
        nvml_check(nvmlDeviceGetNvLinkRemotePciInfo_v2(handle, link, &remote),
                   "nvmlDeviceGetNvLinkRemotePciInfo_v2");
        const std::string remote_id = canonical_pci_id(remote.busId);
        const auto found = std::find_if(
            devices.begin(), devices.end(), [&](const LiveDevice& candidate) {
              return candidate.topology.pci_bus_id == remote_id;
            });
        if (found != devices.end()) {
          const std::size_t peer = static_cast<std::size_t>(found - devices.begin());
          ++devices[i].topology.nvlink_counts[peer];
        }
      }
    }
  } catch (...) {
    (void)nvmlShutdown();
    throw;
  }
  nvml_check(nvmlShutdown(), "nvmlShutdown");
  return devices;
}

void print_json(const std::vector<LiveDevice>& devices,
                const ValidationResult& validation) {
  int runtime_version = 0;
  int driver_version = 0;
  cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  std::cout << "{\n  \"schema\": \"numabraid-x-topology-v1\",\n"
            << "  \"valid\": " << (validation.valid ? "true" : "false") << ",\n"
            << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
            << "  \"cuda_driver_api_version\": " << driver_version << ",\n"
            << "  \"devices\": [\n";
  for (std::size_t i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    std::cout << "    {\"cuda_device\": " << device.topology.cuda_device
              << ", \"pci_bus_id\": " << json_string(device.topology.pci_bus_id)
              << ", \"name\": " << json_string(device.name)
              << ", \"uuid_hex\": " << json_string(device.uuid)
              << ", \"numa_node\": " << device.topology.numa_node
              << ", \"pcie_speed\": " << json_string(device.pcie_speed)
              << ", \"pcie_width\": " << json_string(device.pcie_width)
              << ", \"peer_access\": [";
    for (std::size_t peer = 0; peer < devices.size(); ++peer) {
      if (peer != 0) std::cout << ", ";
      std::cout << (device.topology.peer_access[peer] ? "true" : "false");
    }
    std::cout << "], \"nvlink_counts\": [";
    for (std::size_t peer = 0; peer < devices.size(); ++peer) {
      if (peer != 0) std::cout << ", ";
      std::cout << device.topology.nvlink_counts[peer];
    }
    std::cout << "]}" << (i + 1 == devices.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"roles\": {\"A0\": " << validation.roles.a0
            << ", \"A1\": " << validation.roles.a1
            << ", \"B0\": " << validation.roles.b0
            << ", \"B1\": " << validation.roles.b1 << "},\n"
            << "  \"errors\": [";
  for (std::size_t i = 0; i < validation.errors.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << json_string(validation.errors[i]);
  }
  std::cout << "]\n}\n";
}

void print_human(const std::vector<LiveDevice>& devices,
                 const ValidationResult& validation) {
  std::cout << "numaBraid X-topology: "
            << (validation.valid ? "VALID" : "INVALID") << '\n';
  for (const auto& device : devices) {
    std::cout << "CUDA " << device.topology.cuda_device << "  "
              << device.topology.pci_bus_id << "  NUMA "
              << device.topology.numa_node << "  " << device.name
              << "  PCIe " << device.pcie_speed << " x"
              << device.pcie_width << '\n';
  }
  if (validation.valid) {
    std::cout << "roles: A0=CUDA " << validation.roles.a0
              << " A1=CUDA " << validation.roles.a1
              << " B0=CUDA " << validation.roles.b0
              << " B1=CUDA " << validation.roles.b1 << '\n';
  }
  for (const auto& error : validation.errors) {
    std::cerr << "error: " << error << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool json = argc == 2 && std::string(argv[1]) == "--json";
  if (argc > 2 || (argc == 2 && !json)) {
    std::cerr << "usage: numabraid_topology_probe [--json]\n";
    return 2;
  }
  try {
    const auto devices = observe_devices();
    std::vector<DeviceObservation> topology;
    topology.reserve(devices.size());
    for (const auto& device : devices) topology.push_back(device.topology);
    const auto validation = numabraid::topology::validate_x_topology(topology);
    if (json) {
      print_json(devices, validation);
    } else {
      print_human(devices, validation);
    }
    return validation.valid ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "numabraid topology probe failed: " << error.what() << '\n';
    return 2;
  }
}
