#include "x_topology.hh"

#include <algorithm>
#include <map>
#include <sstream>

namespace numabraid::topology {
namespace {

void add_error(ValidationResult& result, std::string message) {
  result.errors.push_back(std::move(message));
}

int unique_nvlink_peer(const std::vector<DeviceObservation>& devices,
                       int device, ValidationResult& result) {
  int peer = -1;
  for (std::size_t candidate = 0; candidate < devices.size(); ++candidate) {
    if (candidate == static_cast<std::size_t>(device) ||
        devices[device].nvlink_counts[candidate] == 0) {
      continue;
    }
    if (peer != -1) {
      add_error(result, "GPU " + devices[device].pci_bus_id +
                            " has more than one NVLink peer");
      return -1;
    }
    peer = static_cast<int>(candidate);
  }
  if (peer == -1) {
    add_error(result, "GPU " + devices[device].pci_bus_id +
                          " has no NVLink peer");
  }
  return peer;
}

bool require_peer_access(const std::vector<DeviceObservation>& devices,
                         int source, int destination,
                         ValidationResult& result) {
  if (!devices[source].peer_access[destination]) {
    add_error(result, "CUDA P2P is unavailable from " +
                          devices[source].pci_bus_id + " to " +
                          devices[destination].pci_bus_id);
    return false;
  }
  return true;
}

}  // namespace

ValidationResult validate_x_topology(
    const std::vector<DeviceObservation>& devices) {
  ValidationResult result;
  if (devices.size() != 4) {
    add_error(result, "expected exactly four CUDA-visible GPUs, observed " +
                          std::to_string(devices.size()));
    return result;
  }

  for (const auto& device : devices) {
    if (device.peer_access.size() != devices.size() ||
        device.nvlink_counts.size() != devices.size()) {
      add_error(result, "incomplete relationship matrix for GPU " +
                            device.pci_bus_id);
      return result;
    }
    if (device.numa_node < 0) {
      add_error(result, "GPU " + device.pci_bus_id +
                            " has no authoritative NUMA node");
    }
  }
  if (!result.errors.empty()) {
    return result;
  }

  std::map<int, std::vector<int>> by_numa;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    by_numa[devices[i].numa_node].push_back(static_cast<int>(i));
  }
  if (by_numa.size() != 2) {
    add_error(result, "expected exactly two GPU NUMA nodes");
    return result;
  }
  for (const auto& [node, members] : by_numa) {
    if (members.size() != 2) {
      add_error(result, "NUMA node " + std::to_string(node) +
                            " does not contain exactly two GPUs");
    }
  }
  if (!result.errors.empty()) {
    return result;
  }

  auto node = by_numa.begin();
  const int node0 = node->first;
  const auto side0 = node->second;
  ++node;
  const int node1 = node->first;
  const auto side1 = node->second;
  (void)node1;

  auto first_side = side0;
  std::sort(first_side.begin(), first_side.end(), [&](int left, int right) {
    return devices[left].pci_bus_id < devices[right].pci_bus_id;
  });

  result.roles.a0 = first_side[0];
  result.roles.b0 = first_side[1];
  result.roles.a1 = unique_nvlink_peer(devices, result.roles.a0, result);
  result.roles.b1 = unique_nvlink_peer(devices, result.roles.b0, result);

  for (std::size_t i = 0; i < devices.size(); ++i) {
    (void)unique_nvlink_peer(devices, static_cast<int>(i), result);
  }
  if (!result.errors.empty()) {
    return result;
  }

  if (result.roles.a1 == result.roles.b1 ||
      std::find(side1.begin(), side1.end(), result.roles.a1) == side1.end() ||
      std::find(side1.begin(), side1.end(), result.roles.b1) == side1.end()) {
    add_error(result, "NVLink pairs do not form a carrier-spanning matching");
    return result;
  }
  if (devices[result.roles.a1].nvlink_counts[result.roles.a0] == 0 ||
      devices[result.roles.b1].nvlink_counts[result.roles.b0] == 0) {
    add_error(result, "NVLink carrier relationship is not symmetric");
  }

  const std::array<std::pair<int, int>, 8> required_legs{{
      {result.roles.a0, result.roles.a1},
      {result.roles.a1, result.roles.a0},
      {result.roles.b0, result.roles.b1},
      {result.roles.b1, result.roles.b0},
      {result.roles.a0, result.roles.b0},
      {result.roles.b0, result.roles.a0},
      {result.roles.a1, result.roles.b1},
      {result.roles.b1, result.roles.a1},
  }};
  for (const auto& [source, destination] : required_legs) {
    (void)require_peer_access(devices, source, destination, result);
  }

  if (devices[result.roles.a0].numa_node != node0 ||
      devices[result.roles.b0].numa_node != node0 ||
      devices[result.roles.a1].numa_node == node0 ||
      devices[result.roles.b1].numa_node == node0) {
    add_error(result, "role assignment does not span the two NUMA nodes");
  }

  result.valid = result.errors.empty();
  return result;
}

}  // namespace numabraid::topology
