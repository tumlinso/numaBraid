#include "../../src/topology/x_topology.hh"

#include <cassert>
#include <string>
#include <vector>

using numabraid::topology::DeviceObservation;

namespace {

std::vector<DeviceObservation> valid_x() {
  std::vector<DeviceObservation> devices(4);
  for (int i = 0; i < 4; ++i) {
    devices[i].cuda_device = i;
    devices[i].pci_bus_id = "0000:0" + std::to_string(i + 2) + ":00.0";
    devices[i].peer_access.assign(4, true);
    devices[i].peer_access[i] = false;
    devices[i].nvlink_counts.assign(4, 0);
  }
  devices[0].numa_node = 0;
  devices[1].numa_node = 0;
  devices[2].numa_node = 1;
  devices[3].numa_node = 1;
  devices[0].nvlink_counts[2] = 6;
  devices[2].nvlink_counts[0] = 6;
  devices[1].nvlink_counts[3] = 6;
  devices[3].nvlink_counts[1] = 6;
  return devices;
}

}  // namespace

int main() {
  {
    const auto result = numabraid::topology::validate_x_topology(valid_x());
    assert(result.valid);
    assert(result.roles.a0 == 0);
    assert(result.roles.a1 == 2);
    assert(result.roles.b0 == 1);
    assert(result.roles.b1 == 3);
  }
  {
    auto devices = valid_x();
    devices[2].numa_node = 0;
    const auto result = numabraid::topology::validate_x_topology(devices);
    assert(!result.valid);
  }
  {
    auto devices = valid_x();
    devices[0].nvlink_counts[2] = 0;
    devices[2].nvlink_counts[0] = 0;
    const auto result = numabraid::topology::validate_x_topology(devices);
    assert(!result.valid);
  }
  {
    auto devices = valid_x();
    devices[0].peer_access[1] = false;
    const auto result = numabraid::topology::validate_x_topology(devices);
    assert(!result.valid);
  }
  return 0;
}
