#ifndef NUMABRAID_SRC_TOPOLOGY_X_TOPOLOGY_HH
#define NUMABRAID_SRC_TOPOLOGY_X_TOPOLOGY_HH

#include <array>
#include <string>
#include <vector>

namespace numabraid::topology {

struct DeviceObservation {
  int cuda_device = -1;
  std::string pci_bus_id;
  int numa_node = -1;
  std::vector<bool> peer_access;
  std::vector<unsigned int> nvlink_counts;
};

struct XRoles {
  int a0 = -1;
  int a1 = -1;
  int b0 = -1;
  int b1 = -1;
};

struct ValidationResult {
  bool valid = false;
  XRoles roles;
  std::vector<std::string> errors;
};

// Validate only the four-GPU X arrangement numaBraid supports. Carrier A is
// chosen deterministically from the lexically first NUMA-0 PCI identity.
ValidationResult validate_x_topology(
    const std::vector<DeviceObservation>& devices);

}  // namespace numabraid::topology

#endif  // NUMABRAID_SRC_TOPOLOGY_X_TOPOLOGY_HH
