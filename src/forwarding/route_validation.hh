#ifndef NUMABRAID_FORWARDING_ROUTE_VALIDATION_HH
#define NUMABRAID_FORWARDING_ROUTE_VALIDATION_HH

#include <string>
#include <vector>

namespace numabraid::forwarding {

struct RouteObservation {
  int source_device{-1};
  int relay_device{-1};
  int destination_device{-1};
  std::string source_pci;
  std::string relay_pci;
  std::string destination_pci;
  std::string source_name;
  std::string relay_name;
  std::string destination_name;
  int source_numa{-1};
  int relay_numa{-1};
  int destination_numa{-1};
  bool relay_can_access_source{false};
  bool destination_can_access_relay{false};
  bool source_can_access_destination{false};
  unsigned int relay_destination_nvlinks{0};
};

[[nodiscard]] std::vector<std::string> validate_route(
    const RouteObservation& route);

}  // namespace numabraid::forwarding

#endif  // NUMABRAID_FORWARDING_ROUTE_VALIDATION_HH
