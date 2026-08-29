#include "route_validation.hh"

#include <array>
#include <string_view>

namespace numabraid::forwarding {
namespace {

bool is_v100(std::string_view name) { return name.find("V100") != name.npos; }

}  // namespace

std::vector<std::string> validate_route(const RouteObservation& route) {
  std::vector<std::string> errors;
  const std::array devices{route.source_device, route.relay_device,
                           route.destination_device};
  if (devices[0] < 0 || devices[1] < 0 || devices[2] < 0) {
    errors.emplace_back("all route roles must resolve to CUDA devices");
  }
  if (devices[0] == devices[1] || devices[0] == devices[2] ||
      devices[1] == devices[2]) {
    errors.emplace_back("source, relay, and destination must be distinct GPUs");
  }
  if (route.source_pci.empty() || route.relay_pci.empty() ||
      route.destination_pci.empty()) {
    errors.emplace_back("all route roles require stable PCI identities");
  }
  if (!is_v100(route.source_name) || !is_v100(route.relay_name) ||
      !is_v100(route.destination_name)) {
    errors.emplace_back("numaBraid relay requires V100 GPUs");
  }
  if (route.source_numa < 0 || route.relay_numa < 0 ||
      route.destination_numa < 0) {
    errors.emplace_back("all route roles require known NUMA nodes");
  } else {
    if (route.source_numa != route.relay_numa) {
      errors.emplace_back("source and relay must share a NUMA node");
    }
    if (route.relay_numa == route.destination_numa) {
      errors.emplace_back("relay and destination must span NUMA nodes");
    }
  }
  if (!route.relay_can_access_source) {
    errors.emplace_back("relay cannot peer-read the source GPU");
  }
  if (!route.destination_can_access_relay) {
    errors.emplace_back("destination cannot peer-read the relay GPU");
  }
  if (route.source_can_access_destination) {
    errors.emplace_back(
        "source can directly access destination; a diagonal relay is not required");
  }
  if (route.relay_destination_nvlinks == 0) {
    errors.emplace_back("relay and destination do not have an active NVLink");
  }
  return errors;
}

}  // namespace numabraid::forwarding
