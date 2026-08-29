#include "route_validation.hh"

#include <algorithm>
#include <cassert>
#include <string>

namespace {

numabraid::forwarding::RouteObservation valid_route() {
  return {
      .source_device = 0,
      .relay_device = 1,
      .destination_device = 3,
      .source_pci = "0000:02:00.0",
      .relay_pci = "0000:03:00.0",
      .destination_pci = "0000:84:00.0",
      .source_name = "Tesla V100-SXM2-16GB",
      .relay_name = "Tesla V100-SXM2-16GB",
      .destination_name = "Tesla V100-SXM2-16GB",
      .source_numa = 0,
      .relay_numa = 0,
      .destination_numa = 1,
      .relay_can_access_source = true,
      .destination_can_access_relay = true,
      .source_can_access_destination = false,
      .relay_destination_nvlinks = 6,
  };
}

bool contains(const std::vector<std::string>& errors, const std::string& text) {
  return std::any_of(errors.begin(), errors.end(), [&](const auto& error) {
    return error.find(text) != error.npos;
  });
}

}  // namespace

int main() {
  assert(numabraid::forwarding::validate_route(valid_route()).empty());

  auto wrong_numa = valid_route();
  wrong_numa.relay_numa = 1;
  assert(contains(numabraid::forwarding::validate_route(wrong_numa),
                  "source and relay"));

  auto no_nvlink = valid_route();
  no_nvlink.relay_destination_nvlinks = 0;
  assert(contains(numabraid::forwarding::validate_route(no_nvlink), "NVLink"));

  auto no_peer = valid_route();
  no_peer.relay_can_access_source = false;
  assert(contains(numabraid::forwarding::validate_route(no_peer), "peer-read"));

  auto direct = valid_route();
  direct.source_can_access_destination = true;
  assert(contains(numabraid::forwarding::validate_route(direct),
                  "directly access"));
}
