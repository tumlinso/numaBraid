#ifndef NUMABRAID_FORWARDING_X_RELAY_HH
#define NUMABRAID_FORWARDING_X_RELAY_HH

#include "route_validation.hh"

#include <cstddef>
#include <memory>
#include <string>

namespace numabraid::forwarding {

inline constexpr std::size_t default_chunk_bytes = 12U * 1024U * 1024U;
inline constexpr unsigned int default_staging_buffers = 2;

struct PhysicalRoute {
  std::string source_pci;
  std::string relay_pci;
  std::string destination_pci;
};

struct RelayOptions {
  std::size_t capacity_bytes{};
  std::size_t chunk_bytes{};
  unsigned int staging_buffers{1};
};

[[nodiscard]] RouteObservation observe_route(const PhysicalRoute& route);

// A narrow, single-flight relay for one validated diagonal of the V100 X
// topology. enqueue() is asynchronous with respect to the host. A second
// transfer is rejected until ready() is true or synchronize() has completed;
// staging storage is therefore never overwritten while the NVLink leg reads it.
// The capacity-only constructor uses the measured default chunk/ring settings;
// RelayOptions exists for explicit experiments and unusual message regimes.
class XRelay {
 public:
  XRelay(PhysicalRoute route, std::size_t capacity_bytes);
  XRelay(PhysicalRoute route, RelayOptions options);
  ~XRelay();

  XRelay(const XRelay&) = delete;
  XRelay& operator=(const XRelay&) = delete;
  XRelay(XRelay&&) = delete;
  XRelay& operator=(XRelay&&) = delete;

  void enqueue(const void* source, void* destination, std::size_t bytes);
  [[nodiscard]] bool ready();
  void synchronize();

  [[nodiscard]] std::size_t capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t chunk_bytes() const noexcept;
  [[nodiscard]] unsigned int staging_buffers() const noexcept;
  [[nodiscard]] const RouteObservation& route() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace numabraid::forwarding

#endif  // NUMABRAID_FORWARDING_X_RELAY_HH
