#include "x_relay.hh"

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numabraid::forwarding {
namespace {

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

std::string normalize_pci(std::string value) {
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

int device_for_pci(const std::string& requested) {
  int count = 0;
  cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  for (int device = 0; device < count; ++device) {
    char pci[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device),
               "cudaDeviceGetPCIBusId");
    if (normalize_pci(pci) == normalize_pci(requested)) return device;
  }
  throw std::runtime_error("PCI identity " + requested +
                           " is not CUDA-visible");
}

std::string pci_for_device(int device) {
  char pci[32]{};
  cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device),
             "cudaDeviceGetPCIBusId");
  return normalize_pci(pci);
}

int numa_for_pci(const std::string& pci) {
  const auto path = std::filesystem::path("/sys/bus/pci/devices") /
                    normalize_pci(pci) / "numa_node";
  std::ifstream input(path);
  int numa = -1;
  if (!(input >> numa)) {
    throw std::runtime_error("cannot read NUMA node from " + path.string());
  }
  return numa;
}

std::string device_name(int device) {
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  return properties.name;
}

bool can_access(int accessor, int allocation_owner) {
  int access = 0;
  cuda_check(cudaDeviceCanAccessPeer(&access, accessor, allocation_owner),
             "cudaDeviceCanAccessPeer");
  return access != 0;
}

unsigned int nvlink_count(const std::string& first_pci,
                          const std::string& second_pci) {
  nvmlDevice_t device{};
  nvml_check(nvmlDeviceGetHandleByPciBusId(first_pci.c_str(), &device),
             "nvmlDeviceGetHandleByPciBusId");
  unsigned int count = 0;
  for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
    nvmlEnableState_t active = NVML_FEATURE_DISABLED;
    const auto state = nvmlDeviceGetNvLinkState(device, link, &active);
    if (state == NVML_ERROR_NOT_SUPPORTED ||
        state == NVML_ERROR_INVALID_ARGUMENT) {
      continue;
    }
    nvml_check(state, "nvmlDeviceGetNvLinkState");
    if (active != NVML_FEATURE_ENABLED) continue;
    nvmlPciInfo_t remote{};
    nvml_check(nvmlDeviceGetNvLinkRemotePciInfo(device, link, &remote),
               "nvmlDeviceGetNvLinkRemotePciInfo");
    if (normalize_pci(remote.busId) == normalize_pci(second_pci)) ++count;
  }
  return count;
}

void enable_peer(int accessor, int allocation_owner, bool& enabled_here) {
  cuda_check(cudaSetDevice(accessor), "cudaSetDevice(enable peer)");
  const auto status = cudaDeviceEnablePeerAccess(allocation_owner, 0);
  if (status == cudaSuccess) {
    enabled_here = true;
    return;
  }
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    static_cast<void>(cudaGetLastError());
    enabled_here = false;
    return;
  }
  cuda_check(status, "cudaDeviceEnablePeerAccess");
}

void require_device_pointer(const void* pointer, int expected_device,
                            const char* role) {
  cudaPointerAttributes attributes{};
  cuda_check(cudaPointerGetAttributes(&attributes, pointer),
             "cudaPointerGetAttributes");
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != expected_device) {
    std::ostringstream message;
    message << role << " pointer must be device memory owned by CUDA device "
            << expected_device;
    throw std::invalid_argument(message.str());
  }
}

}  // namespace

RouteObservation observe_route(const PhysicalRoute& requested) {
  const int source = device_for_pci(requested.source_pci);
  const int relay = device_for_pci(requested.relay_pci);
  const int destination = device_for_pci(requested.destination_pci);
  const auto source_pci = pci_for_device(source);
  const auto relay_pci = pci_for_device(relay);
  const auto destination_pci = pci_for_device(destination);

  nvml_check(nvmlInit(), "nvmlInit");
  try {
    RouteObservation observation{
        .source_device = source,
        .relay_device = relay,
        .destination_device = destination,
        .source_pci = source_pci,
        .relay_pci = relay_pci,
        .destination_pci = destination_pci,
        .source_name = device_name(source),
        .relay_name = device_name(relay),
        .destination_name = device_name(destination),
        .source_numa = numa_for_pci(source_pci),
        .relay_numa = numa_for_pci(relay_pci),
        .destination_numa = numa_for_pci(destination_pci),
        .relay_can_access_source = can_access(relay, source),
        .destination_can_access_relay = can_access(destination, relay),
        .source_can_access_destination = can_access(source, destination),
        .relay_destination_nvlinks = nvlink_count(relay_pci, destination_pci),
    };
    nvml_check(nvmlShutdown(), "nvmlShutdown");
    return observation;
  } catch (...) {
    static_cast<void>(nvmlShutdown());
    throw;
  }
}

struct XRelay::Impl {
  explicit Impl(PhysicalRoute requested, RelayOptions options)
      : observed(observe_route(requested)), capacity(options.capacity_bytes) {
    const auto errors = validate_route(observed);
    if (!errors.empty()) {
      std::ostringstream message;
      message << "invalid numaBraid relay route";
      for (const auto& error : errors) message << "; " << error;
      throw std::invalid_argument(message.str());
    }
    if (capacity == 0) {
      throw std::invalid_argument("relay capacity must be nonzero");
    }
    if (options.chunk_bytes == 0) {
      throw std::invalid_argument("relay chunk size must be nonzero");
    }
    if (options.staging_buffers == 0 || options.staging_buffers > 16) {
      throw std::invalid_argument("relay staging buffer count must be in [1, 16]");
    }
    chunk = std::min(capacity, options.chunk_bytes);
    buffers = options.staging_buffers;
    if (chunk > std::numeric_limits<std::size_t>::max() / buffers) {
      throw std::invalid_argument("relay staging allocation size overflows");
    }
    staging_bytes = chunk * buffers;

    try {
      enable_peer(observed.relay_device, observed.source_device,
                  ingress_peer_enabled_here);
      enable_peer(observed.destination_device, observed.relay_device,
                  egress_peer_enabled_here);

      cuda_check(cudaSetDevice(observed.relay_device),
                 "cudaSetDevice(relay setup)");
      cuda_check(cudaMalloc(&staging, staging_bytes),
                 "cudaMalloc(relay staging)");
      cuda_check(cudaStreamCreateWithFlags(&ingress_stream,
                                           cudaStreamNonBlocking),
                 "cudaStreamCreate(relay ingress)");
      ingress_done.resize(buffers);
      for (auto& event : ingress_done) {
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreate(relay ingress)");
      }

      cuda_check(cudaSetDevice(observed.destination_device),
                 "cudaSetDevice(destination setup)");
      cuda_check(cudaStreamCreateWithFlags(&egress_stream,
                                           cudaStreamNonBlocking),
                 "cudaStreamCreate(relay egress)");
      egress_done.resize(buffers);
      for (auto& event : egress_done) {
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreate(relay egress)");
      }
      cuda_check(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming),
                 "cudaEventCreate(relay completion)");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  void cleanup() noexcept {
    if (egress_stream != nullptr) {
      static_cast<void>(cudaSetDevice(observed.destination_device));
      static_cast<void>(cudaStreamSynchronize(egress_stream));
    }
    if (ingress_stream != nullptr) {
      static_cast<void>(cudaSetDevice(observed.relay_device));
      static_cast<void>(cudaStreamSynchronize(ingress_stream));
    }
    if (completion != nullptr) {
      static_cast<void>(cudaSetDevice(observed.destination_device));
      static_cast<void>(cudaEventDestroy(completion));
      completion = nullptr;
    }
    if (!egress_done.empty()) {
      static_cast<void>(cudaSetDevice(observed.destination_device));
      for (auto& event : egress_done) {
        if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
      }
      egress_done.clear();
    }
    if (egress_stream != nullptr) {
      static_cast<void>(cudaSetDevice(observed.destination_device));
      static_cast<void>(cudaStreamDestroy(egress_stream));
      egress_stream = nullptr;
    }
    if (!ingress_done.empty()) {
      static_cast<void>(cudaSetDevice(observed.relay_device));
      for (auto& event : ingress_done) {
        if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
      }
      ingress_done.clear();
    }
    if (ingress_stream != nullptr) {
      static_cast<void>(cudaSetDevice(observed.relay_device));
      static_cast<void>(cudaStreamDestroy(ingress_stream));
      ingress_stream = nullptr;
    }
    if (staging != nullptr) {
      static_cast<void>(cudaSetDevice(observed.relay_device));
      static_cast<void>(cudaFree(staging));
      staging = nullptr;
    }
    if (egress_peer_enabled_here) {
      static_cast<void>(cudaSetDevice(observed.destination_device));
      static_cast<void>(cudaDeviceDisablePeerAccess(observed.relay_device));
      egress_peer_enabled_here = false;
    }
    if (ingress_peer_enabled_here) {
      static_cast<void>(cudaSetDevice(observed.relay_device));
      static_cast<void>(cudaDeviceDisablePeerAccess(observed.source_device));
      ingress_peer_enabled_here = false;
    }
  }

  RouteObservation observed;
  std::size_t capacity{};
  std::size_t chunk{};
  std::size_t staging_bytes{};
  unsigned int buffers{};
  void* staging{};
  cudaStream_t ingress_stream{};
  cudaStream_t egress_stream{};
  std::vector<cudaEvent_t> ingress_done;
  std::vector<cudaEvent_t> egress_done;
  cudaEvent_t completion{};
  bool ingress_peer_enabled_here{false};
  bool egress_peer_enabled_here{false};
  bool pending{false};
  std::mutex mutex;
};

XRelay::XRelay(PhysicalRoute route, std::size_t capacity_bytes)
    : XRelay(std::move(route),
             RelayOptions{capacity_bytes,
                          std::min(capacity_bytes, default_chunk_bytes),
                          default_staging_buffers}) {}

XRelay::XRelay(PhysicalRoute route, RelayOptions options)
    : impl_(std::make_unique<Impl>(std::move(route), options)) {}

XRelay::~XRelay() = default;

void XRelay::enqueue(const void* source, void* destination, std::size_t bytes) {
  std::scoped_lock lock(impl_->mutex);
  if (impl_->pending) {
    cuda_check(cudaSetDevice(impl_->observed.destination_device),
               "cudaSetDevice(query completion)");
    const auto state = cudaEventQuery(impl_->completion);
    if (state == cudaErrorNotReady) {
      throw std::logic_error("relay already has a transfer in flight");
    }
    cuda_check(state, "cudaEventQuery(relay completion)");
    impl_->pending = false;
  }
  if (bytes > impl_->capacity) {
    throw std::invalid_argument("transfer exceeds relay staging capacity");
  }
  if (bytes == 0) return;
  if (source == nullptr || destination == nullptr) {
    throw std::invalid_argument("nonzero relay transfer requires non-null pointers");
  }
  require_device_pointer(source, impl_->observed.source_device, "source");
  require_device_pointer(destination, impl_->observed.destination_device,
                         "destination");

  try {
    auto* staging = static_cast<std::byte*>(impl_->staging);
    const auto* source_bytes = static_cast<const std::byte*>(source);
    auto* destination_bytes = static_cast<std::byte*>(destination);
    std::size_t offset = 0;
    std::size_t sequence = 0;
    while (offset < bytes) {
      const auto slot = static_cast<unsigned int>(sequence % impl_->buffers);
      const auto current = std::min(impl_->chunk, bytes - offset);

      cuda_check(cudaSetDevice(impl_->observed.relay_device),
                 "cudaSetDevice(relay enqueue)");
      if (sequence >= impl_->buffers) {
        cuda_check(cudaStreamWaitEvent(impl_->ingress_stream,
                                       impl_->egress_done[slot], 0),
                   "cudaStreamWaitEvent(staging reuse)");
      }
      cuda_check(cudaMemcpyPeerAsync(
                     staging + static_cast<std::size_t>(slot) * impl_->chunk,
                     impl_->observed.relay_device, source_bytes + offset,
                     impl_->observed.source_device, current,
                     impl_->ingress_stream),
                 "cudaMemcpyPeerAsync(source to relay)");
      cuda_check(cudaEventRecord(impl_->ingress_done[slot],
                                 impl_->ingress_stream),
                 "cudaEventRecord(relay ingress)");

      cuda_check(cudaSetDevice(impl_->observed.destination_device),
                 "cudaSetDevice(destination enqueue)");
      cuda_check(cudaStreamWaitEvent(impl_->egress_stream,
                                     impl_->ingress_done[slot], 0),
                 "cudaStreamWaitEvent(relay ingress)");
      cuda_check(cudaMemcpyPeerAsync(
                     destination_bytes + offset,
                     impl_->observed.destination_device,
                     staging + static_cast<std::size_t>(slot) * impl_->chunk,
                     impl_->observed.relay_device, current, impl_->egress_stream),
                 "cudaMemcpyPeerAsync(relay to destination)");
      cuda_check(cudaEventRecord(impl_->egress_done[slot], impl_->egress_stream),
                 "cudaEventRecord(relay egress)");
      offset += current;
      ++sequence;
    }
    cuda_check(cudaEventRecord(impl_->completion, impl_->egress_stream),
               "cudaEventRecord(relay completion)");
    impl_->pending = true;
  } catch (...) {
    static_cast<void>(cudaSetDevice(impl_->observed.relay_device));
    static_cast<void>(cudaStreamSynchronize(impl_->ingress_stream));
    static_cast<void>(cudaSetDevice(impl_->observed.destination_device));
    static_cast<void>(cudaStreamSynchronize(impl_->egress_stream));
    impl_->pending = false;
    throw;
  }
}

bool XRelay::ready() {
  std::scoped_lock lock(impl_->mutex);
  if (!impl_->pending) return true;
  cuda_check(cudaSetDevice(impl_->observed.destination_device),
             "cudaSetDevice(query completion)");
  const auto state = cudaEventQuery(impl_->completion);
  if (state == cudaErrorNotReady) return false;
  cuda_check(state, "cudaEventQuery(relay completion)");
  impl_->pending = false;
  return true;
}

void XRelay::synchronize() {
  std::scoped_lock lock(impl_->mutex);
  if (!impl_->pending) return;
  cuda_check(cudaSetDevice(impl_->observed.destination_device),
             "cudaSetDevice(synchronize completion)");
  cuda_check(cudaEventSynchronize(impl_->completion),
             "cudaEventSynchronize(relay completion)");
  impl_->pending = false;
}

std::size_t XRelay::capacity_bytes() const noexcept { return impl_->capacity; }

std::size_t XRelay::chunk_bytes() const noexcept { return impl_->chunk; }

unsigned int XRelay::staging_buffers() const noexcept { return impl_->buffers; }

const RouteObservation& XRelay::route() const noexcept { return impl_->observed; }

}  // namespace numabraid::forwarding
