#include "MemorySnapshot.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>

namespace GRIMText::Training {

MemorySnapshot captureMemorySnapshotOrThrow() {
    MemorySnapshot snap;

#ifdef USE_CUDA
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }
    snap.device = dev;

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, dev);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));
    }
    snap.device_name = props.name;

    size_t free_b = 0, total_b = 0;
    err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(err));
    }
    snap.free_bytes = static_cast<std::uint64_t>(free_b);
    snap.total_bytes = static_cast<std::uint64_t>(total_b);
#else
    throw std::runtime_error("captureMemorySnapshotOrThrow requires USE_CUDA");
#endif

    return snap;
}

} // namespace GRIMText::Training

