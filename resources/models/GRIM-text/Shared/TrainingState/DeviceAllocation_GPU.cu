//======================================================//
//  DeviceAllocation_GPU.cu
//  RAII wrapper for untyped CUDA device allocations
//======================================================//

#include "DeviceAllocation_GPU.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace GRIM {

DeviceAllocation::~DeviceAllocation() {
	reset();
}

DeviceAllocation::DeviceAllocation(DeviceAllocation&& other) noexcept
	: ptr(other.ptr), bytes(other.bytes) {
	other.ptr = nullptr;
	other.bytes = 0;
}

DeviceAllocation& DeviceAllocation::operator=(DeviceAllocation&& other) noexcept {
	if (this != &other) {
		reset();
		ptr = other.ptr;
		bytes = other.bytes;
		other.ptr = nullptr;
		other.bytes = 0;
	}
	return *this;
}

void DeviceAllocation::allocate(std::size_t requested_bytes, const char* label) {
	if (requested_bytes == 0) {
		throw std::runtime_error(std::string(label ? label : "DeviceAllocation") + " allocation requested zero bytes");
	}
	reset();
	cudaError_t err = cudaMalloc(&ptr, requested_bytes);
	if (err != cudaSuccess) {
		ptr = nullptr;
		bytes = 0;
		throw std::runtime_error(std::string(label ? label : "DeviceAllocation") + " cudaMalloc failed: " + cudaGetErrorString(err));
	}
	bytes = requested_bytes;
}

void DeviceAllocation::reset() {
	if (ptr) {
		cudaFree(ptr);
		ptr = nullptr;
	}
	bytes = 0;
}

} // namespace GRIM

#endif // USE_CUDA