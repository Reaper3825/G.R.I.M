//======================================================//
//  DeviceAllocation_GPU.hpp
//  RAII wrapper for untyped CUDA device allocations
//======================================================//

#pragma once

#include <cstddef>

namespace GRIM {

struct DeviceAllocation {
	void* ptr = nullptr;
	std::size_t bytes = 0;

	DeviceAllocation() = default;
	~DeviceAllocation();

	DeviceAllocation(const DeviceAllocation&) = delete;
	DeviceAllocation& operator=(const DeviceAllocation&) = delete;
	DeviceAllocation(DeviceAllocation&& other) noexcept;
	DeviceAllocation& operator=(DeviceAllocation&& other) noexcept;

	void allocate(std::size_t requested_bytes, const char* label);
	void reset();
	void* get() const { return ptr; }
	template <typename T>
	T* as() const { return static_cast<T*>(ptr); }
	explicit operator bool() const { return ptr != nullptr; }
	operator void*() const { return ptr; }
	operator const void*() const { return ptr; }
};

} // namespace GRIM