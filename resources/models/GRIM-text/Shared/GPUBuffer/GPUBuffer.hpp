//======================================================//
//  Shared/GPUBuffer/GPUBuffer.hpp
//  Standalone GPU buffer management utility
//======================================================//

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace GRIM {

#ifdef USE_CUDA

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
								   std::to_string(__LINE__) + " - " + \
								   cudaGetErrorString(err)); \
		} \
	} while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) \
	do { \
		cublasStatus_t status = call; \
		if (status != CUBLAS_STATUS_SUCCESS) { \
			throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
								   std::to_string(__LINE__)); \
		} \
	} while(0)
#endif

// GPU memory buffer with optional pinned-host backing storage
template<typename T>
class GPUBuffer {
public:
	GPUBuffer() = default;
	~GPUBuffer() { free(); }

	GPUBuffer(const GPUBuffer&) = delete;
	GPUBuffer& operator=(const GPUBuffer&) = delete;

	GPUBuffer(GPUBuffer&& other) noexcept {
		moveFrom(other);
	}

	GPUBuffer& operator=(GPUBuffer&& other) noexcept {
		if (this != &other) {
			free();
			moveFrom(other);
		}
		return *this;
	}

	bool allocate(size_t size, bool use_pinned_host = false);
	bool upload(const std::vector<T>& host_data, cudaStream_t stream = 0);

	template<typename Alloc>
	bool upload(const std::vector<T, Alloc>& host_data, cudaStream_t stream = 0) {
		if (!device_ptr_ || host_data.size() != size_) return false;

		const void* src = is_pinned_ ? pinned_host_ptr_ : host_data.data();
		if (is_pinned_) {
			std::memcpy(pinned_host_ptr_, host_data.data(), size_ * sizeof(T));
		}

		if (stream == 0) {
			CUDA_CHECK(cudaMemcpy(device_ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice));
		} else {
			CUDA_CHECK(cudaMemcpyAsync(device_ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
		}
		return true;
	}

	bool download(std::vector<T>& host_data, cudaStream_t stream = 0);
	void free();

	T* ptr() { return static_cast<T*>(device_ptr_); }
	const T* ptr() const { return static_cast<const T*>(device_ptr_); }
	T* hostPtr() { return static_cast<T*>(pinned_host_ptr_); }
	const T* hostPtr() const { return static_cast<const T*>(pinned_host_ptr_); }
	size_t size() const { return size_; }
	bool isPinned() const { return is_pinned_; }

private:
	void moveFrom(GPUBuffer& other) noexcept {
		device_ptr_ = other.device_ptr_;
		pinned_host_ptr_ = other.pinned_host_ptr_;
		size_ = other.size_;
		is_pinned_ = other.is_pinned_;

		other.device_ptr_ = nullptr;
		other.pinned_host_ptr_ = nullptr;
		other.size_ = 0;
		other.is_pinned_ = false;
	}

	void* device_ptr_ = nullptr;
	void* pinned_host_ptr_ = nullptr;
	size_t size_ = 0;
	bool is_pinned_ = false;
};

template<typename T>
bool GPUBuffer<T>::allocate(size_t size, bool use_pinned_host) {
	if (device_ptr_ && size_ == size && is_pinned_ == use_pinned_host) {
		return true;
	}

	if (device_ptr_) {
		cudaFree(device_ptr_);
		device_ptr_ = nullptr;
	}
	if (pinned_host_ptr_) {
		cudaFreeHost(pinned_host_ptr_);
		pinned_host_ptr_ = nullptr;
	}

	size_ = size;
	is_pinned_ = use_pinned_host;

	CUDA_CHECK(cudaMalloc(&device_ptr_, size * sizeof(T)));
	if (is_pinned_) {
		CUDA_CHECK(cudaMallocHost(&pinned_host_ptr_, size * sizeof(T)));
	}
	return true;
}

template<typename T>
bool GPUBuffer<T>::upload(const std::vector<T>& host_data, cudaStream_t stream) {
	if (!device_ptr_ || host_data.size() != size_) return false;

	const void* src = is_pinned_ ? pinned_host_ptr_ : host_data.data();
	if (is_pinned_) {
		std::memcpy(pinned_host_ptr_, host_data.data(), size_ * sizeof(T));
	}

	if (stream == 0) {
		CUDA_CHECK(cudaMemcpy(device_ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice));
	} else {
		CUDA_CHECK(cudaMemcpyAsync(device_ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
	}
	return true;
}

template<typename T>
bool GPUBuffer<T>::download(std::vector<T>& host_data, cudaStream_t stream) {
	if (!device_ptr_) return false;
	host_data.resize(size_);

	void* host_ptr = host_data.data();
	void* dst = is_pinned_ ? pinned_host_ptr_ : host_ptr;

	if (stream == 0) {
		CUDA_CHECK(cudaMemcpy(dst, device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
		if (is_pinned_) {
			std::memcpy(host_ptr, pinned_host_ptr_, size_ * sizeof(T));
		}
	} else {
		CUDA_CHECK(cudaMemcpyAsync(dst, device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
		if (is_pinned_) {
			CUDA_CHECK(cudaStreamSynchronize(stream));
			std::memcpy(host_ptr, pinned_host_ptr_, size_ * sizeof(T));
		}
	}

	return true;
}

template<typename T>
void GPUBuffer<T>::free() {
	if (device_ptr_) {
		cudaFree(device_ptr_);
		device_ptr_ = nullptr;
	}
	if (pinned_host_ptr_) {
		cudaFreeHost(pinned_host_ptr_);
		pinned_host_ptr_ = nullptr;
	}
	size_ = 0;
	is_pinned_ = false;
}

#endif // USE_CUDA

} // namespace GRIM

