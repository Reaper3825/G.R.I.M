//======================================================//
//  CublasHandleOwner_GPU.hpp
//  RAII owner for a cuBLAS handle
//======================================================//

#pragma once

#ifdef USE_CUDA
#include <cublas_v2.h>

namespace GRIM {

struct CublasHandleOwner {
	cublasHandle_t handle = nullptr;

	CublasHandleOwner() = default;
	~CublasHandleOwner();

	CublasHandleOwner(const CublasHandleOwner&) = delete;
	CublasHandleOwner& operator=(const CublasHandleOwner&) = delete;
	CublasHandleOwner(CublasHandleOwner&&) = delete;
	CublasHandleOwner& operator=(CublasHandleOwner&&) = delete;

	cublasHandle_t get() const { return handle; }
	cublasHandle_t* outParam() { return &handle; }
	explicit operator bool() const { return handle != nullptr; }
	operator cublasHandle_t() const { return handle; }
};

} // namespace GRIM

#endif // USE_CUDA