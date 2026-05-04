//======================================================//
//  CublasHandleOwner_GPU.cu
//  RAII owner for a cuBLAS handle
//======================================================//

#include "CublasHandleOwner_GPU.hpp"

#ifdef USE_CUDA

namespace GRIM {

CublasHandleOwner::~CublasHandleOwner() {
	if (handle) {
		cublasDestroy(handle);
		handle = nullptr;
	}
}

} // namespace GRIM

#endif // USE_CUDA