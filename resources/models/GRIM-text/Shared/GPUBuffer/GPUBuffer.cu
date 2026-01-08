//======================================================//
//  Shared/GPUBuffer/GPUBuffer.cu
//  Explicit template instantiations for common buffer types
//======================================================//

#ifdef USE_CUDA

#include <cuda_fp16.h>

#include "GPUBuffer.hpp"

namespace GRIM {

template class GPUBuffer<float>;
template class GPUBuffer<__half>;
template class GPUBuffer<int>;

} // namespace GRIM

#endif // USE_CUDA
