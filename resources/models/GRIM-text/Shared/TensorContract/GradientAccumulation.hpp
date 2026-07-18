#pragma once
//======================================================//
//  GradientAccumulation.hpp
//  Shared CUDA gradient accumulation primitive.
//
//  Contract: dst += src * scale
//  This is the TensorContract GradFn accumulation path for persistent leaf
//  gradient buffers and owned non-leaf scratch buffers. Do not duplicate
//  per-TU kernel_accumulate_grad copies in GradFn files.
//======================================================//

#include <cuda_runtime.h>

#include <cstddef>

namespace GRIM {

struct Tensor;

namespace autograd {

void accumulate_grad(Tensor& dst,
                     const Tensor& src,
                     float scale,
                     cudaStream_t stream,
                     const char* caller);

void accumulate_grad(float* dst,
                     const float* src,
                     std::size_t count,
                     float scale,
                     cudaStream_t stream,
                     const char* caller);

}  // namespace autograd
}  // namespace GRIM
