//======================================================//
//  BatchDeviceUpload.hpp
//
//  Explicit H2D sync boundary for one BatchPayload.
//  Shared/Batching owns this because it translates caller-authored host
//  payload semantics into the borrowed BatchDeviceBindings device view.
//======================================================//

#pragma once

#include "BatchPayload.hpp"
#include "BatchDeviceBindings.hpp"
#include "BatchDeviceStorage.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM {
namespace Batching {

BatchDeviceBindings uploadBatchToDevice(
    const Config::AiConfigSnapshot& config,
    BatchPayload& payload,
    cudaStream_t stream);

}  // namespace Batching
}  // namespace GRIM