//======================================================//
//  ModelForwardRuntimePayload.hpp
//  Reserved mutable runtime boundary for shared forward
//======================================================//

#pragma once

#ifdef USE_CUDA

namespace GRIM {
namespace Forward {

// Shared forward currently has no post-bootstrap mutable runtime. Retaining
// the explicit payload boundary avoids forcing replacement verifier state into
// ModelForwardRequest later.
struct ModelForwardRuntimePayload {};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
