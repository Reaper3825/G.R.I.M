//======================================================//
//  ModelForwardExecutionRuntime.hpp
//  Compatibility shell retained during verifier replacement
//======================================================//

#pragma once

#ifdef USE_CUDA

namespace GRIM {
namespace Forward {

// The legacy execution trace/diagnostic runtime ended at the argument-seed
// boundary. Keep the named shell temporarily so startup-owned context layouts
// do not need to change in the same cut.
struct ModelForwardExecutionRuntime {
    void clear() {}
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
