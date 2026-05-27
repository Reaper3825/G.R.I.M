//======================================================//
//  Shared/PBM/PBMStateOwner.hpp
//  RAII owner for durable ALiBi/RoPE PBM device state
//======================================================//

#pragma once

#include <cuda_runtime.h>

#include "PositionalBiasMethod.hpp"

namespace GRIM::PBM {

// Owns the durable PBM device buffers used by attention:
//   - ALiBi slopes device upload target
//   - RoPE inverse frequencies device upload target
//   - upload event for cross-stream synchronization
//
// This is model-level durable state. It is NOT TrainingState workspace, NOT
// BatchPayload data, and NOT autograd tape state. Consumers borrow const
// PBMState views; only this owner releases PBMState resources.
class PBMStateOwner final {
public:
    PBMStateOwner() = default;
    PBMStateOwner(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose = false);
    ~PBMStateOwner();

    PBMStateOwner(const PBMStateOwner&) = delete;
    PBMStateOwner& operator=(const PBMStateOwner&) = delete;

    PBMStateOwner(PBMStateOwner&& other) noexcept;
    PBMStateOwner& operator=(PBMStateOwner&& other) noexcept;

    void initialize(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose = false);
    bool initialized() const noexcept;

    const PBMState& state() const;

private:
    PBMState state_{};
};

} // namespace GRIM::PBM