//======================================================//
//  Shared/PBM/PBMStateOwner.hpp
//  RAII owner for durable ALiBi/RoPE PBM device state
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "PositionalBiasMethod.hpp"

namespace GRIM::PBM {

// Owns the durable PBM buffers used by attention:
//   - ALiBi slopes device + host mirror
//   - RoPE inverse frequencies device + host mirror
//   - upload event for cross-stream synchronization
//   - grouped construction HP snapshot
//
// This is model-level durable state. It is NOT TrainingState workspace, NOT
// BatchPayload data, and NOT autograd tape state. Consumers borrow PBMSpec or
// const PBMState views; only this owner releases PBMState resources.
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
    void ensure(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose = false);
    void reset() noexcept;

    bool initialized() const noexcept { return state_.initialized; }

    const PBMState& state() const;
    PBMSpec spec() const;

    const float* alibiSlopes() const;
    const float* ropeInvFreq() const;
    const std::vector<float>& alibiSlopesHost() const;
    const std::vector<float>& ropeInvFreqHost() const;

private:
    PBMState state_{};
};

} // namespace GRIM::PBM