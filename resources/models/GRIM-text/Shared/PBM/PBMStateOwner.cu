//======================================================//
//  Shared/PBM/PBMStateOwner.cu
//  RAII owner for durable ALiBi/RoPE PBM device state
//======================================================//

#include "PBMStateOwner.hpp"

#include <stdexcept>
#include <string>

namespace GRIM::PBM {
namespace {

void requireExplicitStream(cudaStream_t stream, const char* caller) {
    bool is_default = (stream == nullptr);
#if defined(cudaStreamLegacy)
    is_default = is_default || (stream == cudaStreamLegacy);
#endif
#if defined(cudaStreamPerThread)
    is_default = is_default || (stream == cudaStreamPerThread);
#endif

    if (is_default) {
        throw std::runtime_error(std::string(caller) +
                                 ": stream is NULL/default - caller MUST provide the StreamController primary stream");
    }
}

PBMState stealState(PBMState& other) noexcept {
    PBMState moved{};
    moved.alibi_slopes = other.alibi_slopes;
    moved.rope_inv_freq = other.rope_inv_freq;
    moved.upload_event = other.upload_event;
    moved.initialized = other.initialized;

    other.alibi_slopes = nullptr;
    other.rope_inv_freq = nullptr;
    other.upload_event = nullptr;
    other.initialized = false;

    return moved;
}

} // namespace

PBMStateOwner::PBMStateOwner(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose) {
    initialize(hp, stream, verbose);
}

PBMStateOwner::~PBMStateOwner() {
    releasePBM(state_);
}

PBMStateOwner::PBMStateOwner(PBMStateOwner&& other) noexcept
    : state_(stealState(other.state_))
{
}

PBMStateOwner& PBMStateOwner::operator=(PBMStateOwner&& other) noexcept {
    if (this != &other) {
        releasePBM(state_);
        state_ = stealState(other.state_);
    }
    return *this;
}

void PBMStateOwner::initialize(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose) {
    requireExplicitStream(stream, "PBMStateOwner::initialize");
    if (state_.initialized) {
        throw std::runtime_error("PBMStateOwner::initialize: PBM state is already initialized - rebuilding PBM resources is unsupported");
    }

    PBMRuntimeOptions runtime{};
    runtime.stream = stream;
    runtime.verbose = verbose;

    if (!initializePBM(hp, state_, runtime)) {
        throw std::runtime_error("PBMStateOwner::initialize: initializePBM failed");
    }
    if (!state_.initialized) {
        throw std::runtime_error("PBMStateOwner::initialize: initializePBM returned success but state is not initialized");
    }
    if (!state_.alibi_slopes || !state_.rope_inv_freq || !state_.upload_event) {
        throw std::runtime_error("PBMStateOwner::initialize: initialized state has NULL PBM resource pointer");
    }
}

bool PBMStateOwner::initialized() const noexcept {
    return state_.initialized;
}

const PBMState& PBMStateOwner::state() const {
    if (!state_.initialized) {
        throw std::runtime_error("PBMStateOwner::state: PBM state is not initialized");
    }
    return state_;
}

} // namespace GRIM::PBM