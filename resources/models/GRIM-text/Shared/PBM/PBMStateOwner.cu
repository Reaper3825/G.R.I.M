//======================================================//
//  Shared/PBM/PBMStateOwner.cu
//  RAII owner for durable ALiBi/RoPE PBM device state
//======================================================//

#include "PBMStateOwner.hpp"

#include <stdexcept>
#include <string>
#include <utility>

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
    moved.alibi_slopes_host = std::move(other.alibi_slopes_host);
    moved.rope_inv_freq = other.rope_inv_freq;
    moved.rope_inv_freq_host = std::move(other.rope_inv_freq_host);
    moved.upload_event = other.upload_event;
    moved.initialized = other.initialized;

    other.alibi_slopes = nullptr;
    other.rope_inv_freq = nullptr;
    other.upload_event = nullptr;
    other.initialized = false;
    other.alibi_slopes_host.clear();
    other.rope_inv_freq_host.clear();

    return moved;
}

} // namespace

PBMStateOwner::PBMStateOwner(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose) {
    initialize(hp, stream, verbose);
}

PBMStateOwner::~PBMStateOwner() {
    reset();
}

PBMStateOwner::PBMStateOwner(PBMStateOwner&& other) noexcept
    : state_(stealState(other.state_))
{
}

PBMStateOwner& PBMStateOwner::operator=(PBMStateOwner&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = stealState(other.state_);
    }
    return *this;
}

void PBMStateOwner::initialize(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose) {
    requireExplicitStream(stream, "PBMStateOwner::initialize");
    if (state_.initialized) {
        throw std::runtime_error("PBMStateOwner::initialize: PBM state is already initialized - call ensure() or reset() explicitly");
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

void PBMStateOwner::ensure(const PBMConstructionHP& hp, cudaStream_t stream, bool verbose) {
    requireExplicitStream(stream, "PBMStateOwner::ensure");

    PBMRuntimeOptions runtime{};
    runtime.stream = stream;
    runtime.verbose = verbose;

    if (!ensurePBM(hp, state_, runtime)) {
        throw std::runtime_error("PBMStateOwner::ensure: ensurePBM failed");
    }
    if (!state_.initialized) {
        throw std::runtime_error("PBMStateOwner::ensure: ensurePBM returned success but state is not initialized");
    }
    if (!state_.alibi_slopes || !state_.rope_inv_freq || !state_.upload_event) {
        throw std::runtime_error("PBMStateOwner::ensure: initialized state has NULL PBM resource pointer");
    }
}

void PBMStateOwner::reset() noexcept {
    releasePBM(state_);
}

const PBMState& PBMStateOwner::state() const {
    requirePBMInitialized(state_, "PBMStateOwner::state");
    return state_;
}

PBMSpec PBMStateOwner::spec() const {
    return getPBMSpec(state());
}

const float* PBMStateOwner::alibiSlopes() const {
    return getAlibiSlopes(state());
}

const float* PBMStateOwner::ropeInvFreq() const {
    return getRoPEInvFreq(state());
}

const std::vector<float>& PBMStateOwner::alibiSlopesHost() const {
    return state().alibi_slopes_host;
}

const std::vector<float>& PBMStateOwner::ropeInvFreqHost() const {
    return state().rope_inv_freq_host;
}

} // namespace GRIM::PBM