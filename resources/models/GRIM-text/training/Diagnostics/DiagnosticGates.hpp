#pragma once
//======================================================//
//  DiagnosticGates.hpp
//  Shared diagnostic gating helpers + debug macros used
//  across the extracted Diagnostics/*.cu files. These
//  were originally in the anonymous namespace at the top
//  of Phase2_TrainingLoop.cu.
//
//  All functions are inline so the header is the single
//  point of definition; no .cpp is needed.
//======================================================//

#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

inline int readEnvInt(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw) return fallback;
    if (value < 0) return fallback;
    return static_cast<int>(value);
}

inline std::string readEnvString(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    return std::string(raw);
}

inline bool isPhase2DebugEnabled() {
    static const bool enabled = readEnvInt("GRIM_PHASE2_DEBUG", 0) > 0;
    return enabled;
}

#define PHASE2_DEBUG_STDERR(...)                          \
    do {                                                  \
        if (::GRIM::Diagnostics::isPhase2DebugEnabled()){ \
            fprintf(stderr, __VA_ARGS__);                 \
        }                                                 \
    } while (0)

#define PHASE2_DEBUG_FLUSH_STDERR()                       \
    do {                                                  \
        if (::GRIM::Diagnostics::isPhase2DebugEnabled()){ \
            fflush(stderr);                               \
        }                                                 \
    } while (0)

// Forward declarations for definitions below — these need TrainingContext's
// full layout, so we define them in DiagnosticGates_inline.hpp included by
// callers that have already included Phase2_TrainingLoop.hpp.
bool shouldSyncDiagnostics(const GRIMText::Training::TrainingContext& ctx, std::size_t batch_idx);
bool shouldLogLogitTrace(const GRIMText::Training::TrainingContext& ctx, std::size_t batch_idx);
bool shouldLogAtomStats(const GRIMText::Training::TrainingContext& ctx, int batch_idx);

} // namespace GRIM::Diagnostics
