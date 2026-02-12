#pragma once

#include <cstddef>
#include <cstdint>

namespace GRIM {

// Shared cache limits so staging helpers and training state stay aligned.
inline constexpr size_t kMaxCachedBatch = 7;
inline constexpr size_t kMaxCachedSeqLen = 8192;

// BatchPreparationResult DELETED — replaced by GRIM::Batching::BatchPayload
// prepareLossBatchInputs DELETED — replaced by GRIM::Batching::buildBatchPayload

}  // namespace GRIM
