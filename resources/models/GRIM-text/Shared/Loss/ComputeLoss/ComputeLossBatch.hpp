// ComputeLossBatch.hpp — DELETED
//
// kMaxCachedBatch / kMaxCachedSeqLen were DEAD CODE (zero callers).
// Actual cache limits flow through:
//   LanguageModelConfig.max_cached_batch / max_cached_seq_len
//     → TrainingState.max_cached_batch / max_cached_seq_len / max_cached_tokens
//     → BatchPayload.validate() enforces cache fit at batch construction time
//
// BatchPreparationResult DELETED — replaced by GRIM::Batching::BatchPayload
// prepareLossBatchInputs DELETED — replaced by GRIM::Batching::buildBatchPayload
