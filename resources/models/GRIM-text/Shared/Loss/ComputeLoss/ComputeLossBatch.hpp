// ComputeLossBatch.hpp — DELETED
//
// kMaxCachedBatch / kMaxCachedSeqLen were DEAD CODE (zero callers).
// Actual cache limits flow through:
//   LanguageModelConfig.max_cached_batch / max_cached_seq_len
//     → TrainingState Tensor allocation shapes
//     → BatchPayload.validate() enforces cache fit at batch construction time
//
// Per-batch seq_len: single source of truth is GRIM::Batching::BatchPayload
// (payload.batch_size, payload.max_seq_len, payload.total_tokens, etc.).
// Do not read sequence length from config when a payload is available.
//
// BatchPreparationResult DELETED — replaced by GRIM::Batching::BatchPayload
// prepareLossBatchInputs DELETED — replaced by GRIM::Batching::buildBatchPayload
