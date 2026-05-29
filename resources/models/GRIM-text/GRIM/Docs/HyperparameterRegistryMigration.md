# Hyperparameter Registry Migration Playbook

This document records the current GRIM-text config migration pattern: one raw snapshot document, one HyperParameters compute/validate boundary, transitional typed handoffs only where old CUDA/model consumers still require direct fields, and immutable grouped read views wherever repeated subsystem slices are needed.

## Target shape

1. `control/ai_config_paths.hpp`
   Owns only `AiConfigSnapshot::document` and the strict raw load operation.
2. `Shared/HyperParameters/HyperParameters_GPU.hpp`
   Consumes `AiConfigSnapshot::document` directly through `finalizeLanguageModelConfig(document, argc, argv, mode)`, performs enum/string parsing, derivation, validation, execution-mode stamping, and path resolution. The remaining `LanguageModelConfig` return is a transitional compatibility handoff for downstream code that still uses direct fields; it is not the root source. There is no transitional `TrainingHyperparameters` alias, `HyperparameterMappings` alias payload, `GenerationConfig` sidecar, or caller-selected validation mode.
3. `Shared/HyperParameters/HyperparameterGroupings.hpp`
   Owns immutable grouped read views. Snapshot-backed overloads are valid only when they consume the raw snapshot document directly or values already computed/validated by HyperParameters; they must not route back through `LanguageModelConfig`.
4. Consumer code
   Consumes the explicit root config it already owns, or an immutable grouping payload. It must not rebuild schema wrappers.

## Root-flattening rule

Flatten authored/config-policy fields into the HyperParameters document finalization boundary first. Do not preserve old subconfig types by wrapping them in new helpers.

- `AiConfigSnapshot::document` is the raw root; `HyperParameters_GPU.hpp` computes and validates; `HyperparameterGroupings.hpp` exposes immutable views. `LanguageModelConfig` is transitional compatibility for direct-field consumers only.
- Do not reintroduce the old training-root alias as an alias, struct, wrapper, base class, or nested owner.
- Do not recreate `ModelArchitecture` or a model-architecture sidecar.
- Do not recreate `GenerationConfig`, `StartupConfig::generation`, or `LanguageModelConfig::generation`; generation authored leaves live directly on `LanguageModelConfig` and are sliced through `GenerationHP` only after final root validation.
- Log recorder and tape logging leaves live directly on `LanguageModelConfig`; `LogRecorderHP` and `TapeLogHP` are immutable views only.
- Groupings are the only place where a subsystem-oriented shape should exist after the root owner is finalized.
- New grouping slicers should prefer finalized snapshot/document-backed values and must not route through alias types.

## Migration steps

When flattening a config subcategory, do exactly this:

1. Add or verify flat root registry entries in `HyperParameters_GPU.hpp`.
   Add authored leaves to `LanguageModelConfig` and direct reads to `loadLanguageModelConfig(const nlohmann::json& document)`. If a root field is enum-native, normalize it in that same root loader so no side alias payload is created.

2. Keep derivation/validation in `HyperParameters_GPU.hpp`.
   Formula-derived values and cross-field checks belong on the root owner before any grouping is sliced.

3. Slice immutable views in `HyperparameterGroupings.hpp`.
   If consumers need a subsystem payload, create or update a `*HP` view and copy finalized values from the root owner.

4. Remove old wrapper/type aliases and helper handoffs.
   Delete nested config structs, typed subobjects, casts, and wrapper-returning mapping helpers once the root fields and grouping slices are live.

5. Update companion docs and validate.
   Run editor diagnostics/build checks and update `Config.md` or the feature doc that named the removed owner.

## Completed root collapses

- Removed `ModelArchitecture`; architecture fields now live directly on `LanguageModelConfig`.
- Removed `LogRecorderConfig`, `LogRecorderConfig::LayerEnables`, and `TapeLogConfig`; their leaves now live directly on `LanguageModelConfig`, while `logRecorderHP()` and `tapeLogHP()` remain immutable read views.
- Removed the duplicate concrete training-root struct and transitional alias; startup/inference model config assembly copies the concrete `LanguageModelConfig` root directly.
- Removed `HyperparameterMappings.hpp` and the `mapHyperparameterDocument()` staging payload; `loadLanguageModelConfig(const nlohmann::json&)` is now the single root registry from the raw document to `LanguageModelConfig`.
- Collapsed split validation helpers (`validateLossDocument`, `validateOptimizerDocument`, `validateTokenizerDocument`, `validateExecutionBlockHyperparameters`, `validateGQADocument`, `validatePBMDocument`, `validateFlashAttentionDocument`, `validateLanguageModelDocument`, and `validateLanguageModelCacheCapacity`) into the single root `validateRootConfigDocument(const LanguageModelConfig&, const char*)` operation.
- Pruned the caller-selected validation-mode enum; root validation is only called on fully stamped startup/inference `LanguageModelConfig` values and validates cache capacity with `max_tokens_per_batch == batch_size * max_cached_seq_len`.
- Added `finalizeLanguageModelConfig(document, argc, argv, mode)` so executables load `AiConfigSnapshot` once and pass `snapshot.document` directly into HyperParameters before Phase1 receives the transitional compatibility handoff.

## What to flatten next

Pick the next smallest remaining typed subobject or wrapper helper that mirrors flat `training.config` leaves. Good targets are structures that:

- exist only to regroup fields already present on `LanguageModelConfig`,
- are stored as a nested member of an old config wrapper or staged before a final root handoff,
- already have, or naturally need, a `HyperparameterGroupings.hpp` immutable view.

Do not choose consumer cleanup alone as the next step. Consumer changes are only valid after the root owner has been flattened and the grouping slice exists.

## Rules for future agents

1. Do not add raw `document.at(...)` reads in consumer code. If a migration pass
   needs snapshot-backed builders before `LanguageModelConfig` is deleted, keep
   them centralized in `HyperparameterGroupings.hpp`, keep them obviously
   transitional, and do not add new architectural dependency chains that route
   snapshot builders back through `LanguageModelConfig`.
2. Do not add new grouped read views in consumer files.
3. Do not let consumers read root owner internals once a grouping exists for that subsystem.
4. Do not turn groupings into mutable owners.
5. Do not create sidecar wrappers in startup code.
6. Do not preserve removed subconfig shapes through helper-returned structs.
7. Prefer one small root-flattening island at a time.

## Minimal checklist

For each subcategory:

1. Verify flat raw leaves in `AiConfigSnapshot::document` and the centralized HyperParameters finalization/load helpers.
2. Compute and validate values in `HyperParameters_GPU.hpp` without adding a new owner.
3. Slice or update immutable `*HP` views in `HyperparameterGroupings.hpp`.
4. Delete the old nested owner/wrapper/helper.
5. Update docs.
6. Validate.