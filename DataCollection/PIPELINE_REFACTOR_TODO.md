# Data Collection Pipeline Refactor TODO

## Architecture Boundary
- **GRIM.exe** (Data Collection) → Outputs `merged_verified_{ID}.jsonl`
- **Training Loop** (Separate) → Consumes `.jsonl` → Creates `.grmt` + vocab
- **`ai_config.json`** = Single source of truth for all paths

---

## Tasks

### Phase 1: Source Type Flexibility ✅ COMPLETE
- [x] Remove hardcoded `SourceType` enum dependency in fetchers
  - Replaced with `FetcherType` enum (8 API types only)
  - Added `source_type_str` field for user-defined categories
- [x] Make `source_type` field optional in `source_data.json` parsing
- [x] Default `source_type` to `"miscellaneous"` when not provided
- [x] Fetcher selection is ALWAYS automatic via URL pattern detection
  - `detectFetcherFromUrl()` analyzes URL patterns (no manual override)
  - Removed explicit `"fetcher"` field support from JSON parsing

### Phase 2: Pipeline Restructure ✅ COMPLETE
The pipeline now runs 5 steps in correct order:

- [x] **Step 1 - Load Data** (`runMerge` step 1/5)
  - [x] Load HuggingFace downloads (JSONL, TXT, PDF)
  - [x] Load checkpoint files (`.ckpt` FlatBuffer format)
  - [x] Load existing verified entries

- [x] **Step 2 - Deduplicate** (`runMerge` step 2/5)
  - [x] Content hash deduplication via `CollectionStateManager`
  - [x] Session-local hash deduplication
  - [x] `--force-rebuild` flag to bypass deduplication state

- [x] **Step 3 - Verify** (`runMerge` step 3/5) ✅ NEW
  - [x] ALL data now goes through Verifier (including HuggingFace)
  - [x] Same quality standards applied to all sources
  - [x] Verification happens AFTER merge, BEFORE preprocessing
  - [x] Progress callback integrated with UI

- [x] **Step 4 - Preprocess** (`runMerge` step 4/5)
  - [x] HTML removal, length filtering, quality filters
  - [x] Long text chunking
  - [x] Special token insertion

- [x] **Step 5 - Output** (`runMerge` step 5/5)
  - [x] Write `merged_verified_cache.jsonl`
  - [x] Save state manager for future runs

### Phase 3: Code Cleanup ✅ COMPLETE
- [x] `merge_checkpoints.cpp` is legacy - `grim_data_pipeline.cpp` is the active tool
- [x] `web_collector.hpp` updated with FetcherType (Phase 1)
- [x] All paths loaded from `ai_config.json` in `StartDataCollection()`

### Phase 4: Progress Reporting ✅ COMPLETE
- [x] `updateCollectionProgress()` now accepts phase label parameter
- [x] `runMerge()` updates `currentPhase` at each step
- [x] Phase labels: "Loading", "Deduplicating", "Verifying", "Preprocessing", "Writing"
- [x] Progress written to FlatBuffer status file for UI consumption
- [x] Progress percentages mapped to step ranges (0-5%, 5-25%, 25-35%, 35-55%, 55-85%, 85-100%)

### Phase 5: Testing
- [ ] Test with empty `.grmt` (new collection)
- [ ] Test with existing `.grmt` (append mode)
- [ ] Test with missing `source_type` in config
- [ ] Verify deduplication works across all 3 streams
- [ ] Verify progress updates show in UI

---

## Files to Modify
1. `DataCollection/web_collector.hpp` - Source type flexibility
2. `DataCollection/grim_data_pipeline.cpp` - Pipeline restructure
3. `DataCollection/source_data.json` - Example with optional source_type
4. `ui/ui_DataCollection.cpp` - UI integration (if needed)

## Files to Reference (Read-Only)
- `ai_config.json` - Path configuration
- `control/ai_config_paths.hpp` - Path loading utilities

---

## Progress Log

### Session: December 13, 2025
- Created TODO tracking file
- **Phase 1 COMPLETE:**
  - Replaced `SourceType` enum (24 hardcoded values) with `FetcherType` (8 API types)
  - Added `source_type_str` for user-defined categories (defaults to "miscellaneous")
  - Added `detectFetcherFromUrl()` for automatic fetcher selection
  - Updated `initializeDefaultFetchers()` to use new enum
  - Updated `fetchFromSource()` logging to show fetcher + category
  - Legacy `SourceType` alias maintained for backward compatibility
- Next: Phase 2 - Pipeline Restructure (Step 1: Collection)

