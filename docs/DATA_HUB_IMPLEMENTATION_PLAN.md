# Data Hub — Implementation Plan

> Refactors the monolithic `UIDataCollectionPanel` into a **tabbed Data Hub** (`UIDataHubPanel`) that serves as both a data **collector** and a data **structurer** for the MMO sub-model ecosystem.
>
> **Reference file**: `ui_DataCollection.hpp/.cpp` is renamed `ui_DataCollection_OLD.hpp/.cpp` and kept as implementation reference. It is not compiled.

---

## Resolved Design Decisions

All open questions have been answered. These are locked in:

| # | Decision |
|---|----------|
| 1 | **Tab mechanism**: Single `UIDataHubPanel` (one `UIPanel`). Each tab is a widget group — clicking a tab hides the current group and shows the next. No sub-panel subclasses. |
| 2 | **Old panel**: Rename to `_OLD`, keep as reference. Not compiled. |
| 3 | **Panel name**: `"DataHub"`. |
| 4 | **File paths are cross-platform**. Global model root derived from repo root (`std::filesystem::path`). "Add new model" creates a temp directory; if name stays unset → delete temp; if name is set → rename after duplicate check. No hardcoded OS paths. |
| 5 | **Structured output is editable**. Two view modes in the Structurer: **Dataset View** (full dataset, controlled by model/dataset dropdown) and **Sequence View** (individual entry). |
| 6 | **"Structure" operates on the dataset selected by the model dropdown**. Each dropdown entry = a model entry = a folder/dataset. |
| 7 | **Single mass dataset** (`merged_verified_cache.jsonl`) shared across all models until per-model datasets are more mature. Same file the router uses. |
| 8 | **HF download flow**: download → append to mass dataset → delete original HF files. |
| 9 | **Source management**: own dedicated section with a `UIScrollBox` of current source entries (not inlined into Home). |
| 10 | **Config persistence**: structuring config persists to `ai_config.json` on panel close. |
| 11 | **Model dropdown**: all models from `ModelRegistry` (router + sub-models). |

---

## 1. Goals

1. **Tab-based navigation** replaces the current side-by-side two-column layout.
2. **Home HUD** — at-a-glance dashboard (progress, status, stats, controls).
3. **Sources** — dedicated source management with scrollbox of entries.
4. **HuggingFace Browser** — search, browse, queue, download HF datasets → append to mass dataset.
5. **Dataset Structurer** — SDG workspace with editable text editor, two view modes (Dataset / Sequence), model dropdown, format selector, interactive structured output editing.
6. **Cross-platform model folder management** — repo-root-relative paths, temp folder creation, rename-on-commit, duplicate check.
7. **Strict separation**: UI files own layout + events only; all logic lives in `DataCollection/` backend classes.

---

## 2. Architecture Overview

```
ui/
  ui_data_hub.hpp              // UIDataHubPanel (UIPanel subclass, tab host + widget groups)
  ui_data_hub.cpp
  ui_DataCollection_OLD.hpp    // RENAMED — kept as reference, not compiled
  ui_DataCollection_OLD.cpp

DataCollection/
  data_collection_manager.hpp/.cpp   // (existing) collection orchestration
  data_structurer.hpp                // (existing) SDG via Ollama
  huggingface_webhook.hpp/.cpp       // (existing) HF API
  dataset_target.hpp                 // NEW: cross-platform model folder + dataset path management
```

### Widget-group visibility model

```
UIDataHubPanel owns ALL widgets.
Each tab has a std::vector<std::shared_ptr<Widget>> of its members.

setView(tab):
  for widget in currentGroup: widget->setVisible(false)
  for widget in newGroup:     widget->setVisible(true)
  activeView_ = tab
```

No sub-view classes or separate files per tab. The tab host owns everything. Widget groups provide the separation.

### Dependency flow

```
UIDataHubPanel
  │
  ├─ [Home group]      ──callbacks──▷  DataCollectionManager
  ├─ [Sources group]   ──callbacks──▷  DataCollectionManager (source enable/disable/add)
  ├─ [HuggingFace group] ──callbacks──▷  HuggingFaceWebhook, DatasetTarget (append + delete)
  └─ [Structurer group]  ──callbacks──▷  DataStructurer, DatasetTarget
                                │
                                └─▷ ModelRegistry (populate model dropdown)
```

UI callbacks fire into logic classes. The UI never calls `curl`, parses JSON files, or touches the filesystem directly.

---

## 3. Tab Definitions

| Tab | Label | Purpose |
|-----|-------|---------|
| **Home** | "Home" | Dashboard HUD — pipeline status, progress bar, action buttons (Full / Collect / Verify / Merge / Rebuild / Stop), config sliders, dataset stats, log viewer. |
| **Sources** | "Sources" | Dedicated source management — scrollbox of `source_data.json` entries with enable/disable toggles, add-source input, filter buttons for source type/status. |
| **HuggingFace** | "HuggingFace" | Search bar, category dropdown, results scrollbox, download queue, HF token input, queue controls. Downloads append to mass dataset then delete originals. |
| **Structurer** | "Structurer" | SDG workspace — model dropdown (all `ModelRegistry` entries), format selector, two view modes (Dataset View / Sequence View), editable text editor, editable structured output, Structure / Save actions. |

---

## 4. Detailed Tab Specs

### 4.1 Home HUD

**Migrated from**: right column of current panel + left column config sliders.

| Section | Widgets | Data source |
|---------|---------|-------------|
| Status banner | Phase label, message, elapsed time | `CollectionStatus` |
| Progress | `UIProgressBar` (animated) | `CollectionStatus.progress` |
| Action row | Full ⏐ Collect ⏐ Verify ⏐ Merge ⏐ Rebuild ⏐ Stop buttons | `DataCollectionManager::startCollection(mode)` / `stopCollection()` |
| Config | Fetch limit, vocab size, verification threshold sliders | write into `CollectionConfig` |
| Stats | Dataset size, checkpoints, verification counts | `DataCollectionManager::getStatus()` |
| Log viewer | Scrollable log with auto-scroll | `logEntries` ring buffer |

### 4.2 Sources

**Migrated from**: left column source toggles + add-source section + filter buttons.

| Section | Widgets | Data source |
|---------|---------|-------------|
| Add source | `UIInputBox` + "Add" button | `DataCollectionManager` (persist to `source_data.json`) |
| Source list | `UIScrollBox` of source entries, each with enable/disable `UIToggle` | `source_data.json` entries loaded into `loadedSources` cache |
| Filters | Web ⏐ HuggingFace ⏐ All ⏐ Clear filter buttons | filter the scrollbox contents |
| Stats row | Source count, enabled/disabled counts | computed from `loadedSources` |

### 4.3 HuggingFace Browser

**Migrated from**: left column HF section + queue section.

| Section | Widgets | Data source |
|---------|---------|-------------|
| Search bar | `UIInputBox` + Search button | `hfWebhook->searchDatasets(query)` |
| Category filter | `UIDropdown` (11 categories) | `hfWebhook->searchByCategory(cat)` |
| Token input | `UIInputBox` (masked) | `hfWebhook->setToken(token)` |
| Results list | `UIScrollBox` of result cards with "Queue" button per card | `hfSearchResults[]` |
| Download queue | `UIScrollBox` of queued items with progress, retry, remove | `downloadQueue[]` |
| Queue controls | Process All ⏐ Clear ⏐ Pause buttons | queue management methods |

**Download flow**:
1. User queues dataset → download starts.
2. On completion: `DatasetTarget::appendToMassDataset(downloaded_entries)` appends to `merged_verified_cache.jsonl`.
3. After successful append: delete the downloaded HF source files.
4. UI updates queue item status to "merged" or "failed".

### 4.4 Dataset Structurer

**New tab** — the primary new functionality.

#### Model / Dataset dropdown

A single `UIDropdown` populated from `ModelRegistry::allModels()`. Each entry = a model = a folder = a dataset. Selecting a model:
- Sets the active `DatasetTarget` to that model's directory.
- Loads that model's dataset into the editor (in Dataset View) or first entry (in Sequence View).

#### View modes

| Mode | What it shows | Editing |
|------|--------------|---------|
| **Dataset View** | Full dataset file in the left editor (`UITextArea`). Right editor shows full structured output. Both editable. Scroll-synced. | Edit any line in either pane. |
| **Sequence View** | Single entry/sequence from the dataset. Navigation arrows or entry index input to move between entries. Left = raw source. Right = structured. | Edit the current entry in either pane. |

A `UIDropdown` or toggle switches between views. Default is Dataset View.

#### Layout

```
┌─────────────────────────────────────────────────────────┐
│ [Model ▼]  [Format ▼]  [View: Dataset ▼]  [Structure]  │
│ [Max entries ═══]  [Parallel ═══]          [Save]       │
├────────────────────────┬────────────────────────────────┤
│  Raw Source (editable) │  Structured Output (editable)  │
│  UITextArea            │  UITextArea                    │
│                        │                                │
│                        │                                │
├────────────────────────┴────────────────────────────────┤
│ Status: 124/500 structured  │  12 failed  │  3 skipped  │
└─────────────────────────────────────────────────────────┘
```

#### Event mapping

| UI event | Handler |
|----------|---------|
| Model dropdown changed | `DatasetTarget::setModel(id)` → load dataset into editor |
| Format dropdown changed | `DataStructuringConfig.mode = value` |
| View dropdown changed | toggle Dataset View ↔ Sequence View |
| "Structure" clicked | `DataStructurer::structureEntry(editor_text)` → populate right pane |
| "Structure All" clicked | `DataStructurer::structureBatch(all_entries)` with progress → populate right pane |
| "Save" clicked | `DatasetTarget::appendEntries(structured_output)` → write to model's dataset |
| Sequence nav arrows | load prev/next entry into both panes |

---

## 5. New Backend: `DatasetTarget`

Cross-platform model folder and dataset management with **ID-based sequence referencing**. All paths are `std::filesystem::path`, relative to a repo-root-derived model store root.

**File**: `DataCollection/dataset_target.hpp`

### 5.1 Sequence representation (UI-side)

The UI needs a lightweight struct to work with sequences. Until the ID system is implemented, `index` (line number in JSONL) serves as the temporary identifier.

```cpp
// Lightweight handle for a sequence in the mass dataset.
// `id` will be a persistent unique ID once the tagging plan lands.
// Until then, `index` (0-based line number) is the stand-in key.
struct SequenceHandle {
    size_t      index = 0;          // line index in mass dataset (temp key)
    std::string id;                 // future: persistent unique ID
    std::string content;            // raw text
    std::string structured;         // structured output (may be empty)

    // Metadata tags — empty until tagging plan lands.
    // UI filter dropdowns bind to these.
    std::string source_type;        // e.g. "web", "huggingface", "manual"
    std::string subject;            // e.g. "physics", "code", "general"
    std::string quality_tier;       // e.g. "high", "medium", "low"
    std::vector<std::string> tags;  // freeform tag list
};
```

### 5.2 Search results for dropdown preview

```cpp
struct SearchResult {
    size_t      index;              // line index (temp key)
    std::string id;                 // future: persistent ID
    std::string preview;            // truncated content for dropdown display
    float       relevance = 0.0f;   // optional ranking score
};
```

### 5.3 DatasetTarget class

```cpp
class DatasetTarget {
public:
    // model_store_root derived from repo root at startup
    explicit DatasetTarget(const std::filesystem::path& model_sto c  3 3eEre_root,
                           const std::filesystem::path& mass_dataset_path);

    // --- Model folder lifecycle ---

    std::filesystem::path createTempModelFolder();
    bool commitModelFolder(const std::filesystem::path& temp_path,
                           const std::string& model_id);
    bool discardTempModelFolder(const std::filesystem::path& temp_path);

    // --- Active model ---

    void setActiveModel(const std::string& model_id);
    std::string activeModelId() const;

    // --- Mass dataset (single source of truth) ---

    // Load all sequences into memory (called once on tab open / dataset change)
    bool loadMassDataset();
    size_t massDatasetSize() const;

    // Get sequence by index
    SequenceHandle getSequence(size_t index) const;

    // Append new entries to mass dataset (used by HF download flow)
    bool appendToMassDataset(const std::vector<std::string>& entries);

    // --- Search & Filter ---

    // String search across all sequence content.
    // Returns up to `max_results` matches with truncated previews.
    std::vector<SearchResult> searchSequences(
        const std::string& query,
        size_t max_results = 8) const;

    // Filtered search — combines string query with metadata tag filters.
    // Empty filter fields are ignored (match all).
    std::vector<SearchResult> searchSequences(
        const std::string& query,
        const std::string& source_type_filter,
        const std::string& quality_filter,
        const std::string& subject_filter,
        size_t max_results = 8) const;

    // --- Per-model sequence assignment (ID references, not copies) ---

    // Assign sequence to current model's dataset (stores ID/index reference)
    bool assignSequenceToModel(size_t seq_index);
    bool assignSequenceToModel(const std::string& seq_id);  // future: by ID

    // Remove sequence assignment from current model
    bool removeSequenceFromModel(size_t seq_index);

    // Get all sequence indices assigned to current model
    std::vector<size_t> getAssignedSequences() const;

    // Check if a sequence is assigned to current model
    bool isAssigned(size_t seq_index) const;

    // Get assigned count for UI
    size_t assignedCount() const;

    // --- Per-model assignment persistence ---
    // Stored as: model_store/<model_id>/dataset_refs.json
    // Contains: {"assigned": [0, 14, 27, 103, ...]} (indices, later IDs)

    bool loadAssignments();
    bool saveAssignments() const;

    // --- Structured output I/O ---

    // Write structured version of a sequence back to per-model structured store
    bool writeStructuredOutput(size_t seq_index, const std::string& structured);
    std::string readStructuredOutput(size_t seq_index) const;

    // --- Cleanup ---

    bool deleteSourceFiles(const std::vector<std::filesystem::path>& files);
};
```

### 5.4 Model folder lifecycle

```
User clicks "Add New Model" (in Model Panel)
  → createTempModelFolder()
  → returns model_store/.tmp_<uuid>/

User sets model name to "science-brick"
  → commitModelFolder(.tmp_<uuid>, "science-brick")
  → checks: model_store/science-brick/ does not exist
  → renames .tmp_<uuid> → science-brick/
  → returns true

User cancels / leaves name blank
  → discardTempModelFolder(.tmp_<uuid>)
  → deletes model_store/.tmp_<uuid>/
```

### 5.5 Filesystem layout

```
resources/models/model_store/
  grim-text-base/                        # router
    dataset_refs.json                    # {"assigned": [0, 3, 7, 14, ...]}
    structured/                          # per-sequence structured outputs
      qa/structured_data.jsonl
      conversation/structured_data.jsonl
  science-brick/                         # example sub-model
    dataset_refs.json                    # {"assigned": [2, 5, 19, ...]}
    structured/
      qa/structured_data.jsonl

resources/models/GRIM-text/data/
  merged_verified_cache.jsonl            # THE mass dataset (single source of truth)
```

### 5.6 ID migration path

When the sequence ID + metadata tagging plan lands:
1. `merged_verified_cache.jsonl` entries gain `"id"`, `"tags"`, `"source_type"`, `"subject"`, `"quality"` fields.
2. `dataset_refs.json` switches from `{"assigned": [index, ...]}` to `{"assigned": ["id-string", ...]}`.
3. `SequenceHandle.id` becomes the primary key; `SequenceHandle.index` becomes a secondary lookup.
4. Search filters bind to real metadata tags instead of placeholder dropdowns.
5. No UI structural changes needed — the filter dropdowns, search bar, and preview are already wired.

---

## 6. File Breakdown & Responsibilities

### UI Layer (layout + events only)

| File | Lines est. | Responsibility |
|------|-----------|----------------|
| `ui/ui_data_hub.hpp` | ~200 | Panel class: `DataHubView` enum, tab buttons, widget groups (`std::vector<std::shared_ptr<Widget>>`), all widget member variables, `setView()` show/hide. Owns `DataCollectionManager`, `HuggingFaceWebhook`, `DataStructurer`, `DatasetTarget` pointers. |
| `ui/ui_data_hub.cpp` | ~1200 | Constructor (create all widgets, assign to groups), `update()` (position + update visible group), `drawOverlay()` (draw tabs + indicator + visible group's widgets). All layout helper methods. Callbacks are lambdas that delegate to logic. |
| `ui/ui_DataCollection_OLD.hpp` | — | **Renamed reference file, not compiled.** |
| `ui/ui_DataCollection_OLD.cpp` | — | **Renamed reference file, not compiled.** |

### Logic Layer (no UI, no rendering)

| File | Change | Responsibility |
|------|--------|----------------|
| `DataCollection/dataset_target.hpp` | **NEW** | Cross-platform model folder lifecycle (temp→commit/discard), mass dataset append/read, per-model dataset I/O, source file cleanup. |
| `DataCollection/data_structurer.hpp` | **Existing** | SDG via Ollama — no changes needed. |
| `DataCollection/data_collection_manager.hpp/.cpp` | **Existing** | Collection pipeline — no changes needed. |
| `DataCollection/huggingface_webhook.hpp/.cpp` | **Minor** | After download completion, return the downloaded file paths so the caller can pass them to `DatasetTarget::deleteSourceFiles()` after appending. |

---

## 7. Implementation Phases

### Phase 1: Scaffold the tab host (no behavior change)

1. Rename `ui_DataCollection.hpp/.cpp` → `ui_DataCollection_OLD.hpp/.cpp`. Remove from CMakeLists compile targets.
2. Create `ui/ui_data_hub.hpp/.cpp` with:
   - `DataHubView` enum: `Home = 0, Sources = 1, HuggingFace = 2, Structurer = 3`
   - `std::vector<std::shared_ptr<Widget>>` per tab group: `homeWidgets_`, `sourcesWidgets_`, `hfWidgets_`, `structWidgets_`
   - Tab buttons (4x `UIButton`), active indicator underline
   - `setView()`: hide current group, show next
3. Register `UIDataHubPanel` in `ui_root.cpp` (replace the old `UIDataCollectionPanel` reference).
4. Verify: panel opens with title "DataHub", 4 tabs render, clicking tabs switches (empty content).

### Phase 2: Migrate Home HUD

1. Create all Home widgets in `UIDataHubPanel` constructor: progress bar, action buttons, config sliders, stats labels, log scrollbox.
2. Add them to `homeWidgets_`.
3. Port layout logic from `_OLD` right-column code. Wire callbacks to `DataCollectionManager`.
4. Verify: Home tab matches old panel's functionality.

### Phase 3: Migrate Sources

1. Create Sources widgets: add-source input + button, `UIScrollBox` of source entries with toggles, filter buttons, stats labels.
2. Add them to `sourcesWidgets_`.
3. Port source management callbacks from `_OLD`.
4. Verify: Sources tab shows all `source_data.json` entries, toggles work, add-source works.

### Phase 4: Migrate HuggingFace Browser

1. Create HF widgets: search input + button, category dropdown, token input, results scrollbox, queue scrollbox, queue controls.
2. Add them to `hfWidgets_`.
3. Port HF callbacks from `_OLD`.
4. Add mass-dataset append flow: on download complete → `DatasetTarget::appendToMassDataset()` → `deleteSourceFiles()`.
5. Verify: HF tab matches old panel's HF functionality + downloads merge to mass dataset.

### Phase 5: Build Dataset Structurer

1. Create `DataCollection/dataset_target.hpp` — full implementation including search, assignment, and structured I/O.
2. Create Structurer widgets:
   - **Search bar** (`UIInputBox`) with live dropdown preview (`UIScrollBox`, max ~8 results).
   - **Filter dropdowns** (Source type, Quality tier, Subject — placeholder values until tagging lands).
   - **Model dropdown** (populated from `ModelRegistry`).
   - **Format dropdown** (Q/A, Conversation, Instruct, Raw).
   - **View mode dropdown** (Dataset View, Sequence View).
   - Left `UITextArea` (raw source, editable).
   - Right `UITextArea` (structured output, editable).
   - Sequence nav buttons (prev/next, visible only in Sequence View).
   - **Assign Selected / Remove buttons** for ID-based model assignment.
   - Structure / Structure All / Save buttons.
   - Status bar labels (total, assigned, structured, failed counts).
3. Add them to `structWidgets_`.
4. Wire callbacks:
   - Search input → `DatasetTarget::searchSequences()` → populate dropdown preview.
   - Search result click → load sequence into both editor panes.
   - "Assign Selected" → `DatasetTarget::assignSequenceToModel()` (stores reference, not copy).
   - "Remove" → `DatasetTarget::removeSequenceFromModel()`.
   - Model dropdown → `DatasetTarget::setActiveModel()` → `loadAssignments()` → refresh.
   - "Structure" → `DataStructurer::structureEntry()`.
   - "Save" → `DatasetTarget::writeStructuredOutput()`.
5. Verify: can search mass dataset, see dropdown preview, assign sequences to models, structure, edit, save.

### Phase 6: Cleanup

1. Confirm all old panel functionality exists in new panel.
2. Delete `ui_DataCollection_OLD.hpp/.cpp` (or keep indefinitely as commented reference).
3. Final pass on CMakeLists, includes, ui_root references.

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single panel with widget-group visibility** | Simple show/hide. No sub-panel overhead, no sub-view classes. Matches tab UX directly. |
| **Old panel renamed, not deleted** | Kept as implementation reference during migration. Can be deleted after Phase 6 confidence. |
| **Cross-platform `std::filesystem::path` everywhere** | No `#include <Windows.h>` in dataset_target. No hardcoded paths. Model store root derived from repo root. |
| **Temp folder → commit/discard pattern** | Prevents orphan directories when user cancels model creation. Duplicate name check before rename. |
| **Mass dataset = single source of truth** | `merged_verified_cache.jsonl` is the one global pool. Per-model datasets are ID reference sets (`dataset_refs.json`), not copies. Eliminates duplication, simplifies control. |
| **ID-based referencing (framework now, IDs later)** | UI framework built around sequence IDs. Until the tagging plan lands, line-index is the temporary key. No UI changes needed when real IDs arrive. |
| **Search bar with dropdown preview** | String search over the mass dataset with truncated preview results. Most direct way to find and assign sequences without scrolling through thousands of entries. |
| **HF download → append → delete originals** | Clean pipeline: no stale HF artifacts accumulating on disk. |
| **Editable structured output** | User can fix SDG output in-place before saving, without re-running the LLM. |
| **Dataset View + Sequence View** | Full dataset view for bulk review / bulk editing. Sequence view for entry-by-entry inspection and fine-grained editing. |
| **Model dropdown = all ModelRegistry entries** | Router + sub-models all appear. Each entry maps to a reference set. Consistent with Model Panel's registry. |
| **Config persists on panel close** | Structuring config written to `ai_config.json` when the Data Hub panel is closed/hidden. No save button needed for config. |
| **Source management in its own tab** | Keeps Home focused on status/controls. Sources scrollbox gets full tab width for comfortable browsing. |

---

## 9. Config Additions

Add to `ai_config.json` under `data_collection`:

```json
{
    "data_collection": {
        "structuring": {
            "enabled": true,
            "default_mode": "qa",
            "ollama_model": "llama3.1:8b",
            "parallel_requests": 4,
            "timeout_ms": 60000,
            "max_input_chars": 3000
        }
    }
}
```

These map 1:1 to the existing `DataStructuringConfig` struct. Persisted on panel close.

---

## 10. MMO Filesystem Integration

The mass dataset (`merged_verified_cache.jsonl`) is the **single source of truth** for all sequence content. Per-model datasets are **ID reference sets** (`dataset_refs.json`) pointing into the mass dataset — no data duplication.

**`DatasetTarget` responsibilities**:
- Mass dataset load, search, append for the shared pipeline.
- Per-model sequence assignment via ID references.
- Per-model structured output storage.
- Cross-platform model folder lifecycle (temp → commit/discard).
- Source file cleanup after HF download merges.

**Data flow — no duplication**:

```
merged_verified_cache.jsonl   (global pool — ALL sequences live here)
         │
         ├──▷ model_store/grim-text-base/dataset_refs.json   → {assigned: [0,3,7,14,…]}
         ├──▷ model_store/science-brick/dataset_refs.json     → {assigned: [2,5,19,…]}
         └──▷ model_store/code-brick/dataset_refs.json        → {assigned: [1,8,22,…]}

No sequence is copied. Only IDs (line indices → later persistent IDs) are stored per model.
Structured outputs are stored per-model under structured/<format>/.
```

**Workflow**:
1. Collect data via Home tab (web crawl pipeline → `merged_verified_cache.jsonl`).
2. Download HF datasets via HuggingFace tab → append to mass dataset → delete originals.
3. Open Structurer tab → search the mass dataset, filter by tags/metadata.
4. Select a model in the dropdown.
5. Assign sequences to that model (ID reference, not copy).
6. Structure assigned entries (SDG via Ollama).
7. Edit structured output if needed.
8. Save → structured output written to the model's `structured/` directory.
9. Training pipeline reads the model's `dataset_refs.json` + resolves content from mass dataset.

**ID migration**: when the sequence ID + metadata tagging plan lands, `dataset_refs.json` transitions from line-index arrays to persistent ID arrays. The UI framework (search bar, filter dropdowns, assign/remove buttons) requires zero structural changes.
