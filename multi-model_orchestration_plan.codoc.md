---
name: MMO Refactor Companion
overview: Living companion to multimodelorchestrationintegration.md. Tracks implemented refactors, ownership boundaries, and canonical hook points. Must be updated whenever a system refactor changes responsibilities or integration points.
todos: []
isProject: false
---

# MMO Refactor Companion

> This is **not** the full MMO architecture map.
> It is the **living refactor ledger** for the MMO plan: what has been changed already, what now owns each concern, and where future systems are allowed to hook in.

## Mandatory maintenance rule

Every time an agent refactors, replaces, splits, renames, or deletes a system that affects MMO architecture or modular boundaries, this file must be updated in the **same change**.

Minimum required update each time:
- what changed
- what now owns the concern
- allowed integration points / hook points
- migrated consumers
- removed or deleted legacy paths
- validation status / known remaining gaps

If a refactor changes code but not this document, the refactor is incomplete.

## Current implemented refactors

### 1. Static hardware inventory

**Owner**
- `MMO/Core/HardwareInventory.hpp`
- `MMO/Core/HardwareInventory.cpp`

**Responsibility**
- Immutable machine/topology snapshot captured once during bootstrap
- OS, CPU, RAM, GPU inventory, monitor topology, audio devices, Wi-Fi presence, voice backend availability, suggested Whisper model

**Does not own**
- Live CPU/RAM/GPU usage
- Admission policy
- Geolocation state

**Canonical hook points**
- Built by `GRIM::MMO::detectHardware()`
- Logged by `GRIM::MMO::logHardwareInventory(...)`
- Stored globally as `g_hardwareInventory`
- Initialized in `bootstrap/bootstrap.cpp`

**Current consumers migrated**
- `bootstrap/bootstrap.cpp`
- `main.cpp`
- `commands/commands_ai.cpp`
- `commands/commands_system.cpp`
- `core/plugin_api_impl.cpp`
- `core/grim_exports.hpp`
- `hardware/resource_values.cpp`
- `ui/ui_root.cpp`
- `ui/ui_training_panel.cpp`
- `perception/multi_monitor.hpp`
- `perception/multi_monitor.cpp`

**Legacy replaced**
- Old `SystemInfo` ownership model
- Old `g_systemInfo` global

**Status**
- Implemented

### 2. Location and network identity

**Owner**
- `location.hpp`
- `location.cpp`

**Responsibility**
- `LocationInfo`
- `g_location`
- `g_wifiConnected`
- `detectWifiConnected()`
- `fetchLocationByIP(...)`

**Does not own**
- Hardware topology
- Resource pressure sampling
- Model loading policy

**Canonical hook points**
- Populated in `bootstrap/bootstrap.cpp`
- Read by location-aware prompt / question systems
- Exported through `core/grim_exports.hpp`

**Current consumers**
- `ai/ai.cpp`
- `commands/commands_question.cpp`
- `bootstrap/bootstrap.cpp`

**Legacy replaced**
- Location state previously embedded inside `system_detect.*`

**Status**
- Implemented

### 3. Live resource signal

**Owner**
- `MMO/Core/ResourceSignal.hpp`
- `MMO/Core/ResourceSignal.cpp`

**Responsibility**
- Background sampler thread for live CPU/RAM/GPU usage
- Publishes latest `ResourceSnapshot`
- Derives shared `PressureState`

**Does not own**
- Static machine inventory
- Admission decisions
- Model lifecycle policy

**Canonical hook points**
- Created as `g_resourceSignal`
- Started in `bootstrap/bootstrap.cpp`
- Consumers must read snapshots from `latest()` instead of probing hardware ad hoc

**Expected consumers**
- `ModelLoader` (future)
- tool/process loaders (future)
- perception scheduling (future)

**Status**
- Implemented core service; wider consumer integration still pending

### 4. Shared resource admission authority

**Owner**
- `MMO/Core/ResourceCoordinator.hpp`
- `MMO/Core/ResourceCoordinator.cpp`

**Responsibility**
- Accept `ResourceClaim`
- Return `ResourceDecision`
- Track active holders
- Enforce reserve thresholds and eviction/defer/throttle decisions

**Does not own**
- Hardware detection
- Live sampling
- Actual model/tool startup logic

**Canonical hook points**
- Created as `g_resourceCoordinator`
- Initialized in `bootstrap/bootstrap.cpp`
- Future resource consumers must submit claims here instead of rolling their own checks

**Expected consumers**
- `ModelLoader` (future)
- model process manager (future)
- plugin/tool loader (future)
- perception jobs (future)

**Fixes applied**
- `signal_.snapshot()` → `signal_.latest()` (matched `ResourceSignal` API)
- `snap.free_ram_mb` → `snap.ram_available_mb` (matched `ResourceSnapshot` field name)
- `g.total_vram_mb - g.used_vram_mb` → `g.vram_free_mb` (matched `GPULiveState` field name)

**Status**
- Implemented core authority; API contract validated against `ResourceSignal` / `ResourceSnapshot` / `GPULiveState`; main downstream integrations still pending

### 5. MMO model contract cleanup

**Owner**
- `MMO/Shared/MMD.hpp`

**Responsibility**
- `BackendType`
- `ModelInfo`
- `RequestEnvelope`
- `ResponseEnvelope`

**Does not own**
- Registry loading
- Route/synthesize validation logic
- Runtime orchestration

**Canonical hook points**
- Consumed by future `ModelRegistry`, router adapters, orchestrator, and process manager
- Router-only personalization fields remain `lora_path` and `hard_copy_path`

**Status**
- Contract cleanup implemented; now consumed by `ModelRegistry`

### 6. Model registry

**Owner**
- `MMO/Core/ModelRegistry.hpp`
- `MMO/Core/ModelRegistry.cpp`

**Responsibility**
- Canonical registry of all known models (router + sub-models)
- Loads and validates model entries from `ai_config.json` → `mmo` section
- Enforces invariants: exactly one router, unique IDs, sub-models must not have `lora_path` / `hard_copy_path`
- Provides lookup by id, by subject tag, router accessor, sub-model list
- Exposes MMO enabled/mode state

**Does not own**
- Model process lifecycle (future `ModelLoader`)
- Runtime orchestration (future `Orchestrator`)
- Resource claims for model loading (goes through `ResourceCoordinator`)
- Server process management (currently `GRIMTextServerManager`, future model-keyed process manager)

**Canonical hook points**
- `ModelRegistry::instance()` — singleton
- `ModelRegistry::loadFromConfig(aiConfig)` — called during bootstrap after config is loaded
- `ModelRegistry::getRouter()` — returns the router `ModelInfo`
- `ModelRegistry::getModelById(id)` — lookup any model
- `ModelRegistry::getSubModels()` — all non-router models
- `ModelRegistry::getModelsBySubjectTag(tag)` — subject-based lookup for routing
- `ModelRegistry::isEnabled()` / `ModelRegistry::mode()` — MMO state

**Config contract**
- `ai_config.json` → `mmo.enabled`, `mmo.mode`, `mmo.router` (object), `mmo.sub_models` (array)
- Each model entry: `id`, `name`, `version`, `subject`, `description`, `model_path`, `backend_type`, `url`, `subject_tags`, `usage_weight`, `lora_path` (router only), `hard_copy_path` (router only), `estimated_ram_mb`, `estimated_vram_mb`

**Expected consumers**
- `ModelLoader` — queries registry for model info before submitting resource claims
- `Orchestrator` (future) — will query registry for routing decisions
- `ModelRouter` (future) — will use subject tag lookups
- bootstrap — will call `loadFromConfig()` after config init

**Legacy replaced**
- None (new system)

**Status**
- Implemented; bootstrap integration pending

### 7. ModelLoader (`MMO/Core/ModelLoader.hpp`, `MMO/Core/ModelLoader.cpp`)

**What changed**
- New system implemented from scratch

**Owns**
- Model lifecycle state machine: `Unloaded → Loading → Loaded → InUse → Idle → EvictEligible → Unloading → Unloaded`
- Per-model runtime state tracking (`ModelSlot`: state, `use_count`, GPU device, timestamps)
- Use-degrading TTL logic: `effective_ttl = min(idle_ttl_ms + use_count * step, hot_ttl_cap_ms)`
- Load/unload orchestration through `StartCallback`/`StopCallback`
- Idle → EvictEligible transition via `tickIdleTimers()`
- Eviction of targets returned by `ResourceCoordinator`

**Does not own**
- Resource admission decisions (delegated to `ResourceCoordinator.requestClaim()`)
- Actual process management (pluggable via `StartCallback`/`StopCallback`)
- Model metadata (queries `ModelRegistry`)
- Hardware probing (never touches hardware directly)
- Request routing (future `Orchestrator`)

**Canonical hook points**
- `ModelLoader::ensureLoaded(model_id)` — main entry, returns `LoadResult`
- `ModelLoader::markInUse(model_id)` / `markIdle(model_id)` — request bracket
- `ModelLoader::tickIdleTimers()` — call periodically from maintenance loop
- `ModelLoader::unload(model_id)` / `unloadAll()` — teardown
- `ModelLoader::setStartCallback(cb)` / `setStopCallback(cb)` — plug in process management
- `ModelLoader::getState(model_id)` / `getSlot(model_id)` — read-only queries

**Config contract**
- `ModelLoaderConfig` struct: `load_timeout_ms` (30000), `idle_ttl_ms` (60000), `hot_ttl_cap_ms` (300000), `use_degrade_step_ms` (5000)
- Future: loaded from `ai_config.json` → `mmo.model_loader`

**Expected consumers**
- `Orchestrator` (future) — will call `ensureLoaded` + `markInUse`/`markIdle` around request dispatch
- Model-keyed process manager (future) — will provide `StartCallback`/`StopCallback`
- Shutdown/cleanup path — will call `unloadAll()`

**Integration with existing systems**
- `ModelRegistry` — read-only dependency, queries model info for resource estimates
- `ResourceCoordinator` — submits claims (`ModelLoad` kind for load, upgrades to `ModelResident` after success), calls `markInUse(consumer_id, bool)`, releases claims on unload
- Consumer ID format: `"model:" + model_id`

**Legacy replaced**
- None (new system)

**Status**
- Implemented; bootstrap integration and process callback wiring pending

## Separation-of-concerns summary

| Concern | Canonical owner | Integration point |
|---|---|---|
| Static machine topology | `HardwareInventory` | `detectHardware()` -> `g_hardwareInventory` |
| Location / Wi-Fi / GeoIP | `location.*` | bootstrap populates `g_location` / `g_wifiConnected` |
| Live resource usage | `ResourceSignal` | `g_resourceSignal->latest()` |
| Resource admission / eviction | `ResourceCoordinator` | `requestClaim(...)` / `releaseClaim(...)` |
| Model metadata envelopes | `MMO/Shared/MMD.hpp` | shared by registry / router / orchestrator |
| Model registry | `ModelRegistry` | `ModelRegistry::instance()` / `loadFromConfig()` |
| Model lifecycle / residency | `ModelLoader` | `ensureLoaded()` / `markInUse()` / `markIdle()` / `tickIdleTimers()` |

## Legacy paths removed

- `system_detect.hpp`
- `system_detect.cpp`
- old `g_systemInfo` ownership path

## Next systems to document when touched

Add an entry here when any of these are refactored:
- `Orchestrator`
- `MemoryFacade`
- session-scoped context authority
- `ToolRegistry`
- `ActionPolicyRegistry`
- UI surface registry / emotion presentation controller
- model-keyed process manager

## Required update checklist for future agents

When refactoring a system, append or update the relevant section in this file with:
1. owning files
2. exact responsibility boundary
3. explicit non-responsibilities
4. canonical integration points
5. migrated consumers
6. deleted legacy paths
7. validation performed
