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

**Current consumers**
- `ResourceCoordinator` — reads snapshots for admission decisions
- `ModelLoader` (transitive via `ResourceCoordinator`)
- `main.cpp` shutdown handlers — calls `stop()` + delete

**Status**
- Implemented; consumer integration complete

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

**Current consumers**
- `ModelLoader` — submits `ModelLoad` / `ModelResident` claims, calls `markInUse()`, `releaseClaim()`
- `main.cpp` shutdown handlers — deletes coordinator during teardown

**Status**
- Implemented; API contract validated; downstream `ModelLoader` integration complete

**Fixes applied**
- `signal_.snapshot()` → `signal_.latest()` (matched `ResourceSignal` API)
- `snap.free_ram_mb` → `snap.ram_available_mb` (matched `ResourceSnapshot` field name)
- `g.total_vram_mb - g.used_vram_mb` → `g.vram_free_mb` (matched `GPULiveState` field name)

**Status**
- Implemented; API contract validated; downstream integrations complete

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

**Current consumers**
- `ModelLoader` — queries registry for model info before submitting resource claims
- `Orchestrator` — queries registry for routing decisions and sub-model validation
- `bootstrap/bootstrap.cpp` — calls `loadFromConfig(aiConfig)` during init
- `ai/ai.cpp` — checks `isEnabled()` / `mode()` in `callAIAsync` MMO branch

**Legacy replaced**
- None (new system)

**Status**
- Implemented; bootstrap integration complete

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

**Current consumers**
- `Orchestrator` — calls `ensureLoaded` + `markInUse`/`markIdle` around request dispatch
- `bootstrap/bootstrap.cpp` — creates loader, wires `StartCallback` to `GRIMTextServerManager::start()` and `StopCallback` to `stopGRIMTextServer()`, spawns background idle-tick thread
- `main.cpp` shutdown handlers — calls `unloadAll()` + delete during teardown

**Integration with existing systems**
- `ModelRegistry` — read-only dependency, queries model info for resource estimates
- `ResourceCoordinator` — submits claims (`ModelLoad` kind for load, upgrades to `ModelResident` after success), calls `markInUse(consumer_id, bool)`, releases claims on unload
- Consumer ID format: `"model:" + model_id`

**Legacy replaced**
- None (new system)

**Status**
- Implemented; bootstrap integration complete; process callbacks wired; idle tick thread running

### 8. Orchestrator (`MMO/Core/Orchestrator.hpp`, `MMO/Core/Orchestrator.cpp`)

**What changed**
- New system implemented from scratch

**Owns**
- Full orchestration flow: ensure router → route → ensure sub-model → generate → synthesize → return
- Envelope construction for route, generate, and synthesize steps (`RequestEnvelope`)
- HTTP dispatch to model backends via cpr (`callBackend()`)
- Response parsing and RouteDecision extraction from router output
- Boundary validation: schema version, request_id correlation, target_model_id correlation at every step
- Error aggregation into `OrchestratorResult`

**Does not own**
- Model lifecycle (delegated to `ModelLoader`)
- Model metadata (queries `ModelRegistry`)
- Resource admission (transitive through `ModelLoader` → `ResourceCoordinator`)
- Memory retrieval / context composition (future `MemoryFacade` + `SessionContextManager`)
- NLP annotation (future — orchestrator will call NLP pipeline before route)
- Action policy / Training Wheels gate (future — post-synthesis)
- Conversation history management (body-side, not orchestrator's concern)

**Canonical hook points**
- `Orchestrator::generate(OrchestratorRequest)` — main entry, returns `OrchestratorResult`
- `Orchestrator::shutdown()` — calls `loader_.unloadAll()`

**Transport contracts**
- Route request: `task="route"`, `payload=prompt`, `scope=metadata_json`
- Generate request: `task="generate"`, `payload=composed_generation` (sub-model sees ONLY this)
- Synthesize request: `task="synthesize"`, `payload=sub_model_result`
- Router result JSON: `{ "sub_model_id", "composed_generation", "diagnostics" }`
- All responses: `{ "schema_version", "request_id", "target_model_id", "status", "result"|"refusal"|"error" }`

**Config contract**
- `OrchestratorConfig` struct: `route_timeout_ms` (10000), `generate_timeout_ms` (30000), `synthesize_timeout_ms` (10000), `max_submodels_per_request` (1)
- Future: loaded from `ai_config.json` → `mmo.orchestrator`

**Current consumers**
- `callAIAsync()` in `ai/ai.cpp` — MMO branch at top of async lambda; routes through `g_orchestrator->generate()` when registry is enabled, falls back to legacy `resolveBackendURL()` in shadow mode on failure, returns error string in enforced mode on failure
- `bootstrap/bootstrap.cpp` — constructs orchestrator after registry and loader, wires process callbacks
- `main.cpp` — `consoleHandler` (Win32) and `signalHandler` (Unix) call `g_orchestrator->shutdown()` + delete during teardown

**Integration with existing systems**
- `ModelRegistry` — read-only dependency for router and sub-model info
- `ModelLoader` — `ensureLoaded()` / `markInUse()` / `markIdle()` bracket around each backend call
- `MMD.hpp` — uses `RequestEnvelope`, `ResponseEnvelope`, `ResponseStatus`, `ModelInfo`, `BackendType`
- HTTP via cpr — same library used by existing `callAIAsync()` and `GRIMBackend`

**Legacy replaced**
- Direct `resolveBackendURL()` call in `callAIAsync()` is bypassed when MMO is enabled (shadow mode preserves fallback; enforced mode does not)

**Status**
- Implemented; bootstrap wiring complete; `callAIAsync` MMO branch active; shutdown handlers wired; NLP/memory context wiring pending

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
| Request orchestration | `Orchestrator` | `generate(OrchestratorRequest)` |
| Memory retrieval / write surface | `MemoryFacade` | `MemoryFacade::instance()` |
| Session interaction state | `SessionContextManager` | `SessionContextManager::instance()` |
| Unified tool registry | `ToolRegistry` | `ToolRegistry::instance()` |
| Action policy / Training Wheels | `ActionPolicyRegistry` | `ActionPolicyRegistry::instance().evaluate()` |
| Bootstrap wiring | `bootstrap/bootstrap.cpp` | globals + init block + idle tick thread |
| AI dispatch replacement | `ai/ai.cpp` | MMO branch in `callAIAsync()` + policy gate in `ai_interpret()` |
| Shutdown teardown | `main.cpp` | `consoleHandler` / `signalHandler` — top-down delete |

## Legacy paths removed

- `system_detect.hpp`
- `system_detect.cpp`
- old `g_systemInfo` ownership path
- `CommandRegistry::generateCompactPrompt()` call in `ai_interpret` replaced by `ToolRegistry::instance().generateCompactPrompt()`
- `CommandRegistry::recordSuccess/recordFailure` in `ai_interpret` replaced by `ToolRegistry` + `ActionPolicyRegistry` analytics
- `CommandRegistry::recordSuccess/recordFailure` in `commands_core.cpp` `dispatchCommand()` → `ToolRegistry::instance().recordSuccess/recordFailure()`
- `CommandRegistry::getAllTools/getToolsByCategory/getTool/getUsageStats` in `commands_system.cpp` → `ToolRegistry::instance()`
- `CommandRegistry::getAllTools/getCategories/getToolsByCategory` in `commands_question.cpp` → `ToolRegistry::instance()`
- `CommandRegistry` include removed from `ai/ai.cpp` (dead include)
- `command_registry.hpp` dead includes removed from `commands_core.cpp`, `commands_system.cpp`, `commands_question.cpp`
- `context_manager.hpp` dead includes removed from `ai_reward.hpp`, `ai_rl.cpp`, `plugin_api_impl.cpp`, `main.cpp`, `MemoryFacade.hpp`
- `ContextManager::setMemoryStorage()` dead call removed from `main.cpp`
- All `GRIM::Feedback::` globals deleted from `commands_feedback.cpp`
- All `GRIM::ContextManager::` static method calls replaced with `SessionContextManager` in `commands_core.cpp`, `commands_feedback.cpp`, `MemoryFacade.cpp`, `proactive_dialogue.cpp`
- `memory/context_manager.hpp` and `memory/context_manager.cpp` deleted — `ContextSnapshot` struct extracted to `memory/context_snapshot.hpp`

## Legacy paths fully migrated

### Feedback system (`commands/commands_feedback.cpp`) — COMPLETED
- All 4 static globals deleted: `g_pendingClarifyCmd`, `g_pendingFeedbackCmd`, `g_isMultiCommandContext`, `g_isVoiceCommand`
- All 12 getter/setter functions now delegate to `SessionContextManager::instance()` via `kDefaultSession`
- `processClarificationResponse` / `processFeedbackResponse` complex state machines preserved — internal calls migrated to SCM
- 2 `ContextManager::recordUsage()` calls → `SCM::recordUsage()`
- `commands_feedback.hpp` declarations unchanged (implementations now delegate to SCM)

### ContextManager statics (`memory/context_manager.hpp`) — COMPLETED
- **commands_core.cpp**: All 30+ call sites migrated — `setPendingIntent`/`getPendingIntent`/`clearPendingIntent` → `SCM::setPending/getPending/clearPending` with `PendingKind`; `getSnapshot` → `SCM::legacySnapshot`; `rememberContextObject`/`decayOldContext`/`recordUsage`/`getCurrentMood` → SCM equivalents
- **MemoryFacade.cpp**: All 5 bridge calls migrated — `getSnapshot` → `SCM::legacySnapshot`, `rememberContextObject` → SCM, `decayOldContext` → SCM
- **main.cpp**: `ContextManager::setMemoryStorage()` call removed (dead — only used by never-called `attachFeedbackMetadata`)
- **proactive_dialogue.cpp**: `ContextManager::usageCount("system")` → `SCM::usageCount("default", "system")`
- Dead includes removed: `ai_reward.hpp`, `ai_rl.cpp`, `plugin_api_impl.cpp`
- `ContextSnapshot` struct extracted to `memory/context_snapshot.hpp`; `context_manager.hpp` and `context_manager.cpp` deleted
- `fast_classifier.cpp`, `intent_gate.cpp`, `SessionContextManager.hpp` now include `memory/context_snapshot.hpp`

### CommandRegistry residual — COMPLETED
- Dead includes removed from `commands_core.cpp`, `commands_system.cpp`, `commands_question.cpp`
- `commands_question.cpp` and `commands_system.cpp` fully migrated to `ToolRegistry::instance()`
- Only remaining code usage: `bootstrap.cpp` line 241 — `CommandRegistry::getAllTools()` at startup to seed ToolRegistry (intentional)

### 9. MemoryFacade (`MMO/Core/MemoryFacade.hpp`, `MMO/Core/MemoryFacade.cpp`)

**What changed**
- New system implemented

**Owns**
- Unified retrieval over split memory subsystem (short-term + long-term + context)
- `MemoryRetrievalResult` projection: relevant memories, context breadcrumbs, timestamps
- Write surface for storing interactions, long-term facts, short-term buffers
- Context snapshot access, decay, flush, compact operations
- Search by tag / multi-tag / free-text

**Does not own**
- Actual memory storage (delegated to `UnifiedMemoryStorage`)
- Session interaction state (moved to `SessionContextManager`)
- Conversation history management (body-side concern)
- Memory persistence format (FlatBuffer schema in `UnifiedMemoryStorage`)

**Canonical hook points**
- `MemoryFacade::instance()` — singleton
- `MemoryFacade::retrieveForPrompt(query, max_results)` — main retrieval for prompt composition
- `MemoryFacade::search(query)` — free-text search
- `MemoryFacade::getByTag(tag)` / `getByTags(tags)` — structured lookup
- `MemoryFacade::recordInteraction(role, content)` — log interaction
- `MemoryFacade::storeLongTerm(key, value)` / `storeShortTerm(key, value)` — write paths
- `MemoryFacade::getContextSnapshot()` / `rememberContext()` / `decayContext()` — context lifecycle

**Config contract**
- None (uses `UnifiedMemoryStorage` config)

**Current consumers**
- `bootstrap/bootstrap.cpp` — owns `g_memoryFacade` pointer
- `main.cpp` — deletes during shutdown
- Future: `Orchestrator` context composition, `SessionContextManager` context enrichment

**Legacy replaced**
- `ContextManager::getSnapshot()` bridge call → `SessionContextManager::legacySnapshot()`
- `ContextManager::rememberContextObject()` bridge call → `SessionContextManager::rememberContextObject()`
- `ContextManager::decayOldContext()` bridge call → `SessionContextManager::decayOldContext()`
- `context_manager.hpp` include → `SessionContextManager.hpp`

**Status**
- Implemented; bootstrap wiring complete; all context bridge calls migrated to SessionContextManager

### 10. SessionContextManager (`MMO/Core/SessionContextManager.hpp`, `MMO/Core/SessionContextManager.cpp`)

**What changed**
- New system implemented

**Owns**
- Session-scoped interaction state: turn records, referent bindings, pending interactions, action episodes
- Append-only `TurnRecord` log with NLP summary and outcome tracking
- `ReferentBinding` — entity tracking for pronoun/deictic resolution ("it", "that file", "the app")
- `PendingInteraction` — unified pending state replacing fragmented globals
  - `PendingKind`: MissingSlot, Clarification, Confirmation, Correction, FollowUp
- `ActionEpisode` — proposal → user response → execution result lifecycle
- `VisualContext` — separate `DigitalVisual` (active window/selection) + `PhysicalVisual` (camera/gesture/gaze)
- `ContextSnapshotV2` — rich projection for router/classifier/policy consumers
- Session lifecycle: `beginTurn()` / `setNlpSummary()` / `setTurnTags()` / `recordOutcome()` / `tick()` / `destroySession()`

**Does not own**
- Memory persistence (delegated to `MemoryFacade`)
- Conversation history for model prompts (body-side concern in `ai.cpp`)
- Action policy evaluation (delegated to `ActionPolicyRegistry`)
- Tool metadata (delegated to `ToolRegistry`)

**Canonical hook points**
- `SessionContextManager::instance()` — singleton
- `beginTurn(session_id)` → returns `turn_id` string
- `setNlpSummary(session_id, turn_id, summary)` — attach NLP parse result to turn
- `setTurnTags(session_id, turn_id, tags)` — attach classification tags
- `recordOutcome(session_id, turn_id, outcome)` — mark how turn ended
- `addReferent(session_id, referent)` / `resolveReference(session_id, phrase)` — entity tracking
- `setPending(session_id, interaction)` / `getPending(session_id)` / `clearPending(session_id)` — pending state
- `recordProposal(session_id, episode)` / `recordUserResponse(session_id, turn_id, accepted, correction)` / `recordExecutionResult(session_id, turn_id, success, output)` — action lifecycle
- `updateDigitalVisual(session_id, visual)` / `updatePhysicalVisual(session_id, visual)` — visual context
- `snapshot(session_id)` → `ContextSnapshotV2` — full session projection
- `tick(session_id)` — expire stale referents and pending interactions

**Replaces**
- `memory/context_manager.hpp` static singletons: `recentContext`, `pendingIntent`, `ContextSnapshot`
- `commands/commands_feedback.cpp` globals: `g_pendingClarifyCmd`, `g_pendingFeedbackCmd`, `g_isMultiCommandContext`, `g_isVoiceCommand`
- Fragmented clarification/confirmation state across free functions
- `ContextManager::getSnapshot()` / `rememberContextObject()` / `decayOldContext()` / `recordUsage()` / `getCurrentMood()` / `usageCount()` in all consumers
- `ContextManager::setMemoryStorage()` (dead path removed)

**Migration status**
- SessionContextManager implemented and bootstrapped
- `ai_interpret()` in `ai/ai.cpp` sets `PendingInteraction` when `ActionPolicyRegistry` requires verification
- ALL legacy callers migrated:
  - `commands_core.cpp` — 30+ call sites rewritten to SCM
  - `commands_feedback.cpp` — 4 statics deleted, 12 functions delegate to SCM
  - `MemoryFacade.cpp` — 5 bridge calls replaced
  - `main.cpp` — dead `setMemoryStorage` call removed
  - `proactive_dialogue.cpp` — `usageCount` migrated
  - `wake_voice.cpp` — uses wrapper functions from `commands_core.hpp` (no direct API calls)
- New methods added this migration: `setVoiceCommand/isVoiceCommand`, `setMultiCommandContext/isMultiCommandContext`, `recordUsage/usageCount`, `rememberContextObject`, `recallContextByType`, `recallContextByIntent`, `decayOldContext`, `legacySnapshot()`
- `legacySnapshot()` projects V2 session state to V1 `ContextSnapshot` for IntentGate/FastClassifier compatibility

**Status**
- Implemented; fully integrated; all legacy caller migrations complete

### 11. ToolRegistry (`MMO/Core/ToolRegistry.hpp`, `MMO/Core/ToolRegistry.cpp`)

**What changed**
- New system implemented

**Owns**
- Canonical registry of all tools (built-in commands + plugin commands)
- `ToolDescriptor` record: identity, documentation, policy flags, parameters, preconditions, context requirements, affordance tokens, aliases, keywords, capability tags, hot-swap state, usage stats
- Hot-swap lifecycle: `ToolSwapState` (Loaded/Loading/Unloading/Unavailable)
- Atomic batch registration/unregistration for plugin hot-swap
- Alias → tool_id resolution
- Model-facing prompt generation (compact and full)
- Prompt cache version counter (incremented on mutation)
- Usage analytics (success/failure recording)
- `ToolRegistryListener` observer interface for registry changes

**Does not own**
- Command execution (dispatched by `dispatchCommand()` in `commands_core.cpp`)
- Action policy evaluation (delegated to `ActionPolicyRegistry`)
- Plugin lifecycle management (future plugin manager)
- Session context (delegated to `SessionContextManager`)

**Canonical hook points**
- `ToolRegistry::instance()` — singleton
- `registerTool(descriptor)` / `registerSimple(tool_id, desc, cat)` — registration
- `unregisterTool(tool_id)` — removal
- `registerBatch(plugin_name, tools)` / `unregisterByProvider(plugin_name)` — atomic plugin operations
- `getTool(tool_id)` / `getAllTools()` / `getByCategory(cat)` / `getByCapabilityTag(tag)` — queries
- `resolveAlias(alias)` — alias resolution
- `generateCompactPrompt()` / `generateFullPrompt()` — model-facing prompt text
- `version()` — monotonic prompt invalidation counter
- `recordSuccess(tool_id)` / `recordFailure(tool_id)` — analytics
- `addListener(listener)` / `removeListener(listener)` — observer pattern

**Config contract**
- No dedicated config — populated at bootstrap from `CommandRegistry::getAllTools()`
- Future: plugin-contributed tools registered via `registerBatch()`

**Current consumers**
- `bootstrap/bootstrap.cpp` — imports all `CommandRegistry` tools into `ToolRegistry` at startup
- `ai/ai.cpp` — `ai_interpret()` uses `generateCompactPrompt()` for model context, `recordSuccess/recordFailure` for analytics
- `ActionPolicyRegistry` — queries `getTool()` for risk derivation and tool existence checks
- Future: `Orchestrator` context composition, plugin hot-swap manager

**Replaces**
- `GRIM::CommandRegistry::generateCompactPrompt()` call in `ai_interpret()`
- `GRIM::CommandRegistry::recordSuccess/recordFailure` in `ai_interpret()`
- `CommandRegistry::getAllTools/getToolsByCategory/getTool/getUsageStats` in `commands_system.cpp`
- `CommandRegistry::getAllTools/getCategories/getToolsByCategory` in `commands_question.cpp`
- `CommandRegistry::recordSuccess/recordFailure` in `commands_core.cpp` `dispatchCommand()`
- Dead `command_registry.hpp` includes removed from `commands_core.cpp`, `commands_system.cpp`, `commands_question.cpp`, `ai/ai.cpp`
- (CommandRegistry class still exists — `bootstrap.cpp` calls `getAllTools()` once at startup to seed ToolRegistry)

**Status**
- Implemented; bootstrap import wired; all consumer migrations complete; only bootstrap seed import remains

### 12. ActionPolicyRegistry (`MMO/Core/ActionPolicyRegistry.hpp`, `MMO/Core/ActionPolicyRegistry.cpp`)

**What changed**
- New system implemented — the Training Wheels unified verification gate

**Owns**
- Action policy evaluation: `evaluate(ActionProposal)` → `PolicyDecision`
- Risk computation: per-tool `ToolPolicyOverride` risk category + `ToolDescriptor` metadata fallback
- `RiskCategory` classification: Safe / Low / Medium / High / Critical
- Confidence computation: `GC_action_confidence = min(body_confidence, calibrated_router_confidence)`
  - Body confidence = min of all body-side signals (parse_certainty, memory_match, referent_resolution, grounding_coverage, tool_preconditions, historical_success)
  - Router confidence calibrated via `CalibrationBucket` observed success rates
- `PolicyVerdict`: Allow / VerifyRisk / VerifyConfidence / VerifyBoth / Deny
- Verification prompt generation for user confirmation
- Per-tool policy overrides: block, always_confirm, risk category override, sandbox mode
- Per-category risk thresholds
- RL-style calibration: `recordOutcome(tool_id, router_conf, success)` updates success rate buckets
- Warm-up rule: tools with < `calibration_min_samples` outcomes use `uncalibrated_router_conf`

**Does not own**
- Tool metadata (queries `ToolRegistry`)
- Session state / pending interaction storage (uses `SessionContextManager` when verification needed)
- Command dispatch / execution
- Memory retrieval
- NLP parsing / classification

**Canonical hook points**
- `ActionPolicyRegistry::instance()` — singleton
- `configure(ActionPolicyConfig)` — configure thresholds, floors, calibration params
- `setToolPolicy(ToolPolicyOverride)` / `removeToolPolicy(tool_id)` / `getToolPolicy(tool_id)` — per-tool overrides
- `evaluate(ActionProposal)` → `PolicyDecision` — the single gate
- `recordOutcome(tool_id, router_confidence, success)` — RL calibration feedback
- `getCalibratedConfidence(tool_id, raw_router_confidence)` — read calibrated confidence
- `isBlocked(tool_id)` / `getRiskCategory(tool_id)` / `getThreshold(category)` — queries
- `config()` — read current config

**Config contract**
- `ActionPolicyConfig` struct: `enabled` (true), `risk_threshold` (0.6f), `min_confidence_floor` (0.3f), `uncalibrated_router_conf` (0.5f), `calibration_min_samples` (10), `per_category_thresholds` map
- Loaded from `ai_config.json` → `training_wheels` section in `bootstrap/bootstrap.cpp`
- Per-tool overrides: `ToolPolicyOverride` { `tool_id`, `risk_category`, `blocked`, `always_confirm`, `sandbox` }

**Current consumers**
- `bootstrap/bootstrap.cpp` — configures from `ai_config.json` → `training_wheels`
- `ai/ai.cpp` — `ai_interpret()` gates every AI-suggested command through `evaluate()` before dispatch
  - `Deny` → returns error to user
  - `Verify*` → sets `PendingInteraction` via `SessionContextManager`, returns verification prompt
  - `Allow` → dispatches command, records outcome for calibration
- `ToolRegistry` — transitive: `ActionPolicyRegistry` queries `ToolRegistry` for risk derivation

**Replaces**
- Direct `ActionExecutor::executeAction()` / `dispatchCommand()` in `ai_interpret()` now gated
- No previous action policy existed — this is the first unified gate

**Status**
- Implemented; bootstrap configuration wired; `ai_interpret()` gate active; calibration feedback loop connected

## Next systems to document when touched

Add an entry here when any of these are refactored:
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
