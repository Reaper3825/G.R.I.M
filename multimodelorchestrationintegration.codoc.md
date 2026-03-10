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
- Loaded from `ai_config.json` → `mmo.model_loader` in `bootstrapMMOLayer()` (see §26)

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
- Loaded from `ai_config.json` → `mmo.orchestrator` in `bootstrapMMOLayer()` (see §26)

**Current consumers**
- `callAIAsync()` in `ai/ai.cpp` — sole dispatch path; all AI calls route through `g_orchestrator->generate()`, throws on null orchestrator
- `bootstrap/bootstrap.cpp` — constructs orchestrator after registry and loader, wires process callbacks
- `main.cpp` — `consoleHandler` (Win32) and `signalHandler` (Unix) call `g_orchestrator->shutdown()` + delete during teardown

**Integration with existing systems**
- `ModelRegistry` — read-only dependency for router and sub-model info
- `ModelLoader` — `ensureLoaded()` / `markInUse()` / `markIdle()` bracket around each backend call
- `MMD.hpp` — uses `RequestEnvelope`, `ResponseEnvelope`, `ResponseStatus`, `ModelInfo`, `BackendType`
- HTTP via cpr — same library used by existing `callAIAsync()` and `GRIMBackend`

**Legacy replaced**
- `resolveBackendURL()` deleted entirely — no fallback, no shadow mode
- `callAIAsync()` rewritten: orchestrator-only, no legacy backend branches
- `ai_process_stream()` deleted — callers migrated to `callAIAsync()`
- `warmupOllamaModel()` deleted — no model-specific warmup
- `stopGRIMTextServer()` legacy fallback calls removed from `main.cpp`

**Status**
- Implemented; bootstrap wiring complete; `callAIAsync` is orchestrator-only; shutdown handlers wired; NLP/memory context wiring pending

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
| Request orchestration | `Orchestrator` | `generate(RequestContext)` |
| Backend dispatch | `IGenerationBackend` impls | `Orchestrator::callBackend()` → `backends_[model_id]->generate()` |
| Memory retrieval / write surface | `MemoryFacade` | `MemoryFacade::instance()` |
| Session interaction state | `SessionContextManager` | `SessionContextManager::instance()` |
| Unified tool registry | `ToolRegistry` | `ToolRegistry::instance()` |
| Action policy / Training Wheels | `ActionPolicyRegistry` | `ActionPolicyRegistry::instance().evaluate()` |
| Bootstrap wiring | `bootstrap/bootstrap.cpp` | globals + init block + idle tick thread |
| AI dispatch replacement | `ai/ai.cpp` | `callAIAsync()` → orchestrator-only, no legacy backend paths |
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
- `resolveBackendURL()` deleted from `ai/ai.cpp`, `ai/ai.hpp`, `core/grim_exports.hpp`
- `warmupOllamaModel()` deleted from `ai/ai.cpp`, `ai/ai.hpp`, `bootstrap/bootstrap.cpp`
- `ai_process_stream()` deleted from `ai/ai.cpp`, `ai/ai.hpp`; caller migrated to `callAIAsync()` in `voice/voice_stream.cpp`
- `g_currentModel` / `g_modelWarmedUp` statics deleted from `ai/ai.cpp`
- `autoSelectBackend()` deleted from `commands/commands_ai.cpp`
- `callAIAsync()` legacy backend branches (ollama/localai/openai/grim_native direct HTTP) deleted
- `Orchestrator::callBackend()` direct HTTP fallback deleted
- `GrimNativeBackend` `j["response"]` format fallback deleted
- `cmdGrimAi()` direct HTTP backend branches deleted (~100 lines)
- `cmdAiBackend()` `resolveBackendURL()` call and `autoSelectBackend()` call deleted
- 3 `stopGRIMTextServer()` legacy fallback calls deleted from `main.cpp`
- `cpr/cpr.h` include removed from `commands/commands_ai.cpp`

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
- `main.cpp` — creates `g_memoryFacade` after memory system init; deletes during shutdown
- `Orchestrator` — receives via `setMemoryFacade()` in `main.cpp`
- Future: `SessionContextManager` context enrichment

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
- Added `ChatMessage` struct (`role`, `content`) and conversation history API
- Added `conversation_history` vector and `system_prompt_set` flag to `SessionState`
- Conversation history for model prompts migrated from `ai.cpp` globals into session-scoped state

**Owns**
- Session-scoped interaction state: turn records, referent bindings, pending interactions, action episodes
- Session-scoped conversation history (`ChatMessage` log per session)
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
- `addMessage(session_id, role, content)` — append to conversation history
- `setSystemPrompt(session_id, content)` — insert system prompt at position 0 (idempotent)
- `getMessages(session_id)` → `vector<ChatMessage>` — full conversation history
- `clearHistory(session_id)` — wipe conversation history + reset system prompt flag
- `trimHistory(session_id, max_messages)` — remove oldest non-system messages

**Replaces**
- `memory/context_manager.hpp` and `memory/context_manager.cpp` — **DELETED**. `ContextSnapshot` struct extracted to `memory/context_snapshot.hpp`.
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
- Implemented; fully integrated; all legacy caller migrations complete; conversation history migrated from ai.cpp globals

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

### 17. Contracts (Phase 0)

**What changed**
- New `MMO/Core/Contracts.hpp` — `ContractError` enum, `ContractViolation` struct, envelope validators, envelope builders, `ComposedGenerationSpec` struct, render/parse functions
- New `MMO/Core/Contracts.cpp` — implementation of all validation and builder logic

**Owns**
- `ContractError` enum (17 structured error codes: MissingRequestId through SynthesisRefused)
- `ContractViolation` struct (code + field + human-readable message)
- `validateRequest(RequestEnvelope)` → optional violation (checks request_id, session_id, target_model_id, task, payload)
- `validateResponse(ResponseEnvelope, expected_request_id, expected_target_model_id)` → optional violation (checks schema_version, request_id/target_model_id correlation, non-empty result on Ok)
- `buildRouteRequest()` / `buildGenerateRequest()` / `buildSynthesizeRequest()` — envelope construction with Rule 20 throws on missing required fields
- `ComposedGenerationSpec` struct — TASK/SCOPE/ALLOWED_ASSUMPTIONS/OUTPUT_SCHEMA/REFUSE_IF/STYLE/MAX_LENGTH/injected_context
- `renderComposedGeneration()` — spec → tagged text
- `parseComposedGeneration()` — tagged text → optional spec

**Does not own**
- Transport envelope types (those live in `MMO/Shared/MMD.hpp`)
- HTTP dispatch (that's `Orchestrator::callBackend()`)
- Route decision parsing (that's `ModelRouter`)

**Canonical hook points**
- `validateRequest(env)` / `validateResponse(env, rid, mid)` — call before sending / after receiving
- `buildRouteRequest()` / `buildGenerateRequest()` / `buildSynthesizeRequest()` — construct validated envelopes
- `renderComposedGeneration(spec)` / `parseComposedGeneration(text)` — structured prompt template

**Current consumers**
- None yet — `Orchestrator` has its own `buildRouteEnvelope` / `buildGenerateEnvelope` / `buildSynthesizeEnvelope` private methods that should migrate to use these

**Legacy replaced**
- `Orchestrator::buildRouteEnvelope()` etc. will be replaced by `Contracts::buildRouteRequest()` etc. in a future migration

**Status**
- Implemented; not yet integrated into Orchestrator

### 18. RequestContext (Phase 0)

**What changed**
- New `MMO/Core/RequestContext.hpp` — `TurnSummary` struct, `RequestContext` struct

**Owns**
- `TurnSummary` struct — minimal turn record for cross-boundary transfer (turn_id, role, text, token_count)
- `RequestContext` struct — immutable per-request snapshot: correlation IDs (request_id, session_id, turn_id), prompt, system_prompt, metadata_json, deadline, recent_turns, tool_registry_version
- `hasDeadline()` / `remainingMs()` helpers

**Does not own**
- Session lifecycle (owned by ContextManager / body main loop)
- Turn storage (reads from session authority, does not write)
- Metadata generation (receives pre-built metadata_json from RouterMetadataBuilder)

**Canonical hook points**
- Callers construct `RequestContext` before calling `Orchestrator::generate()`
- Orchestrator passes fields to envelope builders as needed
- `tool_registry_version` enables cached tool-summary invalidation

**Current consumers**
- None yet — `OrchestratorRequest` currently serves this role; will be replaced/extended by `RequestContext`

**Legacy replaced**
- `OrchestratorRequest` fields (request_id, session_id, turn_id, prompt, system_prompt, metadata_json) duplicated in `RequestContext` — migration will unify

**Status**
- Implemented; not yet integrated

### 19. IGenerationBackend (Phase 1.1)

**What changed**
- New directory `ai/backends/`
- New `ai/backends/IGenerationBackend.hpp` — `GenerationOptions`, `HistoryEntry`, `GenerationResult` structs, `IGenerationBackend` interface

**Owns**
- `GenerationOptions` struct — per-call inference parameters (max_tokens, temperature, top_p, top_k, metadata_json, tool_summary, timeout_ms)
- `HistoryEntry` struct — single conversation turn for multi-turn generation
- `GenerationResult` struct — success/error + generated text + tokens_used
- `IGenerationBackend` interface — `generate()`, `generateWithHistory()`, `isAvailable()`, `getBackendId()`, `getBackendType()`

**Does not own**
- HTTP transport implementation (each concrete backend provides its own)
- Model loading/unloading (ModelLoader responsibility)
- LoRA loading (separate pipeline)
- Weight writes of any kind (all backends are read-only)

**Canonical hook points**
- `backend->generate(prompt, options)` — single-turn generation
- `backend->generateWithHistory(prompt, history, options)` — multi-turn
- `backend->isAvailable()` — health check before dispatching
- `backend->getBackendType()` — matches `BackendType` enum from MMD.hpp

**Future implementations**
- `LlamaCppBackend` — llama.cpp server for sub-models
- `ExternalBackend` — arbitrary HTTP endpoints

**Concrete implementations**
- `GrimNativeBackend` — see §23
- `OllamaBackend` — see §24

**Current consumers**
- `Orchestrator::callBackend()` — dispatches through registered `IGenerationBackend` instances; falls back to direct HTTP only if no backend registered for the model
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` factory creates concrete backends; `registerBackendsForModels()` registers them on the orchestrator for router + all sub-models

**Legacy replaced**
- `Orchestrator::callBackend()` direct cpr HTTP dispatch now gated behind backend registry lookup — direct HTTP is the fallback path, not the primary path

**Status**
- Interface defined; GrimNativeBackend and OllamaBackend implemented; Orchestrator dispatch wired; LlamaCpp/External pending

### 20. ModelRouter (Phase 1.3)

**What changed**
- Replaced empty stub `MMO/Router/ModelRouter.hpp` with full type definitions
- New `MMO/Router/ModelRouter.cpp` — JSON parsing implementation

**Owns**
- `RouteConfidence` struct — overall, user_intent, domain_match float scores
- `ParsedRouteResult` struct — success flag, RouteDecision, confidence scores, refusal, error
- `ModelRouter::parseRouteResponse(ResponseEnvelope)` — extracts RouteDecision from router envelope
- `ModelRouter::parseRouteJson(string)` — parses raw JSON router response

**Does not own**
- Route intelligence (grim-text model provides all routing logic)
- Model selection policy (body-side; ModelRouter only parses)
- HTTP transport (Orchestrator / backend handles that)
- ComposedGeneration template structure (Contracts.hpp owns that)

**Canonical hook points**
- `ModelRouter::parseRouteResponse(response)` → `ParsedRouteResult`
- `ModelRouter::parseRouteJson(json_text)` → `ParsedRouteResult`
- Expected router JSON: `{ "sub_model_id": "...", "composed_generation": "...", "confidence": 0.92, "diagnostics": {...} }`

**Current consumers**
- None yet — `Orchestrator::parseRouteResponse()` currently does its own parsing; will migrate to use `ModelRouter`

**Legacy replaced**
- `Orchestrator::parseRouteResponse()` private method will be replaced

**Status**
- Implemented; not yet integrated into Orchestrator

### 21. ToolGapPlanner (Phase 0)

**What changed**
- New `MMO/Core/ToolGapPlanner.hpp` — `ToolGapReason` enum, `ProposedToolSpec`, `ToolGapProposal`, `ToolGapDecision` enum, `ToolGapPlanner` class
- New `MMO/Core/ToolGapPlanner.cpp` — evaluation, parsing, and rationale formatting

**Owns**
- `ToolGapReason` enum (NoMatchingCapability, CapabilityMismatch, PermissionInsufficient, PolicyBlocked)
- `ProposedToolSpec` struct — minimal new-tool specification for user review
- `ToolGapProposal` struct — missing_capability, reason, closest_tool context, proposed_spec, rationale, original_request_id
- `ToolGapDecision` enum (Pending, Approved, Rejected, Deferred)
- `ToolGapPlanner::evaluate(capability, request_id)` — checks ToolRegistry for matching capability; returns proposal if nothing fits
- `ToolGapPlanner::parseProposedSpec(json)` — parses model-emitted JSON into `ProposedToolSpec`
- `ToolGapPlanner::formatRationale(proposal)` — generates user-visible explanation
- `findClosestTool()` — keyword/tag heuristic for nearest-miss context

**Does not own**
- Tool creation / scaffolding (separate build pipeline)
- Plugin compilation or hot-loading (PluginManager responsibility)
- ToolRegistry mutations (only reads)
- User interaction for approval (caller manages UI)
- Reward assignment for tool-gap detection (training pipeline)

**Canonical hook points**
- `planner.evaluate(capability, request_id)` → `optional<ToolGapProposal>`
- `ToolGapPlanner::parseProposedSpec(json)` → `ProposedToolSpec`
- `ToolGapPlanner::formatRationale(proposal)` → user-visible string
- On approval: caller takes `ProposedToolSpec` → scaffold → build → `PluginManager::loadPlugin()` → `ToolRegistry::registerBatch()`

**Current consumers**
- None yet — will be consumed by synthesize-step handling in Orchestrator when router returns a tool-gap signal

**Legacy replaced**
- No previous tool-gap detection existed — this is new

**Status**
- Implemented; not yet integrated

### 22. Orchestrator migration to Contracts / ModelRouter / RequestContext

**What changed**
- `MMO/Core/Orchestrator.hpp` — full rewrite:
  - Deleted `OrchestratorRequest` struct (replaced by `RequestContext`)
  - Deleted `RouteDecision` struct (moved to `Contracts.hpp`)
  - Deleted private methods: `buildRouteEnvelope`, `buildGenerateEnvelope`, `buildSynthesizeEnvelope`, `parseRouteResponse`, `validateCorrelation`
  - Added `ModelRouter router_` member
  - Added private `buildRouterScope(const RequestContext&)` helper for memory enrichment
  - `generate()` now takes `const RequestContext& ctx` instead of `const OrchestratorRequest&`
- `MMO/Core/Orchestrator.cpp` — full rewrite:
  - `generate()` uses Contracts builders: `buildRouteRequest()`, `buildGenerateRequest()`, `buildSynthesizeRequest()`
  - Validates every envelope with `validateRequest()` before sending
  - Validates every response with `validateResponse()` (replaces hand-rolled `validateCorrelation`)
  - Route parsing delegated to `router_.parseRouteResponse()` (ModelRouter)
  - Memory enrichment extracted to `buildRouterScope()` (preserves `memory_->retrieveForPrompt()` logic)
  - `callBackend()` unchanged (IGenerationBackend migration is Phase 1.1)
- `MMO/Core/Contracts.hpp` — `RouteDecision` struct added (moved from Orchestrator.hpp)
- `MMO/Router/ModelRouter.hpp` — include changed from `Orchestrator.hpp` to `Contracts.hpp` (breaks circular dependency)
- `ai/ai.cpp` — caller migrated from `OrchestratorRequest` to `RequestContext`

**Owns**
- `Orchestrator` class — orchestration flow (route → generate → synthesize)
- `buildRouterScope()` — memory enrichment scope JSON for route requests
- `callBackend()` — HTTP dispatch to model servers (temporary; future IGenerationBackend)

**Does not own**
- Envelope construction (Contracts: `buildRouteRequest`, `buildGenerateRequest`, `buildSynthesizeRequest`)
- Envelope/response validation (Contracts: `validateRequest`, `validateResponse`)
- Route response parsing (ModelRouter: `parseRouteResponse`)
- Request context creation (caller creates `RequestContext`)
- Memory retrieval (MemoryFacade)

**Canonical hook points**
- `orchestrator.generate(RequestContext)` → `OrchestratorResult`
- `orchestrator.setMemoryFacade(MemoryFacade*)` — late-bind memory
- `orchestrator.shutdown()` — unloads all models
- Contracts builders/validators called at every envelope boundary
- ModelRouter called once per route response

**Current consumers**
- `ai/ai.cpp::callAIAsync()` — sole caller, creates `RequestContext` with request_id + prompt

**Legacy replaced**
- `OrchestratorRequest` struct → `RequestContext`
- `RouteDecision` in Orchestrator.hpp → `RouteDecision` in Contracts.hpp
- `buildRouteEnvelope()` → `buildRouterScope()` + `buildRouteRequest()`
- `buildGenerateEnvelope()` → `buildGenerateRequest()`
- `buildSynthesizeEnvelope()` → `buildSynthesizeRequest()`
- `parseRouteResponse()` → `ModelRouter::parseRouteResponse()`
- `validateCorrelation()` → `Contracts::validateResponse()`

**Status**
- Fully implemented; compile verification pending

## Next systems to document when touched

- MemoryBufferRotation long-term flush scheduling (periodic vs shutdown-only)
- Physical visual context (camera/real-world semantic interpretation)
- Plugin hot-reload ToolRegistry cache invalidation
- LoRA training pipeline from ToolTrainingExamples + CorrectionTuples

### 13. Bootstrap phase separation (Phase 0.1)

**What changed**
- Refactored `runBootstrapChecks()` from one monolithic function into 5 named phase functions

**Owns**
- `bootstrapConfigAndStatics(argc, argv)` — Phase 1: config, aliases, fonts
- `bootstrapHardwareAndResources()` — Phase 2: hardware inventory, location, ResourceSignal, ResourceCoordinator
- `bootstrapSubsystems()` — Phase 3: voice (Coqui), RL bridge, GRIM-text server
- `bootstrapMMOLayer()` — Phase 4: ModelRegistry, ModelLoader, ProcessManager, Orchestrator, ToolRegistry seed, ActionPolicyRegistry, SessionContextManager
- `bootstrapWarmup()` — Phase 5: Ollama warmup, Whisper preload
- `runBootstrapChecks(argc, argv)` — public orchestrator that calls phases 1-5 in sequence
- `stopMMOIdleTick()` — stops the background idle-tick thread

**Does not own**
- Individual subsystem internals (each phase delegates to its own system)
- Shutdown teardown (owned by `main.cpp` signal handlers)

**Canonical hook points**
- `runBootstrapChecks(argc, argv)` — sole public entry point
- Each phase function is `static` (internal linkage, not callable externally)

**Status**
- Implemented; no behavioral change, purely structural

### 14. Plugin → ToolRegistry sync (Phase 0.25)

**What changed**
- `core/plugin_api_impl.cpp` now includes `MMO/Core/ToolRegistry.hpp`
- `api_register_command()` — after registering in `s_commands` and `commandMap`, also builds a `ToolDescriptor` (provider_type=Plugin, provider_name=plugin_name, permission_bits from plugin context) and registers in `ToolRegistry::instance()`
- `api_unregister_command()` — after removing from maps, calls `ToolRegistry::instance().unregisterTool(cmd_name)`
- `cleanupPluginContext()` — calls `ToolRegistry::instance().unregisterByProvider(plugin_name)` for atomic batch cleanup after per-command removal from `s_commands`/`commandMap`

**Owns**
- Plugin ↔ ToolRegistry synchronization at the command registration boundary

**Does not own**
- ToolRegistry lifecycle (singleton, bootstrapped independently)
- Plugin lifecycle / hot-reload detection (owned by `PluginManager`)
- Command execution dispatch (owned by `commands_core.cpp`)

**Canonical hook points**
- `api_register_command()` — registers tool descriptor
- `api_unregister_command()` — unregisters individual tool
- `cleanupPluginContext()` — atomic batch unregister via `unregisterByProvider()`
- `PluginManager::checkForHotReload()` — triggers unload→reload which flows through the above hooks

**Current consumers**
- Any plugin that calls `api->register_command()` now automatically appears in ToolRegistry
- Hot-reload (file timestamp change) triggers unload→cleanup→reload→re-register cycle, keeping ToolRegistry in sync

**Legacy replaced**
- Plugin commands were previously invisible to ToolRegistry (only visible through `s_commands` and `commandMap`)
- Bootstrap's CommandRegistry seed already handled built-in tools; plugin tools were the gap

**Status**
- Implemented; ToolRegistry version counter increments on every plugin load/unload/reload

### 15. NlpAnnotation (Phase 0.5)

**What changed**
- New `nlp/NlpAnnotation.hpp` — canonical `NlpAnnotation` struct replacing `Intent` as the authoritative NLP output
- New `nlp/NlpAnnotation.cpp` — `annotate()` function that wraps `NLP::parse()`, `IntentGate::decide()`, `FastClassifier::evaluate()` into the new payload

**Owns**
- `Entity` struct — recognized semantic spans (type-aligned with GRIM-text atom concepts)
- `Ambiguity` struct — unresolved competing interpretations
- `UtterancePriors` struct — cheap command/question/banter probabilities
- `ConfidenceSummary` struct — aggregate annotation quality signal
- `NlpAnnotation` struct — full payload: raw/normalized text, language, utterance priors, entities, action affordances, candidate tool tokens, tool context hints, memory/router/context/risk tags, references, ambiguities, confidence summary, legacy compatibility fields
- `annotate(raw_input, context)` — main entry point producing `NlpAnnotation` from existing NLP infrastructure
- Regex-based entity extraction (URL, email, path, number — aligned with GRIM-text atoms)
- Regex-based affordance detection (open, search, navigate, edit, delete, create, copy, move, close)
- Regex-based reference detection (pronouns and deictics)
- Regex-based context tag detection (coding, filesystem, web, system, ui, voice)
- Regex-based risk tag detection (destructive, system, network, credential_sensitive)

**Does not own**
- NLP rule management (still `NLP` class in `nlp/nlp.hpp`)
- Intent classification weights (still `FastClassifier` + `IntentGate`)
- Command dispatch or execution (annotation only — no executable coupling)
- Memory storage or retrieval (annotation provides tags consumed by `MemoryFacade`)

**Canonical hook points**
- `GRIM::annotate(raw_input, context)` — produces `NlpAnnotation`
- Legacy compatibility: `legacy_intent_name`, `legacy_category`, `legacy_matched` fields populated from `NLP::parse()` output

**Current consumers**
- None yet — newly created; will be consumed by `RouterMetadataBuilder`, `SessionContextManager::beginTurn()`, `MemoryFacade` tagging, and `Orchestrator` context composition
- `Intent` remains usable for legacy consumers during migration

**Integrated call sites**
- `ai/ai.cpp::callAIAsync()` — calls `GRIM::annotate(prompt, snapshot)` before building `RequestContext`; annotation feeds into `RouterMetadataBuilder`
- `ai/ai.cpp::ai_interpret()` — calls `GRIM::annotate(input, snapshot)` to derive real confidence signals for `ActionProposal` (replaces hardcoded 0.7f)

**Legacy replaced**
- `Intent` is now a secondary output; `NlpAnnotation` is the canonical payload
- Hardcoded `proposal.router_confidence = 0.7f` → `policyAnn.confidence_summary.overall`
- Hardcoded `proposal.parse_certainty = 1.0f` → `policyAnn.confidence_summary.intent_confidence`
- Hardcoded `referent_resolution = 1.0f` → conditional on `policyAnn.references.empty()`
- Hardcoded `grounding_coverage = 1.0f` → `policyAnn.confidence_summary.entity_confidence`
- Hardcoded `0.7f` in `recordOutcome()` calls → `policyAnn.confidence_summary.overall`

**Status**
- Implemented; integrated into `callAIAsync` and `ai_interpret`

### 16. RouterMetadataBuilder (Phase 0.5)

**What changed**
- New `nlp/RouterMetadataBuilder.hpp` — `RouterMetadata` struct + `RouterMetadataBuilder` builder
- New `nlp/RouterMetadataBuilder.cpp` — builds JSON-serializable metadata envelope for the router model

**Owns**
- `RouterMetadata` struct — the envelope sent to grim-text router: raw/normalized input, serialized NlpAnnotation, serialized ContextSnapshot, memory tags, tool summary, visual context (physical + digital), risk tags, action policy hints, confidence snapshot
- `RouterMetadata::toJson()` — full JSON serialization
- `RouterMetadataBuilder` — fluent builder: `setAnnotation()`, `setContext()`, `setToolSummary()`, `setPhysicalVisualContext()`, `setDigitalVisualContext()`, `setActionPolicyHints()`, `build()`
- NlpAnnotation → JSON serialization (all fields: entities, tags, ambiguities, confidence, legacy)
- ContextSnapshot → JSON serialization (recent intents/commands, mood, depth, etc.)

**Does not own**
- NLP analysis (delegates to `NlpAnnotation`)
- Context state management (reads from `ContextSnapshot`)
- Tool registry content (receives compact prompt string)
- Visual context capture (receives pre-built JSON from perception subsystem)
- Action policy evaluation (receives pre-built hints)
- HTTP transport to router (consumed by `Orchestrator`)

**Canonical hook points**
- `RouterMetadataBuilder().setAnnotation(ann).setContext(ctx).setToolSummary(prompt).build()` — produces `RouterMetadata`
- `RouterMetadata::toJson()` — serializes for HTTP payload to router model
- Builder throws `std::runtime_error` if annotation or context are not set (Rule 20: fail loud)

**Current consumers**
- `ai/ai.cpp::callAIAsync()` — builds `RouterMetadata` from `NlpAnnotation` + `ContextSnapshot` + `ToolRegistry::generateCompactPrompt()`, serializes to JSON for `RequestContext::metadata_json`
- `Orchestrator::buildRouterScope()` — receives pre-built metadata via `ctx.metadata_json`, merges with memory retrieval results

**Legacy replaced**
- Orchestrator previously sent raw prompt text as scope; now receives structured `RouterMetadata` JSON envelope

**Status**
- Implemented; integrated into `callAIAsync` → Orchestrator pipeline

## Required update checklist for future agents

When refactoring a system, append or update the relevant section in this file with:
1. owning files
2. exact responsibility boundary
3. explicit non-responsibilities
4. canonical integration points
5. migrated consumers
6. deleted legacy paths
7. validation performed

### 23. GrimNativeBackend (`MMO/Backends/GrimNativeBackend.hpp`, `MMO/Backends/GrimNativeBackend.cpp`)

**What changed**
- New concrete `IGenerationBackend` for grim_text_server.exe
- Added `generateEnvelope()` private method — POSTs full `RequestEnvelope` JSON to MMO endpoints
- `generate()` now checks `options.envelope_json` + `options.mmo_endpoint`; routes through `generateEnvelope()` when present, falls through to `generateWithHistory()` via `/api/chat` otherwise

**Owns**
- HTTP transport to grim_text_server `/api/chat` endpoint using Ollama-compatible JSON format
- HTTP transport to grim_text_server `/api/mmo/*` endpoints using full `RequestEnvelope` JSON
- `generateEnvelope()` — POSTs `envelope_json` to `base_url + mmo_endpoint`, returns raw response text
- Model name forced to `"grim-text"` regardless of config model_id
- Health check via GET `/api/tags`
- Single-turn and multi-turn generation (history mapped to messages array)
- Timeout enforcement via cpr::Timeout
- Backend ID format: `"grim_text:{model_id}"`

**Does not own**
- grim_text_server process lifecycle (ModelLoader + StartCallback)
- Model loading/unloading decisions
- LoRA or fine-tune management
- Response envelope parsing (caller's concern — `resultToEnvelope()` in Orchestrator)

**Canonical hook points**
- `backend->generate(prompt, options)` — dispatches to envelope or chat path
- `backend->generateWithHistory(prompt, history, options)` — multi-turn via `/api/chat`
- `backend->isAvailable()` — HTTP health check
- `backend->getBackendId()` — `"grim_text:{model_id}"`
- `backend->getBackendType()` — `BackendType::GrimTextServer`
- `generateEnvelope(options)` — internal, POSTs to `/api/mmo/route`, `/api/mmo/generate`, or `/api/mmo/synthesize`

**Current consumers**
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` creates for `BackendType::GrimTextServer`
- `Orchestrator` — dispatches through `IGenerationBackend` interface via `callBackend()`

**Status**
- Implemented; registered at bootstrap; Orchestrator dispatch wired; MMO envelope dispatch added

### 24. OllamaBackend (`MMO/Backends/OllamaBackend.hpp`, `MMO/Backends/OllamaBackend.cpp`)

**What changed**
- New concrete `IGenerationBackend` for Ollama API sub-models

**Owns**
- HTTP transport to Ollama `/api/chat` endpoint
- Ollama-specific options: `num_predict`, `temperature`, `top_p`, `top_k`
- `keep_alive: "30m"` for model residency
- Separate `ollama_model` name (e.g., `"llama3.1:8b"`) from MMO `model_id`
- Health check via GET `/api/tags`
- Single-turn and multi-turn generation
- Backend ID format: `"ollama:{model_id}"`

**Does not own**
- Ollama server lifecycle
- Model pull/download
- LoRA management
- Response envelope parsing

**Canonical hook points**
- `backend->generate(prompt, options)` — single-turn
- `backend->generateWithHistory(prompt, history, options)` — multi-turn
- `backend->isAvailable()` — HTTP health check
- `backend->getBackendId()` — `"ollama:{model_id}"`
- `backend->getBackendType()` — `BackendType::Ollama`

**Config contract**
- `model_path` in `ModelInfo` is used as `ollama_model` name
- `url` in `ModelInfo` is the Ollama server URL (e.g., `http://localhost:11434`)

**Current consumers**
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` creates for `BackendType::Ollama`
- `Orchestrator` — dispatches through `IGenerationBackend` interface via `callBackend()`

**Status**
- Implemented; registered at bootstrap; Orchestrator dispatch wired

### 25. callBackend → IGenerationBackend dispatch (Phase 1.1 wiring)

**What changed**
- `Orchestrator::callBackend()` migrated from direct-HTTP-only to backend registry dispatch
- `Orchestrator.hpp` — added `backends_` member (`unordered_map<string, unique_ptr<IGenerationBackend>>`), `registerBackend()`, `resultToEnvelope()`
- `Orchestrator.cpp` — `callBackend()` now checks `backends_` first, dispatches via `backend->generate()`, converts result via `resultToEnvelope()`; direct HTTP is fallback only
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` factory + registration loop for router and all sub-models
- Endpoint paths updated: `/api/route` → `/api/mmo/route`, `/api/generate` → `/api/mmo/generate`, `/api/synthesize` → `/api/mmo/synthesize`
- IGenerationBackend dispatch path now serializes full `RequestEnvelope` into `opts.envelope_json` + `opts.mmo_endpoint` before calling `backend->generate()`

**Owns**
- Backend registry on Orchestrator (`backends_` map)
- `registerBackend(model_id, backend)` — stores backends
- `resultToEnvelope()` — converts `GenerationResult` to `ResponseEnvelope` (parses structured JSON if present, wraps raw text otherwise)
- Backend dispatch priority: registered backend → throw on missing (no HTTP fallback)
- Envelope serialization into `GenerationOptions` for backend consumption

**Does not own**
- Concrete backend implementations (see §23, §24)
- Backend factory logic (bootstrap owns `createBackendForModel()`)
- Model lifecycle (ModelLoader)

**Canonical hook points**
- `orchestrator.registerBackend(model_id, backend)` — called during bootstrap
- `callBackend()` — internal, checks `backends_[model_id]` before HTTP fallback
- `opts.envelope_json` / `opts.mmo_endpoint` — populated before `backend->generate()` when envelope available

**Legacy replaced**
- `callBackend()` was direct cpr HTTP only → now gated behind backend registry
- Endpoint paths `/api/route`, `/api/generate`, `/api/synthesize` → `/api/mmo/*` prefix

**Status**
- Implemented; all configured models get backends registered at bootstrap; envelope transport wired; direct HTTP fallback deleted from `callBackend()` — throws `runtime_error` if backend not registered

### 26. Config loading from ai_config.json (Phase 3 wiring)

**What changed**
- `ai_config.json` — added `mmo.model_loader` section (load_timeout_ms, idle_ttl_ms, hot_ttl_cap_ms, use_degrade_step_ms)
- `ai_config.json` — added `mmo.orchestrator` section (route_timeout_ms, generate_timeout_ms, synthesize_timeout_ms, max_submodels_per_request)
- `ai_config.json` — added top-level `training_wheels` section (enabled, risk_threshold, min_confidence_floor, uncalibrated_router_confidence, calibration_min_samples, per_category_thresholds)
- `bootstrap/bootstrap.cpp` — `bootstrapMMOLayer()` now reads `aiConfig["mmo"]["model_loader"]` into `ModelLoaderConfig` fields and `aiConfig["mmo"]["orchestrator"]` into `OrchestratorConfig` fields instead of using hardcoded defaults

**Owns**
- Config JSON schema for `mmo.model_loader`, `mmo.orchestrator`, and `training_wheels`
- Bootstrap config reading logic (JSON → struct field mapping)

**Does not own**
- Config struct definitions (ModelLoaderConfig in ModelLoader.hpp, OrchestratorConfig in Orchestrator.hpp, ActionPolicyConfig in ActionPolicyRegistry.hpp)
- Config validation beyond JSON field presence

**Canonical hook points**
- `aiConfig["mmo"]["model_loader"]` → `ModelLoaderConfig` fields
- `aiConfig["mmo"]["orchestrator"]` → `OrchestratorConfig` fields
- `aiConfig["training_wheels"]` → `ActionPolicyConfig` (already wired previously)

**Current consumers**
- `bootstrap/bootstrap.cpp::bootstrapMMOLayer()` — sole reader

**Legacy replaced**
- Hardcoded default `ModelLoaderConfig loaderCfg;` and `OrchestratorConfig orchCfg;` — now config-driven with same defaults as fallback

**Status**
- Implemented; build verified

### 27. grim_text_server MMO endpoints (`resources/models/GRIM-text/GRIM/grim_text_server.cpp`)

**What changed**
- Extracted `applyGenerationOverrides(json, GenerationConfig&)` helper — replaces duplicated config parsing in `/api/generate` and `/api/chat`
- Added `POST /api/mmo/route` — validates `RequestEnvelope`, constructs routing prompt from `scope` + `payload`, generates, returns `ResponseEnvelope`
- Added `POST /api/mmo/generate` — validates envelope, generates from `payload` (composed generation), returns envelope
- Added `POST /api/mmo/synthesize` — validates envelope, generates synthesis from `payload` (sub-model results), returns envelope
- Added `validateEnvelope` lambda for shared `RequestEnvelope` field validation (schema_version, request_id, session_id)

**Owns**
- HTTP endpoint handlers for `/api/mmo/route`, `/api/mmo/generate`, `/api/mmo/synthesize`
- `applyGenerationOverrides()` — centralized config override extraction from JSON into `GenerationConfig`
- `validateEnvelope()` — shared validation for incoming `RequestEnvelope` fields
- `ResponseEnvelope` JSON construction for MMO responses (status, result text)

**Does not own**
- Orchestrator routing logic (Orchestrator owns route→generate→synthesize flow)
- `RequestEnvelope` / `ResponseEnvelope` struct definitions (`MMO/Shared/MMD.hpp`)
- Model loading/inference — delegates to existing generation pipeline

**Canonical hook points**
- `POST /api/mmo/route` — Orchestrator calls this for routing decisions
- `POST /api/mmo/generate` — Orchestrator calls this for composed generation
- `POST /api/mmo/synthesize` — Orchestrator calls this for multi-result synthesis
- `applyGenerationOverrides(body, config)` — called by `/api/generate`, `/api/chat`, and all MMO endpoints

**Current consumers**
- `Orchestrator::callBackend()` — direct HTTP fallback path posts to these endpoints
- `GrimNativeBackend::generateEnvelope()` — IGenerationBackend path posts to these endpoints

**Legacy replaced**
- Inline duplicated config parsing in `/api/generate` and `/api/chat` → `applyGenerationOverrides()`

**Status**
- Implemented; build verified

### 28. Session-scoped conversation history migration (`ai/ai.cpp`)

**What changed**
- Removed static globals: `g_conversationHistory` (vector<json>), `g_systemPromptAdded` (bool)
- Added `prepareConversationMessages(prompt, session_id)` helper — handles system prompt, user message, trimming, JSON serialization
- Both grim_native and ollama backend paths in `ai_interpret()` now use `prepareConversationMessages()` + `SessionContextManager::addMessage()` for assistant responses
- `clearConversationHistory()` now delegates to `SessionContextManager::clearHistory(kDefaultSessionId)`

**Owns**
- `prepareConversationMessages()` — builds messages array via SessionContextManager API
- `kDefaultSessionId` constant (`"default"`) — session key for legacy single-session path
- Integration point between `ai_interpret()` and SessionContextManager conversation history

**Does not own**
- Conversation history storage (delegated to `SessionContextManager` §10)
- Session lifecycle
- Message format for non-Ollama/non-grim-native backends

**Canonical hook points**
- `prepareConversationMessages(prompt, session_id)` → returns `nlohmann::json` messages array
- `SessionContextManager::addMessage(session_id, "assistant", response)` — called after generation
- `clearConversationHistory()` → `SessionContextManager::clearHistory(kDefaultSessionId)`

**Legacy replaced**
- `static std::vector<nlohmann::json> g_conversationHistory` — **DELETED**
- `static bool g_systemPromptAdded` — **DELETED**
- Inline conversation history management in grim_native/ollama paths → `prepareConversationMessages()`

**Status**
- Implemented; build verified; all conversation state now session-scoped via SessionContextManager

### 29. IGenerationBackend envelope transport (`ai/backends/IGenerationBackend.hpp`)

**What changed**
- Added `std::string envelope_json` to `GenerationOptions` — full `RequestEnvelope` serialized as JSON
- Added `std::string mmo_endpoint` to `GenerationOptions` — target endpoint path (e.g., `/api/mmo/route`)
- Populated by `Orchestrator::callBackend()` when dispatching through IGenerationBackend with an envelope

**Owns**
- `envelope_json` and `mmo_endpoint` fields on `GenerationOptions` struct

**Does not own**
- Envelope serialization (Orchestrator §25 owns)
- Envelope consumption (concrete backends §23, §24 own)

**Canonical hook points**
- `options.envelope_json` — non-empty when an MMO envelope should be forwarded
- `options.mmo_endpoint` — the endpoint path the backend should POST to
- Backends check `!options.envelope_json.empty()` to decide between envelope dispatch and legacy chat dispatch

**Current consumers**
- `GrimNativeBackend::generate()` — routes through `generateEnvelope()` when envelope present
- `OllamaBackend` — currently ignores (Ollama doesn't serve MMO endpoints)

**Status**
- Implemented; build verified

### 30. LlamaCppBackend (`MMO/Backends/LlamaCppBackend.hpp`, `MMO/Backends/LlamaCppBackend.cpp`)

**What changed**
- New concrete `IGenerationBackend` for llama.cpp server

**Owns**
- HTTP transport to llama.cpp's OpenAI-compatible `/v1/chat/completions` endpoint
- Health check via GET `/health`
- Single-turn and multi-turn generation using OpenAI chat completions format
- Timeout enforcement via cpr::Timeout
- Backend ID format: `"llamacpp:{model_id}"`

**Does not own**
- llama.cpp server process lifecycle (ModelLoader + StartCallback)
- Model loading/unloading decisions
- LoRA or GGUF management
- Response envelope parsing (caller's concern — `resultToEnvelope()` in Orchestrator)

**Canonical hook points**
- `backend->generate(prompt, options)` — single-turn via `/v1/chat/completions`
- `backend->generateWithHistory(prompt, history, options)` — multi-turn
- `backend->isAvailable()` — GET `/health`
- `backend->getBackendId()` — `"llamacpp:{model_id}"`
- `backend->getBackendType()` — `BackendType::LlamaCpp`

**Config contract**
- `model_path` in `ModelInfo` is used as the model name sent to the server
- `url` in `ModelInfo` is the llama.cpp server URL (e.g., `http://localhost:8080`)

**Current consumers**
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` creates for `BackendType::LlamaCpp`
- `Orchestrator` — dispatches through `IGenerationBackend` interface via `callBackend()`

**Status**
- Implemented; registered at bootstrap; Orchestrator dispatch wired

### 31. ExternalBackend (`MMO/Backends/ExternalBackend.hpp`, `MMO/Backends/ExternalBackend.cpp`)

**What changed**
- New concrete `IGenerationBackend` for arbitrary HTTP model endpoints

**Owns**
- HTTP POST transport to user-configured endpoint path
- Multi-format response parsing: OpenAI → Ollama → `text`/`response` field → raw text fallback
- Health check via GET `/` (accepts any 2xx)
- Single-turn and multi-turn generation using OpenAI-compatible message format
- Timeout enforcement via cpr::Timeout
- Backend ID format: `"external:{model_id}"`

**Does not own**
- External server process lifecycle
- Model loading/unloading decisions
- Authentication (future concern — no auth headers yet)
- Response envelope parsing (caller's concern)

**Canonical hook points**
- `backend->generate(prompt, options)` — single-turn via configured endpoint
- `backend->generateWithHistory(prompt, history, options)` — multi-turn
- `backend->isAvailable()` — GET `/`
- `backend->getBackendId()` — `"external:{model_id}"`
- `backend->getBackendType()` — `BackendType::External`

**Config contract**
- `model_path` in `ModelInfo` is used as the endpoint path (e.g., `/v1/chat/completions`, `/generate`)
- `url` in `ModelInfo` is the external server base URL
- If `model_path` is empty, defaults to `/v1/chat/completions`

**Current consumers**
- `bootstrap/bootstrap.cpp` — `createBackendForModel()` creates for `BackendType::External`
- `Orchestrator` — dispatches through `IGenerationBackend` interface via `callBackend()`

**Status**
- Implemented; registered at bootstrap; Orchestrator dispatch wired

---

### 32. ProcessManager — model-keyed server process lifecycle

**What changed**
- New `MMO/Core/ProcessManager.hpp` + `ProcessManager.cpp`
- Replaces the singleton `GRIMTextServerManager` in the MMO path
- Bootstrap callbacks on `ModelLoader` now delegate to `ProcessManager::start(model)` / `stop(model_id)` instead of the singleton
- `main.cpp` shutdown handlers call `g_processManager->stopAll()` before legacy `stopGRIMTextServer()`
- Global `g_processManager` declared in `bootstrap.hpp`, allocated in `bootstrapMMOLayer()`

**Owns**
- One OS process per model ID (model-keyed `ProcessSlot` map)
- Port uniqueness validation — throws `std::runtime_error` if two models claim the same port
- Per-model Windows mutex identity (`Global\GRIMTextServer_{model_id}`)
- GrimTextServer process spawning (path resolution, `CreateProcessA`, health poll)
- Graceful + forceful process termination (CTRL_C_EVENT → TerminateProcess fallback)

**Does not own**
- Model load/unload state machine (owned by `ModelLoader`)
- Config loading — receives `ModelInfo` which already has paths/ports populated
- Backend logic — only manages the OS process; backends are registered separately
- Non-GrimTextServer backends (Ollama, LlamaCpp, External) — marked as running but not process-managed

**Canonical hook points**
- `g_processManager->start(model)` — ModelLoader start callback
- `g_processManager->stop(model_id)` — ModelLoader stop callback
- `g_processManager->stopAll()` — main.cpp shutdown
- `g_processManager->checkHealth(model_id, timeout_ms)` — HTTP GET health check
- `g_processManager->isRunning(model_id)` — OS-level process alive check
- `g_processManager->getUrl(model_id)` — retrieve the URL for a running model

**Current consumers**
- `bootstrap/bootstrap.cpp` — creates `g_processManager`, wires into ModelLoader callbacks
- `main.cpp` — `consoleHandler()` and normal shutdown path call `stopAll()` + delete

**Status**
- Implemented; replaces singleton delegation in MMO path; legacy `stopGRIMTextServer()` fallback calls deleted from `main.cpp`

### 33. Legacy path purge — no backwards compatibility

**What changed**
- `resolveBackendURL()` deleted from `ai/ai.cpp`, `ai/ai.hpp`, `core/grim_exports.hpp`
- `warmupOllamaModel()` deleted from `ai/ai.cpp`, `ai/ai.hpp`, `bootstrap/bootstrap.cpp`
- `ai_process_stream()` deleted from `ai/ai.cpp`, `ai/ai.hpp`; caller in `voice/voice_stream.cpp` migrated to `callAIAsync()`
- `g_currentModel` / `g_modelWarmedUp` statics deleted from `ai/ai.cpp`
- `callAIAsync()` rewritten to orchestrator-only — no shadow mode, no legacy backend branches (ollama/localai/openai/grim_native direct HTTP deleted)
- `Orchestrator::callBackend()` — direct HTTP fallback deleted; throws `runtime_error` if no backend registered
- `GrimNativeBackend::generate()` — `j["response"]` fallback format deleted; enforces canonical `j["message"]["content"]` only
- `cmdAiBackend()` — `resolveBackendURL()` call deleted, `autoSelectBackend()` helper deleted, simplified to config-set-only
- `cmdGrimAi()` — all direct HTTP backend branches deleted (~100 lines); delegates to `ai_process()` only
- `main.cpp` — 3 `stopGRIMTextServer()` legacy fallback calls deleted (lines 103, 157, 500)
- `bootstrap.cpp` — `warmupOllamaModel()` call removed from Phase 5 warmup
- `cpr/cpr.h` include removed from `commands/commands_ai.cpp`

**Deleted functions**
- `resolveBackendURL()` — was in `ai/ai.cpp`, exported via `grim_exports.hpp`
- `warmupOllamaModel()` — was in `ai/ai.cpp`
- `ai_process_stream()` — was in `ai/ai.cpp`
- `autoSelectBackend()` — was in `commands/commands_ai.cpp`

**Deleted statics**
- `g_currentModel` — was in `ai/ai.cpp`
- `g_modelWarmedUp` — was in `ai/ai.cpp`

**Files modified**
- `ai/ai.cpp` — bulk of deletions
- `ai/ai.hpp` — 3 declarations removed
- `core/grim_exports.hpp` — `resolveBackendURL` export removed
- `commands/commands_ai.cpp` — `autoSelectBackend`, direct HTTP branches, `cpr` include removed
- `voice/voice_stream.cpp` — `ai_process_stream` → `callAIAsync`
- `main.cpp` — 3 `stopGRIMTextServer` calls removed
- `bootstrap/bootstrap.cpp` — `warmupOllamaModel` call + comment removed
- `MMO/Core/Orchestrator.cpp` — direct HTTP fallback in `callBackend()` removed
- `MMO/Backends/GrimNativeBackend.cpp` — `j["response"]` format fallback removed

**Validation status**
- All `resolveBackendURL` references eliminated (verified via grep)
- All `ai_process_stream` references eliminated
- All `warmupOllamaModel` references eliminated
- All `stopGRIMTextServer` legacy fallback calls eliminated
- CommandRegistry → ToolRegistry seeding bridge retained (data sync, not compat shim)

**Remaining gaps**
- `warmupAI()` in `ai/ai.cpp` still exists as a minimal hello-world warmup — delegates through `callAIAsync` → orchestrator, so it's clean
- `ai_process()` export in `grim_exports.hpp` retained — used by external consumers; internally routes through orchestrator

---

## §34 — Config hardening + ContextSnapshotV2 wiring + ToolGapPlanner integration + MMD.cpp

**What changed**

1. **ai_config.json** — Added three missing config sections that bootstrap code already read with fallback defaults:
   - `mmo.model_loader`: `load_timeout_ms`, `idle_ttl_ms`, `hot_ttl_cap_ms`, `use_degrade_step_ms`
   - `mmo.orchestrator`: `route_timeout_ms`, `generate_timeout_ms`, `synthesize_timeout_ms`, `max_submodels_per_request`
   - `training_wheels` (top-level): `enabled`, `risk_threshold`, `min_confidence_floor`, `uncalibrated_router_confidence`, `calibration_min_samples`, `per_category_thresholds`

2. **ContextSnapshotV2 wired end-to-end** — Primary AI dispatch paths now use V2 rich snapshots instead of V1 legacy projections:
   - `NlpAnnotation.hpp/.cpp` — Added `annotate(raw_input, ContextSnapshotV2)` overload. Projects V2→V1 internally for IntentGate/FastClassifier (which still consume V1 fields).
   - `RouterMetadataBuilder.hpp/.cpp` — Added `setContextV2(ContextSnapshotV2)`. Stores owned V1 copy (projected from V2) + serializes V2-only rich fields (referents, turn summaries, visual context, mood, pressure) into `context_snapshot.v2` in the router metadata JSON. Also auto-populates `visual_context_physical` and `visual_context_digital` from V2.
   - `ai.cpp` — Both call sites (`callAIAsync()` and `ai_interpret()` policy evaluation) switched from `scm.legacySnapshot("default")` → `scm.snapshot("default")`, using V2 overloads of `annotate()` and `setContextV2()`.

3. **ToolGapPlanner integrated into Orchestrator** — When the model suggests a tool/command that isn't in the ToolRegistry, the system now returns a structured gap proposal instead of a cryptic error:
   - `Orchestrator.hpp` — Added `ToolGapPlanner tool_gap_planner_` member, `evaluateToolGap(capability, request_id)` public method.
   - `Orchestrator.cpp` — Constructor initializes `tool_gap_planner_(ToolRegistry::instance())`. `evaluateToolGap()` delegates to planner.
   - `ai.cpp` — In `ai_interpret()`, after parsing a suggested command, checks `toolRegGap.isRegistered(cmd)` and `resolveAlias(cmd)`. If unrecognized, calls `g_orchestrator->evaluateToolGap()` and returns the gap rationale to the user.

4. **MMD.cpp implemented** — `getSubjectTags(raw_input)` declared in `MMO/Shared/MMD.hpp`, implemented in new `MMO/Shared/MMD.cpp`:
   - Keyword-based subject tag extraction for subject-based routing
   - Maps ~45 keywords across 9 subject categories (math, science, programming, technology, language, writing, creative, knowledge, general)
   - Returns `{"general"}` when no specific tags match
   - Tags match against `ModelInfo::subject_tags` for sub-model selection

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Config defaults for MMO subsystems | `ai_config.json` (explicit) + bootstrap fallbacks (safety net) |
| Rich context for AI dispatch | `SessionContextManager::snapshot()` → `ContextSnapshotV2` |
| V1 context for IntentGate/FastClassifier | `annotate(V2)` internal projection; `legacySnapshot()` for other callers |
| Visual context in router metadata | `RouterMetadataBuilder::setContextV2()` auto-populates from V2 |
| Tool gap detection | `Orchestrator::evaluateToolGap()` → `ToolGapPlanner::evaluate()` |
| Subject tag extraction | `GRIM::MMO::getSubjectTags()` in `MMO/Shared/MMD.cpp` |

**Canonical integration points**

- `annotate(raw_input, ContextSnapshotV2)` — V2-aware NLP annotation entry point
- `RouterMetadataBuilder::setContextV2(v2)` — V2-aware metadata builder
- `Orchestrator::evaluateToolGap(capability, request_id)` — structured tool gap check
- `GRIM::MMO::getSubjectTags(raw_input)` — subject tag extraction for routing

**Files modified**
- `ai_config.json` — 3 new config sections
- `nlp/NlpAnnotation.hpp` — V2 overload declaration
- `nlp/NlpAnnotation.cpp` — V2 overload implementation + `#include SessionContextManager.hpp`
- `nlp/RouterMetadataBuilder.hpp` — `setContextV2()` method, `has_v2_`/`owned_v1_`/`context_v2_json_` members
- `nlp/RouterMetadataBuilder.cpp` — `setContextV2()` impl, `build()` updated for V2
- `ai/ai.cpp` — Both `legacySnapshot()` call sites → `snapshot()` + V2 overloads; tool gap check added
- `MMO/Core/Orchestrator.hpp` — `ToolGapPlanner` include + member + `evaluateToolGap()` method
- `MMO/Core/Orchestrator.cpp` — `ToolGapPlanner` init in constructor + `evaluateToolGap()` impl
- `MMO/Shared/MMD.hpp` — `getSubjectTags()` declaration
- `MMO/Shared/MMD.cpp` — **NEW FILE** — keyword-based subject tag extraction

**Validation status**
- `legacySnapshot()` still exists on SessionContextManager for `commands_core.cpp` (3 sites) and `MemoryFacade.cpp` (2 sites) — these only need V1 fields (mood, consecutive commands)
- Primary AI dispatch (`callAIAsync` + `ai_interpret` policy) fully on V2
- ToolGapPlanner fires before ActionPolicyRegistry in `ai_interpret()`, so unrecognized tools never hit the policy gate
- `ai_config.json` validated as valid JSON

**Remaining gaps**
- `commands_core.cpp` and `MemoryFacade.cpp` still use `legacySnapshot()` — low priority, they only need mood/consecutive-command V1 fields
- `getSubjectTags()` not yet called from router — ModelRouter needs wiring to use it during route decision
- ToolGapPlanner returns proposals but no UI flow for user approval/scaffold/hot-load yet
- Visual context in V2 not yet populated by any perception system (fields are empty defaults)

---

## §35 — Subject tags in routing + NLP stubs fixed + EmotionPresentationController + Correction tuples

**What changed**

1. **Subject tags wired into routing metadata** — `RouterMetadataBuilder::build()` now calls `GRIM::MMO::getSubjectTags()` on the raw user input and includes the extracted subject tags in the router metadata envelope (`subject_tags` field). The router model can use these to inform sub-model selection alongside `ModelInfo::subject_tags`.
   - `RouterMetadataBuilder.hpp` — Added `subject_tags` field to `RouterMetadata` struct
   - `RouterMetadataBuilder.cpp` — Added `#include "../MMO/Shared/MMD.hpp"`, calls `getSubjectTags()` in `build()`, serializes in `toJson()`

2. **NLP stubs resolved** — Three TODO stubs replaced with real implementations:
   - **Language detection**: `NlpAnnotation.cpp` — Added `detectLanguage()` heuristic. Counts characters in Latin, CJK, Cyrillic, and Arabic UTF-8 ranges. Returns ISO 639-1 codes (`en`, `zh`, `ru`, `ar`). Replaces hardcoded `"en"`.
   - **loadLearnedRules**: `nlp.cpp` — Loads all learned commands from `UnifiedMemoryStorage::getAllLearnedCommands()`, reconstructs `NLP::Rule` objects (pattern, intent, success_rate), skips duplicates and malformed patterns.
   - **saveLearnedRules**: `nlp.cpp` — Iterates rules where `learned == true`, stores each via `storage.storeLearnedCommand(pattern_str, intent, success_rate)`.
   - Note: `grammar_parser.cpp` `matchTemplate()` left as-is — requires significant template parsing design, not a simple stub fill.

3. **EmotionPresentationController created** — Separates internal mood state (PersonalityManager, used by routing/memory/policy/reward) from outward expression (UI/voice/avatar):
   - `MMO/UI/EmotionPresentationController.hpp` — **NEW FILE**. Defines `PresentationChannel` enum (Text/Voice/Avatar), `EmotionPresentation` struct (mood_label, text_prefix, voice_pitch/rate/emphasis, avatar_expression, intensity), `EmotionPresentationController` singleton.
   - `MMO/UI/EmotionPresentationController.cpp` — **NEW FILE**. Maps `PersonalityManager::get()` Mood values to outward presentation parameters. When disabled, returns neutral defaults. Per-channel enable/disable supported. All atomics, lock-free.
   - Mood→Expression mapping: Curious=raised pitch + "Hmm, " prefix; Playful=bright voice; Irritated=lower pitch + emphasis; Tired=slower/softer; Focused=slightly faster; Calm=all neutral.

4. **CorrectionTuple system created** — Typed artifact for user correction → LoRA fine-tuning pipeline:
   - `MMO/Core/CorrectionTuple.hpp` — **NEW FILE**. Defines `CorrectionTuple` struct (rejected proposal + user correction + NLP/routing context + signal type) and `CorrectionTupleCollector` singleton (collect, markConfirmed, flush to JSONL, clear).
   - `MMO/Core/CorrectionTuple.cpp` — **NEW FILE**. Implements collection (builds full tuple from episode + context), confirmation marking, JSONL append-mode export with complete serialization.
   - `SessionContextManager.cpp` — `recordUserResponse()` now auto-collects a correction tuple when `rejected == true`, capturing tool_id, args, confidence, risk, correction_text, NLP summary, router tags, route, and mood from the current session state.
   - `Orchestrator.cpp` — `shutdown()` now flushes pending correction tuples to `correction_tuples.jsonl` before unloading models.

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Subject tags in router metadata | `RouterMetadataBuilder::build()` → `getSubjectTags()` |
| Language detection | `detectLanguage()` in `NlpAnnotation.cpp` |
| NLP rule persistence | `NLP::loadLearnedRules()` / `saveLearnedRules()` via `UnifiedMemoryStorage` |
| Outward emotion expression | `EmotionPresentationController::currentPresentation()` |
| Internal mood state | `PersonalityManager` (unchanged — EPC reads, does not own) |
| Correction data for LoRA | `CorrectionTupleCollector` (auto-collected on rejection) |
| Correction flush at shutdown | `Orchestrator::shutdown()` |

**Canonical integration points**

- `RouterMetadata::subject_tags` — available in router metadata JSON for route decisions
- `EmotionPresentationController::instance().currentPresentation()` — get outward expression for current mood
- `EmotionPresentationController::instance().setEnabled(false)` — suppress all outward emotion
- `CorrectionTupleCollector::instance().flush(path)` — export pending corrections to JSONL
- `CorrectionTupleCollector::instance().markConfirmed(session_id, turn_id)` — mark correction as successfully executed

**Files modified**

- `nlp/RouterMetadataBuilder.hpp` — `subject_tags` field added to `RouterMetadata`
- `nlp/RouterMetadataBuilder.cpp` — `#include MMD.hpp`, `getSubjectTags()` call in `build()`, serialized in `toJson()`
- `nlp/NlpAnnotation.cpp` — `detectLanguage()` function added, replaces hardcoded `"en"`
- `nlp/nlp.cpp` — `loadLearnedRules()` and `saveLearnedRules()` implemented (was TODO stubs)
- `MMO/UI/EmotionPresentationController.hpp` — **NEW FILE**
- `MMO/UI/EmotionPresentationController.cpp` — **NEW FILE**

### 28. Legacy NLP decoupling and conversation history wiring

**What changed**

- Removed direct `ActionExecutor::isActionCommand()` bypass that skipped ActionPolicy gate — action commands now flow through the same policy evaluation as all other commands
- Deleted `g_nlp.learnPattern()` calls from `ai_interpret()` — NLP teaching was legacy coupling; `ToolTrainingParser` now records all training examples via MMO pipeline
- Deleted `FastClassifier::boostCommandWeight()` calls from `ai_interpret()` — weight boosting was ad-hoc learning superseded by ToolTraining data
- Deleted dead `prepareConversationMessages()` function — replaced by inline SessionContextManager recording
- Removed dead includes: `nlp/nlp.hpp`, `fast_classifier.hpp`, `<sstream>` from `ai/ai.cpp`
- Fixed bare `#include "commands_core.hpp"` in `ai/ai_rl.hpp`, `ai/ai_reward.hpp`, `ai/ai_rl.cpp` → `"commands/commands_core.hpp"` (resolved Clang include-path failures cascading through main.cpp)
- Wired conversation history into `ai_interpret()`: user message recorded at entry via `SessionContextManager::addMessage("user")`; assistant responses recorded at command-execution and conversation exit paths via `addMessage("assistant")`
- System prompt now set via `SessionContextManager::setSystemPrompt()` on each `ai_interpret()` call using personality config

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Online learning from AI dispatch | `ToolTrainingParser::record()` — structured examples for LoRA fine-tuning |
| Action command policy gating | `ActionPolicyRegistry::evaluate()` in `ai_interpret()` — same gate as all commands |
| Action command execution | `ActionExecutor::executeAction()` (still used, but now inside policy-gated path) |
| Conversation history storage | `SessionContextManager::addMessage()` in `ai_interpret()` |
| System prompt management | `SessionContextManager::setSystemPrompt()` in `ai_interpret()` (reads `ai_config.json → personality`) |
| History trimming | `SessionContextManager::trimHistory()` — capped by `ai_config.json → conversation_history_size` |

**Deleted legacy paths**

- `extern NLP g_nlp; g_nlp.learnPattern(input, suggested)` — two call sites removed
- `GRIM::FastClassifier::boostCommandWeight(word, 1.5f)` — verb-extraction loop removed
- `prepareConversationMessages()` — dead helper removed
- `try { ... } catch (...) {}` swallowing errors around NLP teaching — deleted (Rule 20)

**Canonical integration points**

- `SessionContextManager::instance().addMessage(session_id, role, content)` — record user/assistant turns
- `SessionContextManager::instance().setSystemPrompt(session_id, prompt)` — idempotent system prompt
- `SessionContextManager::instance().trimHistory(session_id, max)` — keep history bounded
- `ToolTrainingParser::instance().record(example)` — all training data collection

**Validation status**

- Build verified: `cmake --build --preset release` passes with zero errors
- IntelliSense errors in `main.cpp` are IDE artifacts (Clang can't resolve `helpers/widget.hpp` from `ui/ui_panel.hpp` — CMake include dirs handle this at compile time)
- ActionExecutor path now gated by ActionPolicy — action commands no longer bypass training wheels

**Files modified**

- `ai/ai.cpp` — legacy NLP/FastClassifier blocks removed; conversation history wiring added; dead function deleted
- `ai/ai_rl.hpp` — include path fixed
- `ai/ai_reward.hpp` — include path fixed
- `ai/ai_rl.cpp` — include path fixed
- `MMO/Core/CorrectionTuple.hpp` — **NEW FILE**
- `MMO/Core/CorrectionTuple.cpp` — **NEW FILE**
- `MMO/Core/SessionContextManager.cpp` — `#include CorrectionTuple.hpp`, auto-collect on rejection in `recordUserResponse()`
- `MMO/Core/Orchestrator.cpp` — `#include CorrectionTuple.hpp`, flush in `shutdown()`

**Validation status**

- Subject tags now flow: user input → `getSubjectTags()` → `RouterMetadata::subject_tags` → router JSON. Router model can match against `ModelInfo::subject_tags`.
- Language detection is heuristic (character-range counting) — sufficient for body-side classification; router model does its own language handling.
- `loadLearnedRules` / `saveLearnedRules` consume the existing `UnifiedMemoryStorage` API — no new storage schema needed.
- EPC is fully functional but not yet called from any response path — consumers (TTS, UI, avatar) need to query `currentPresentation()`.
- Correction tuples auto-collect on rejection but `markConfirmed()` not yet called after successful correction execution.
- `grammar_parser.cpp` `matchTemplate()` still returns false (placeholder) — deferred to dedicated template parsing design.

**Remaining gaps**

- EPC not yet consumed by UI response rendering, TTS engine, or avatar system
- `markConfirmed()` not called after correction execution — needs wiring in `ai_interpret()` success path
- `flush()` writes to `correction_tuples.jsonl` in working directory — may need configurable output path
- UISurfaceSpec/Registry still planned only
- Shadow mode still planned only
- 3-buffer memory rotation still planned only

---

## §36 — Correction Loop Wiring, EPC Consumer, UISurfaceSpec, Shadow Mode

**What changed**

1. **Correction loop fully wired** — `processFeedbackResponse()` in `commands_feedback.cpp` now calls `SessionContextManager::recordUserResponse()` on all feedback paths (positive confirmation, negative rejection, low-confidence correction). `CorrectionTupleCollector::markConfirmed()` is called on positive feedback using the current turn_id from snapshot. `ai_interpret()` now calls `SessionContextManager::recordProposal()` when policy requires verification, so the `ActionEpisode` exists for subsequent correction tuple collection on rejection.

2. **EPC wired into response path** — `commands_core.cpp` `handleCommand()` now calls `EmotionPresentationController::instance().currentPresentation()` for conversational/banter/question responses and prepends `text_prefix` to `finalText` before output. Voice parameters (pitch, rate, emphasis) are available in the `EmotionPresentation` struct but `Voice::speak()` doesn't accept them yet — voice modulation deferred to TTS system upgrade.

3. **UISurfaceSpec + UISurfaceRegistry created** — `MMO/UI/UISurfaceSpec.hpp/.cpp` defines the validated UI surface contract: `SurfaceKind` (OverlayPanel/Popup/Modal/Toast/ToolWindow/Inspector), `LifetimePolicy`, `VisibilityState`, `InputPolicy`, `WidgetSpec`, `LayoutSpec`. `validateSurfaceSpec()` enforces constraints (Modal requires Exclusive input, Toast requires AutoDismiss, etc.). `MMO/UI/UISurfaceRegistry.hpp/.cpp` is the runtime registry with create/update/show/hide/destroy lifecycle, unique surface_id keying, event callbacks, and query methods.

4. **Shadow mode implemented** — `ai_interpret()` checks `ModelRegistry::mode() == "shadow"` after policy allows a command. In shadow mode, the proposed command is logged but not executed; user sees `[Shadow] I would run: <cmd>`. Config key `mmo.mode` already existed in `ai_config.json` with value `"shadow"`.

**Ownership boundaries**

| Concern | Owner | Hook point |
|---------|-------|------------|
| Correction tuple collection | `SessionContextManager::recordUserResponse()` | auto-collects via `CorrectionTupleCollector::collect()` |
| Correction confirmation | `processFeedbackResponse()` | calls `CorrectionTupleCollector::markConfirmed()` |
| Action proposal tracking | `ai_interpret()` | calls `SessionContextManager::recordProposal()` before setPending |
| EPC text injection | `handleCommand()` in `commands_core.cpp` | reads `currentPresentation().text_prefix` |
| UI surface lifecycle | `UISurfaceRegistry` singleton | create/show/hide/destroy with event callbacks |
| Shadow mode gate | `ai_interpret()` in `ai.cpp` | checks `ModelRegistry::mode()` before command execution |

**Files touched**

- `ai/ai.cpp` — `#include ModelRegistry.hpp`, `recordProposal()` call in verification path, shadow mode check before command execution
- `commands/commands_feedback.cpp` — `#include CorrectionTuple.hpp`, `recordUserResponse()` + `markConfirmed()` in positive/negative/correction feedback paths
- `commands/commands_core.cpp` — `#include EmotionPresentationController.hpp`, EPC `text_prefix` injection for conversational responses
- `MMO/UI/UISurfaceSpec.hpp` — **NEW FILE** (surface spec types + validation declaration)
- `MMO/UI/UISurfaceSpec.cpp` — **NEW FILE** (validation implementation)
- `MMO/UI/UISurfaceRegistry.hpp` — **NEW FILE** (registry declaration + event callback types)
- `MMO/UI/UISurfaceRegistry.cpp` — **NEW FILE** (registry implementation)

**Validation status**

- Correction loop is end-to-end: propose → setPending → user responds → processFeedbackResponse → recordUserResponse (triggers CorrectionTupleCollector) or markConfirmed. All three feedback response paths (positive, negative, low-confidence correction) are wired.
- EPC text_prefix is injected for `conversation`, `banter`, and `question` category responses. Other categories (command execution, verification, etc.) show neutral text.
- UISurfaceRegistry enforces spec constraints at create/update time. Modals require Exclusive input, Toasts require AutoDismiss + positive timeout.
- Shadow mode is config-driven (`mmo.mode: "shadow"` vs `"enforced"`). In enforced mode (or any non-shadow value), commands execute normally.
- CMake glob_recurse auto-includes new UI files.

**Remaining gaps**

- EPC voice parameters not consumed — `Voice::speak()` needs pitch/rate/emphasis parameters
- EPC avatar expression not consumed — no avatar rendering system exists yet
- UISurfaceRegistry not yet wired to ToolRegistry as UI tools (ui.create_surface, etc.)
- UISurfaceRegistry not consumed by any UI renderer — need host/renderer integration
- `flush()` output path still hardcoded — needs configurable output path
- 3-buffer memory rotation still planned only

---

## §37 — UISurfaceRegistry bootstrap wiring

**What changed**

- `bootstrap/bootstrap.hpp` — Added `#include "../MMO/UI/UISurfaceRegistry.hpp"` so the type is visible to bootstrap consumers
- `bootstrap/bootstrap.cpp` — In `bootstrapMMOLayer()`, after `SessionContextManager ready` log, added `UISurfaceRegistry::instance()` touch + readiness log
- `main.cpp` — Added `#include "MMO/UI/UISurfaceRegistry.hpp"`. Added `GRIM::MMO::UISurfaceRegistry::instance().destroyAll()` call to all 3 shutdown paths (console signal handler, `CTRL_CLOSE_EVENT` handler, normal exit cleanup), positioned after ProcessManager teardown and before ResourceCoordinator cleanup

**Owns**

- UISurfaceRegistry singleton initialization at bootstrap time
- UISurfaceRegistry teardown (destroyAll) at all shutdown paths

**Does not own**

- UISurfaceRegistry class implementation (owned by `MMO/UI/UISurfaceRegistry.cpp`)
- UI surface creation/update (owned by future tool handlers or ToolRegistry-backed UI tool commands)
- SurfaceChangeCallback registration for rendering (owned by UI renderer, not yet wired)

**Canonical hook points**

- `bootstrapMMOLayer()` — singleton alive before any tool/command could reference it
- All 3 `main.cpp` shutdown paths — `destroyAll()` fires Destroyed events for each surface, clearing the registry
- `UISurfaceRegistry::instance().onSurfaceChange(callback)` — renderers hook here (deferred to renderer integration)

**Current consumers**

- `bootstrap/bootstrap.cpp` — touches singleton for initialization
- `main.cpp` — calls `destroyAll()` on all shutdown paths

**Legacy replaced**

- None (new system)

**Status**

- Implemented; singleton lifecycle managed (init at bootstrap Phase 4, cleanup at shutdown)

---

## §38 — UI tools registered in ToolRegistry

**What changed**

- `bootstrap/bootstrap.cpp` — After `UISurfaceRegistry` is alive in `bootstrapMMOLayer()`, 4 UI tool descriptors are registered in `ToolRegistry`:
  - `ui.create_surface` — Create a new UI surface (needs_confirmation=true). Parameters: surface_id, kind, title. Capability tags: ui, create_surface, display.
  - `ui.show_surface` — Show a previously created surface. Parameters: surface_id. Capability tags: ui, show_surface.
  - `ui.hide_surface` — Hide a visible surface without destroying it. Parameters: surface_id. Capability tags: ui, hide_surface.
  - `ui.destroy_surface` — Permanently remove a surface (needs_confirmation=true). Parameters: surface_id. Capability tags: ui, destroy_surface.
- All 4 tools have `category = "ui"`, `is_informational = false`, `provider_type = Builtin`.

**Owns**

- ToolRegistry registration of UI surface tool entries at bootstrap

**Does not own**

- Command dispatch for these tools (command handlers not yet implemented)
- UISurfaceRegistry lifecycle (§37)
- UI rendering (deferred to renderer integration)

**Canonical hook points**

- Model can discover these tools via `ToolRegistry::generateCompactPrompt()` / `generateFullPrompt()`
- `ToolGapPlanner::evaluate("ui.create_surface", ...)` returns `nullopt` — tool exists
- When command handlers are wired, dispatch will route `ui.create_surface surface_id` to `UISurfaceRegistry::create()`

**Current consumers**

- `ToolRegistry::generateCompactPrompt()` — includes UI tools in model-facing tool summary

**Status**

- Tool descriptors registered; command handlers wired in §39

---

## §39 — UI tool command handlers wired to dispatch

**What changed**

- `commands/commands_ui.hpp` — Declared 4 new handlers: `cmdCreateSurface`, `cmdShowSurface`, `cmdHideSurface`, `cmdDestroySurface`
- `commands/commands_ui.cpp` — Implemented all 4 handlers:
  - `cmdCreateSurface(arg)` — Parses JSON arg (`surface_id`, `kind`, `title`, optional `host_target`, `auto_dismiss_ms`), maps `kind` string → `SurfaceKind` enum, validates via `UISurfaceRegistry::create()`.
  - `cmdShowSurface(arg)` — Accepts plain `surface_id` string or JSON `{"surface_id":"..."}`, calls `UISurfaceRegistry::show()`.
  - `cmdHideSurface(arg)` — Same arg pattern, calls `UISurfaceRegistry::hide()`.
  - `cmdDestroySurface(arg)` — Same arg pattern, calls `UISurfaceRegistry::destroy()`.
- `plugins/core_plugin.cpp` — Registered all 4 as `REGISTER_TOOL("ui.create_surface", ...)` etc. in the UI Controls block, between `toggle_settings` and `system_info`. This inserts them into both `commandMap` (for `dispatchCommand` lookup) and `GRIM::CommandRegistry` (for metadata).

**Owns**

- Command handler implementations for UI surface CRUD
- String→`SurfaceKind` enum mapping (`parseSurfaceKind`)
- JSON arg parsing for surface creation parameters

**Does not own**

- ToolRegistry descriptors (§38, bootstrap)
- UISurfaceRegistry lifecycle (§37, bootstrap + shutdown)
- SurfaceChangeCallback rendering (deferred)
- ActionPolicyRegistry gating (handled by `ai_interpret()` before dispatch)

**Canonical hook points**

- `dispatchCommand("ui.create_surface", jsonArg)` — creates surface via registry
- `dispatchCommand("ui.show_surface", surfaceId)` — shows surface
- `dispatchCommand("ui.hide_surface", surfaceId)` — hides surface
- `dispatchCommand("ui.destroy_surface", surfaceId)` — destroys surface
- AI flow: model output → `ai_interpret()` → `ActionPolicyRegistry.evaluate()` → `dispatchCommand()` → handler → `UISurfaceRegistry`

**Current consumers**

- `dispatchCommand()` in `commands_core.cpp` — routes by `commandMap` lookup
- `ai_interpret()` in `ai/ai.cpp` — dispatches model-suggested tools through policy gate
- `ToolRegistry` analytics — `recordSuccess`/`recordFailure` called by dispatch wrapper

**Legacy replaced**

- None (new system)

**Status**

- Fully wired: tool descriptors (§38) + command handlers (§39) + dispatch registration (core_plugin.cpp)

---

## §40 — EPC voice prosody wired to TTS pipeline

**What changed**

- `voice/voice_speak.hpp` — Added `Voice::VoiceParams` struct (`pitch`, `rate`, `emphasis` floats) and new overload `Voice::speak(text, category, params)`.
- `voice/voice_speak.cpp` — Queue item changed from `pair<string,string>` to `SpeakItem{text, category, VoiceParams}`. Worker applies `voice_rate` as speed multiplier (`effectiveSpeed = g_speed * params.rate`). Old `speak(text, category)` overload delegates to new one with default `VoiceParams{}`.
- `commands/commands_core.cpp` — Main `handleCommand()` output path now fetches `EmotionPresentationController::currentPresentation()` and passes `voice_pitch`/`voice_rate`/`voice_emphasis` into `Voice::speak()` via `VoiceParams`.

**Owns**

- `VoiceParams` struct definition
- Mapping from `EmotionPresentation` voice fields → TTS speed/pitch
- Queue transport of prosody parameters

**Does not own**

- EmotionPresentationController mood→presentation mapping (EPC owns that)
- TTS bridge protocol extension for pitch (future: add `pitch` key to coquiSpeak JSON request)
- EPC mood selection (PersonalityManager drives that)

**Canonical hook points**

- `Voice::speak(text, category, params)` — any caller can supply prosody
- `speakWorker()` — applies `params.rate` as speed multiplier to `coquiSpeak`
- Future: extend `coquiSpeak` JSON request with `{"pitch": params.pitch}` when TTS bridge supports it

**Status**

- Wired end-to-end: EPC → VoiceParams → speak queue → worker → coquiSpeak speed. Pitch passthrough to TTS bridge pending (bridge needs server-side support).

---

## §41 — Configurable correction output path

**What changed**

- `MMO/Core/Orchestrator.hpp` — Added `std::string correction_output_path = "correction_tuples.jsonl"` to `OrchestratorConfig`.
- `MMO/Core/Orchestrator.cpp` — `shutdown()` now uses `config_.correction_output_path` instead of hardcoded `"correction_tuples.jsonl"`.
- `bootstrap/bootstrap.cpp` — Loads `correction_output_path` from `ai_config.json → mmo.orchestrator.correction_output_path` with default fallback.

**Owns**

- Config field for correction output path
- Loading from JSON config

**Does not own**

- CorrectionTupleCollector implementation (unchanged)
- Orchestrator lifecycle (unchanged)

**Canonical hook points**

- `ai_config.json → mmo.orchestrator.correction_output_path` — set custom path
- `OrchestratorConfig.correction_output_path` — read by `Orchestrator::shutdown()`

**Status**

- Implemented; defaults to `"correction_tuples.jsonl"` when config key absent

---

## §42 — ToolTrainingParser (Phase 1.0)

**What changed**

- `MMO/Core/ToolTrainingParser.hpp` — New file. `ParsedModelOutput` struct (valid flag, intent, suggested_command, response_text, extracted_json, parsed JSON object). `ToolTrainingExample` struct (full interaction record: input, model output, parsed fields, outcome enum, confidence signals). `ToolTrainingParser` singleton class with `parseModelOutput()`, `record()`, `flush()`, `clear()`, `pendingCount()`.
- `MMO/Core/ToolTrainingParser.cpp` — Implementation. `findBalancedJson()` uses brace-depth counting with string-literal awareness (handles escaped quotes, nested objects/arrays). `parseModelOutput()` replaces the naive first-`{`-last-`}` extraction. `flush()` appends JSONL to disk (same pattern as CorrectionTupleCollector). `exampleToJson()` serializes input/model_output/outcome/confidence.
- `ai/ai.cpp` — Replaced naive JSON extraction block with `ToolTrainingParser::parseModelOutput()`. Added `#include "ToolTrainingParser.hpp"`. Every exit path in `ai_interpret()` now records a `ToolTrainingExample` with the appropriate outcome (ParseFailure, GapDetected, PolicyDenied, PolicyVerify, ShadowSuppressed, Success, Failure, Conversation).
- `MMO/Core/Orchestrator.cpp` — `shutdown()` now flushes `ToolTrainingParser::instance().flush("tool_training_examples.jsonl")` alongside CorrectionTuples.

**Owns**

- Robust brace-depth JSON extraction from model output
- `ToolTrainingExample` data contract (full-spectrum training signal)
- JSONL export of all model interactions (positive + negative)
- `ParsedModelOutput` as the canonical parse result type

**Does not own**

- CorrectionTupleCollector (captures rejections with user corrections — complementary, not replaced)
- Model prompt construction (still in `ai_interpret()`)
- ToolGapPlanner evaluation (called by `ai_interpret()`, feeds outcome into training example)
- ActionPolicyRegistry gating (feeds verdict into training example)

**Canonical hook points**

- `ToolTrainingParser::instance().parseModelOutput(reply)` — replaces raw JSON extraction
- `ToolTrainingParser::instance().record(example)` — called at each `ai_interpret()` exit
- `ToolTrainingParser::instance().flush(path)` — called from `Orchestrator::shutdown()`
- `tool_training_examples.jsonl` — output file (append mode, one JSON object per line)

**Relationship to CorrectionTupleCollector**

- CorrectionTuples capture **user corrections** (negative signal with corrected tool_id)
- ToolTrainingExamples capture **all interactions** (positive signal from successes, negative from failures/gaps/denials, neutral from conversations)
- Together they provide a complete LoRA fine-tuning dataset

**Status**

- Phase 1.0 complete. Robust parser wired, all exit paths recorded, JSONL flushed at shutdown.

---

## §43 — UI Renderer Integration (UISurfaceRendererBridge)

**What changed**

- Added `UIRoot::removePanel(const std::string& name)` — safe panel removal from `m_panelMap` + `m_panels` vector, wrapped in `postTask` for thread safety.
- Created `DynamicSurfacePanel` (`ui/dynamic_surface_panel.hpp`, `ui/dynamic_surface_panel.cpp`) — `UIPanel` subclass that takes a `UISurfaceSpec` and renders its widgets via `OverlayRenderer`. Maps `SurfaceKind` to panel properties (draggable, resizable, chrome). Renders label/button/progress widget types.
- Created `UISurfaceRendererBridge` (`ui/ui_surface_renderer_bridge.hpp`, `ui/ui_surface_renderer_bridge.cpp`) — static `install()` registers a `SurfaceChangeCallback` on `UISurfaceRegistry::instance()`. Routes all UIRoot mutations through `postTask` for thread safety.
- Wired `UISurfaceRendererBridge::install()` in `main.cpp` after UIRoot and panels are initialized (after line 386).

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Surface lifecycle events | `UISurfaceRegistry` (unchanged) |
| Event → panel mapping | `UISurfaceRendererBridge::install()` callback |
| Dynamic panel rendering | `DynamicSurfacePanel::drawOverlay()` |
| Panel add/remove/visibility | `UIRoot` (extended with `removePanel`) |

**Canonical integration points**

- `UISurfaceRendererBridge::install()` — call once after `UIRoot::get().init()` and `UISurfaceRegistry::instance()` are both live
- `DynamicSurfacePanel::updateSpec(spec)` — called on `SurfaceEvent::Updated` to re-apply spec changes
- `UIRoot::get().removePanel(name)` — called on `SurfaceEvent::Destroyed`

**Event → action mapping**

| SurfaceEvent | Bridge action |
|-------------|---------------|
| Created | `make_shared<DynamicSurfacePanel>(spec)` → `UIRoot::addPanel()` |
| Shown | `UIRoot::setVisible(id, true)` |
| Hidden | `UIRoot::setVisible(id, false)` |
| Updated | `panel->updateSpec(newSpec)` |
| Destroyed | `UIRoot::removePanel(id)` |

**Thread safety**

All UIRoot mutations routed through `UIRoot::get().postTask()` since `SurfaceChangeCallback` fires under `UISurfaceRegistry`'s mutex (potentially from any thread).

**Status**

- Phase 1.0 complete. Bridge wired, all 5 surface events handled, DynamicSurfacePanel renders label/button/progress widgets.

---

## §44 — 3-Buffer Memory Rotation + Atomic Writes

**What changed**

- Created `AtomicWriter` (`memory/atomic_writer.hpp`) — static utility for write-temp + rename. Three overloads: `write(path, data, size)`, `writeString(path, content)`, `writeWith(path, callback)`. Creates parent directories, writes to `.tmp`, flushes, then `std::filesystem::rename` for atomicity.
- Created `MemoryBufferRotation` (`memory/memory_buffer_rotation.hpp`, `memory/memory_buffer_rotation.cpp`) — singleton 3-buffer pipeline:
  - **Working** — active context for current session
  - **Preprocessing** — incoming data staged here via `preprocess(obj)`
  - **SyncAndClear** — `syncToLongTerm(storage)` moves working → sync, persists to long-term, clears
  - `mergeToWorking()` drains preprocessing into working
  - `workingSnapshot()` returns a copy of working buffer for building precomposed context
  - Thread-safe under mutex
- Modified `UnifiedMemoryStorage::saveToDisk()` — replaced raw `ofstream` with `AtomicWriter::writeString()`. Inference never reads partially-written JSON.
- Modified `UnifiedMemoryStorage::saveToFlatBuffer()` — replaced raw `ofstream` with `AtomicWriter::write()`. Same atomic guarantee for FlatBuffer format.
- Modified `ai/ai.cpp::saveMemory()` — replaced raw `ofstream` with `GRIM::AtomicWriter::writeString()`. The global `memory.json` is now crash-safe.
- Added `#include "atomic_writer.hpp"` to `unified_memory_storage.cpp` and `ai/ai.cpp`.

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Atomic file writes | `GRIM::AtomicWriter` (header-only utility) |
| 3-buffer rotation | `GRIM::MemoryBufferRotation` singleton |
| Long-term persistence | `UnifiedMemoryStorage::saveToDisk/saveToFlatBuffer` (now atomic) |
| Legacy `memory.json` writes | `saveMemory()` in `ai/ai.cpp` (now atomic) |

**Canonical integration points**

- `MemoryBufferRotation::instance().preprocess(obj)` — stage incoming memory
- `MemoryBufferRotation::instance().mergeToWorking()` — drain preprocessing into working
- `MemoryBufferRotation::instance().syncToLongTerm(storage)` — persist and clear
- `MemoryBufferRotation::instance().workingSnapshot()` — read working context for precomposed metadata
- `GRIM::AtomicWriter::writeString(path, content)` — atomic text file write
- `GRIM::AtomicWriter::write(path, data, size)` — atomic binary file write

**Relationship to existing memory system**

- `UnifiedMemoryStorage` remains the long-term store (FlatBuffer + JSON)
- `MemoryBufferRotation` sits in front as the short-term pipeline
- Orchestrator will call `preprocess` for incoming data, `mergeToWorking` before building context, and `syncToLongTerm` periodically or at shutdown

**Status**

- Phase 1.0 complete. 3-buffer pipeline implemented, all file writes now atomic.

---

## §45 — TTS Bridge Pitch Input Port

**What changed**

- `voice_speak.hpp` — Added `double pitch = 1.0` parameter to `coquiSpeak()` (default preserves all existing callers).
- `grim_exports.hpp` — Same signature change on the DLL export declaration.
- `voice_speak.cpp` — `speakWorker()` now reads `item.params.pitch` into `effectivePitch` and passes it to `coquiSpeak()`.
- `voice_speak.cpp` — `coquiSpeak()` implementation accepts `pitch` and includes `{"pitch", pitch}` in the JSON request sent to the Python bridge.

**What was NOT changed**

- `coqui_bridge.py` — Python side ignores unknown JSON keys, so `pitch` is a no-op until the TTS model handles it.
- Existing callers (`commands_voice.cpp`, `preCacheCommonPhrases`) — unchanged, default `pitch=1.0` applies.

**Data flow**

```
EPC VoiceParams.pitch → speakWorker effectivePitch → coquiSpeak(pitch) → JSON {"pitch": 1.2} → bridge stdin
```

**Status**

- Input port wired end-to-end. Python-side pitch handling deferred until custom TTS model is ready.

---

## §46 — MemoryBufferRotation Consumer Wiring

**What changed**

- `commands/commands_memory.cpp` — `cmdRemember()` now stages via `MemoryBufferRotation::instance().preprocess(obj)` instead of direct `g_memoryStorage.storeLongTerm(obj)`. Added `#include "memory/memory_buffer_rotation.hpp"`.
- `commands/commands_question.cpp` — `storeContextMemory()` now stages via `preprocess()` instead of direct `storeLongTerm()`. Added `#include "memory/memory_buffer_rotation.hpp"`.
- `commands/commands_core.cpp` — After `rememberContextObject()`, also calls `preprocess()` to stage the context object in the rotation pipeline. Added `#include "memory/memory_buffer_rotation.hpp"`.
- `MMO/Core/Orchestrator.cpp` — `buildRouterScope()` calls `MemoryBufferRotation::instance().mergeToWorking()` before building scope JSON. This ensures all preprocessed memory is available in the working context before requests. Added `#include "../../memory/memory_buffer_rotation.hpp"`.
- `main.cpp` — All three shutdown paths (Win32 CTRL_C handler, Unix signal handler, normal shutdown §13) now flush the rotation pipeline before MMO teardown: `mergeToWorking()` → `syncToLongTerm(g_memoryStorage)` → `clear()`. Added `#include "memory/memory_buffer_rotation.hpp"`.

**Data flow**

```
cmdRemember / storeContextMemory / rememberContextObject
  → preprocess(obj) → preprocessing_ buffer
  → (on next AI request) mergeToWorking() drains into working_ buffer
  → (on shutdown) syncToLongTerm() persists working_ → g_memoryStorage
```

**What now owns the concern**

| Concern | Owner |
|---------|-------|
| Staging incoming memory | `MemoryBufferRotation::preprocess()` |
| Draining to working context | `Orchestrator::buildRouterScope()` triggers `mergeToWorking()` |
| Persisting to long-term | All shutdown paths trigger `syncToLongTerm()` |
| Working context reads | `MemoryBufferRotation::workingSnapshot()` (available for future consumers) |

**Does not own**

- Long-term storage format or indexing (owned by `UnifiedMemoryStorage`)
- MemoryFacade retrieval API (unchanged, still reads from `UnifiedMemoryStorage`)
- Session context objects (still managed by `SessionContextManager` independently)

**Canonical hook points**

- `GRIM::MemoryBufferRotation::instance().preprocess(obj)` — all new memory objects
- `GRIM::MemoryBufferRotation::instance().mergeToWorking()` — called in `buildRouterScope()`
- `GRIM::MemoryBufferRotation::instance().syncToLongTerm(g_memoryStorage)` — all shutdown paths

**Status**

- Pipeline fully wired. Incoming memory flows through preprocessing → working → long-term on shutdown.
