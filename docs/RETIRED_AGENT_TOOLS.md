# Retained AI, NLP, and administration handlers

On October 29, 2026, eleven legacy commands were removed from the built-in
agent tool registrations in `plugins/core_plugin.cpp`. At that point, their implementations,
declarations, and underlying services remained available in the host source.
They are not migrated plugins and are not available through normal registry
discovery or command dispatch unless explicitly registered again.

The model's tool catalog should expose concrete task capabilities. Developer
diagnostics, grammar maintenance, session administration, and generic calls back
into answer generation do not need to be model-selectable tools. Removing these
registrations does not remove the normal conversation or model-routing paths.
The static utility `help` response also stops advertising `ai_backend` and the
historical `reloadnlp` spelling.

On November 17, 2026, the implementations of `question`, `nlp_learn`, and
`nlp_save` were removed. The NLP diagnostic still checks parsing and statistics.
Dynamic rule authoring and persistence were removed, while static and plugin
rule support and the memory file format were preserved.

These are source changes. An existing executable retains its old registrations
until the host is rebuilt, and a deployed utility DLL retains its old help text
until that plugin is rebuilt and replaced. No binaries are rebuilt or deployed
as part of this retirement change.

The original purposes below are inferred from implementation comments, handler
behavior, and command help; they are not a claim about undocumented design history.
Future uses are possibilities, not implemented integrations or commitments.

| Former tool / retained handler | What it does today | Apparent original purpose | Why retained / possible future use |
|---|---|---|---|
| `reload_nlp` / `cmdReloadNlp` | Calls `reloadNlpRules()` and adjusts result category and voice text. | Reload language rules without restarting the application. | Explicit host/developer grammar reload control. |
| `clear_context` / `cmdClearContext` | Calls `clearConversationHistory()`, which clears the default session's history. It is not a reset of every memory or session. | Start a fresh conversation through a command. | User-operated session reset; a future multi-session control would need an explicit session identity. |
| `grim_ai` / `cmdGrimAi` | Passes input to `ai_process`, supplies default presentation fields, and maps failures through ErrorManager. | Direct/catch-all entry point into AI processing. | Host console or integration adapter; underlying AI processing remains needed for conversation. |
| `test_intent` / `cmdTestIntent` | Prints fixed input/expected-intent cases with pending-implementation status. It does not run a classifier. | Scaffold for classifier diagnostics. | Reference cases for a real developer test harness; current output must not be treated as measured accuracy. |
| `test_nlp` / `cmdTestNlp` | Exercises grammar statistics, parsing, contextual parsing, and fuzzy matches. Records parse successes. | Interactive grammar regression checks. | Material for isolated developer tests. It updates live parser statistics. |
| `nlp_stats` / `cmdNlpStats` | Reads `g_nlp.getStats()` and formats rule counts and parse statistics. | Inspect legacy parser coverage and behavior. | Developer diagnostics or a host grammar inspector. |
| `settings` / `cmdSettings` | Prints a resource/config path, static command guidance, and hardcoded quick-setting values. It does not inspect complete current configuration. | Temporary command-based settings/help surface; source labels it a stub. | Historical UI/reference material. A future configuration inspector should read actual state rather than reuse these values as facts. |
| `ai_backend` / `cmdAiBackend` | Reads or assigns the in-memory `aiConfig["backend"]` string. It does not validate or persist the supplied value. AI routing reads this key, including direct `grim_native` and `ollama` paths. | Command-based backend selection. | User/developer routing controls; a future interface should validate supported selections and define persistence explicitly. |

## Remaining source and ownership

- `commands/commands_ai.cpp` and `.hpp`: backend, context, NLP reload, and AI entry handlers.
- `commands/commands_system.cpp` and `.hpp`: diagnostics, grammar maintenance, and settings stub.
- `ai/ai.cpp`: underlying AI routing and default-session history clearing.
- `nlp/`: grammar implementation and shared state.

No replacement API services or plugin registrations were added for these eleven
names. Existing learned routes or legacy grammar strings that refer to a retired
name do not restore its registry handler. Any future reintroduction should define
the intended consumer, actual metadata, state ownership, and side effects first.

## Removed disconnected host paths

On November 8, 2026, the separate dead-code cleanup removed 29 source/header
files: the old feedback and command-execution adapters, host PPO/reward adapters,
training-server manager and training-command adapters, ActionExecutor parser,
LM intent bridge, proactive dialogue, unused grammar-test command and command
helpers, style loader, orphan perception/capability types, and empty alerts files.

Their PPO bootstrap/shutdown hooks, unused training-manager shutdown hook, and
wake-voice pending-feedback capture branch were removed together. Normal wake
capture still calls [handleCommand](../commands/commands_core.cpp); explicit
tool dispatch uses [ToolRegistry::invoke](../MMO/Core/Registries/ToolRegistry.hpp). Grounded click failure tests now
exercise the [host input service](../perception/input_command_service.cpp).

That cleanup left all eleven handlers intact. Underlying conversation,
grammar, learned-memory storage, session context, configuration UI, input, and
voice services remain. The exported [verification reward API](../ai/verification_reward.hpp)
and standalone Python/control-server sources were not deletion targets.
Action-policy enforcement and classifier initialization remain separate
integration questions; this cleanup does not connect them.

Only focused tests and static diagnostics validate this cleanup. The host
executable and model/training runtimes are not rebuilt or deployed by it.

## Perception ownership break (November 13, 2026)

API v9 removes the retained question handler's automatic vision-question branch
and its direct OCR/detection helpers. Its unrelated conversation, memory, OSINT,
and generation logic remains; the retired `question` tool is not re-registered.
Perception now exposes explicit plugin-owned observation tools. Legacy
`PerceptionContextManager`, host capture/engine lifecycle, automatic visual-memory
writers, and session projectors were removed rather than adapted. Input devices
are accessed independently through `core/input/input_service.cpp`.

## Voice ownership break (November 13, 2026)

Voice now owns its runtime in `plugins/voice_plugin`. Streaming and wake detection
no longer call AI/conversation handlers. Normal typed conversation handling and
retained AI/NLP handlers remain. `handleCommand` no longer automatically speaks
its result; speech must be requested explicitly through the voice provider.

## OSINT and input ownership break (November 13, 2026)

API v11 removes the host input primitive and moves native execution into
`plugins/input_plugin`. Pure targeting geometry and checked-click helpers remain
under `core/input`; `click_on` stays retired. Host UI keybindings remain host-owned.

OSINT implementation now lives exclusively in `plugins/osint_plugin`. Before its
November 17 removal, the retained question handler invoked `osint_report` through ordinary tool dispatch and consumed
its copied result. The handler had dropped its direct collector dependency and
separate 10-second configuration.
The retained OSINT panel consumes copied findings from `osint_show_secrets` rather
than reading provider cache files. Its `osint_show_ui` handler remains unregistered.
