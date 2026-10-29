# Retained AI, NLP, and administration handlers

On October 29, 2026, eleven legacy commands were removed from the built-in
agent tool registrations in `plugins/core_plugin.cpp`. Their implementations,
declarations, and underlying services remain available in the host source.
They are not migrated plugins and are not available through normal registry
discovery or command dispatch unless explicitly registered again.

The model's tool catalog should expose concrete task capabilities. Developer
diagnostics, grammar maintenance, session administration, and generic calls back
into answer generation do not need to be model-selectable tools. Removing these
registrations does not remove the normal conversation or model-routing paths.
The static utility `help` response also stops advertising `ai_backend` and the
historical `reloadnlp` spelling.

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
| `test_nlp` / `cmdTestNlp` | Exercises grammar statistics, parsing, learning, contextual parsing, and fuzzy matches. Records parse successes and learns a sample pattern. | Interactive grammar regression checks. | Material for isolated developer tests. It changes live grammar/statistics and should not be treated as a read-only diagnostic. |
| `nlp_stats` / `cmdNlpStats` | Reads `g_nlp.getStats()` and formats rule counts and parse statistics. | Inspect legacy parser coverage and behavior. | Developer diagnostics or a host grammar inspector. |
| `nlp_learn` / `cmdNlpLearn` | Parses a `user input -> intent_name` string and calls `g_nlp.learnPattern`. | Teach command-language patterns interactively. | Explicit host grammar authoring, if still desired; separate from model training and from installing executable tools. |
| `nlp_save` / `cmdNlpSave` | Calls `g_nlp.saveLearnedRules(g_memoryStorage)`. | Persist learned grammar rules. | Persistence action for a future host grammar editor. |
| `settings` / `cmdSettings` | Prints a resource/config path, static command guidance, and hardcoded quick-setting values. It does not inspect complete current configuration. | Temporary command-based settings/help surface; source labels it a stub. | Historical UI/reference material. A future configuration inspector should read actual state rather than reuse these values as facts. |
| `question` / `cmdQuestion` | Runs `QuestionHandler::process`: keyword-selected perception/OSINT/tool knowledge, memory searches, external lookup, and AI generation paths. Successful answers can trigger conditional memory recording; output includes a shortened voice response. | A preassembled natural-language question-answering pipeline. | Legacy behavior/reference and potential explicit host integrations. For an agent, retrieval, observation, and external lookup should be separate capabilities that the model composes. Reuse must account for network, observation, generation, and memory-write effects. |
| `ai_backend` / `cmdAiBackend` | Reads or assigns the in-memory `aiConfig["backend"]` string. It does not validate or persist the supplied value. AI routing reads this key, including direct `grim_native` and `ollama` paths. | Command-based backend selection. | User/developer routing controls; a future interface should validate supported selections and define persistence explicitly. |

## Source and ownership

- `commands/commands_ai.cpp` and `.hpp`: backend, context, NLP reload, and AI entry handlers.
- `commands/commands_system.cpp` and `.hpp`: diagnostics, grammar maintenance, and settings stub.
- `commands/commands_question.cpp` and `.hpp`: question pipeline and its helpers.
- `ai/ai.cpp`: underlying AI routing and default-session history clearing.
- `nlp/`: grammar implementation and shared state.

No replacement API services or plugin registrations were added for these eleven
names. Existing learned routes or legacy grammar strings that refer to a retired
name do not restore its registry handler. Any future reintroduction should define
the intended consumer, actual metadata, state ownership, and side effects first.
