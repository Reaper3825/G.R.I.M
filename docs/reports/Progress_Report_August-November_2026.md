# GRIM research progress: August 13–November 13, 2026

**Research question**

> To what extent can small, independently trained specialist models achieve complex goal-directed agentic behavior through natural-language delegation from a shared reasoning controller, while maintaining practical coexistence with user workloads on consumer-grade hardware?

**Progress statement**

Over the past three months, work on GRIM has established much of the experimental infrastructure needed to investigate this question. The current system brings together a structured reasoning contract, reusable task environments, independently checked goals, recurrent model–environment interaction, model-inspection tools, and a modular capability architecture.

The principal contribution at this stage is an instrumented research platform. The implementation provides mechanisms for experiments; it does not yet establish the achievable performance of independently trained specialists or demonstrate acceptable interference with concurrent user workloads. Those are the central empirical questions for the next phase.

This report summarizes the resulting state rather than individual fixes or intermediate designs. “Implemented” refers to repository-supported functionality, not a claim that every path has been validated in an end-to-end model run. Existing infrastructure is included where it supports the research question; not every component originated within this reporting window.

**1. A controlled environment for studying agentic behavior**

World Lab provides reusable, editable task worlds backed by isolated TextWorld workers. Worlds can be associated with curriculum courses, inspected manually, and used for goal-driven episodes.

The TextWorld environment is implemented. General simulation of arbitrary production tools is implemented by the tool registry's reserved sandbox mode.

**2. A shared Determine–Define–Execute–Answer reasoning contract**

A representation contract linking authored training examples to structured inference. Determine identifies the contextualized work; Define introduces useful local terms; Execute performs work or submits an environment action; Answer supplies the final response and, in the environment controller, submits the attempt for evaluation. An optional Update span can express verified progress and remaining work.

Goals, known information, unresolved information, and environment observations are distinguished from generated output.

This example starts in GRIM’s System Hub. It shows an authored demonstration with replayed TextWorld feedback. The observations are shortened to their location paths. [GRIM environment demonstration](../../DataCollection/curated/grim_internal_state_v1/demonstrations/examples.md#grim_internal_state_v1_computer_closed)

```text
<prompt>
Go to the computer room. Do not open the tool catalog.
</prompt>
<environment_context>
Location path:
/
</environment_context>
<determine>
Move to the computer room without opening the tool catalog.
</determine>
<execute>
enter computer
</execute>
<environment_context>
Location path:
/computer
</environment_context>
<answer>
I am in the computer room and did not open the tool catalog.
</answer>
```

This demonstration omits `<define>` because the task needs no local definitions.

**User-facing view**

**User**  
Go to the computer room. Do not open the tool catalog.

**GRIM**  
I am in the computer room and did not open the tool catalog.


**3. Goal following with independent outcome checks**

Goals specify a target state, success criteria, and constraints. In supported TextWorld tasks, private evidence conditions allow the environment to check whether criteria were achieved or constraints violated. The model receives the task requirements and permitted observations, while hidden world facts and grading evidence remain outside its input.

The controller keeps the supplied goal fixed throughout an attempt. Model-generated goals and live-system goal verification remain separate future work.

**4. Recurrent interaction with real feedback**

The environment controller can pause generation when an Execute span closes, submit its command, append the permitted environment response, and continue the same attempt. Answer closes the attempt for grading. Action receipts prevent duplicate submissions and preserve uncertain outcomes instead of silently repeating an action.

The outer loop also supports a scheduled collection of curriculum examples. Each example begins with a fresh world state and generation context.

The controller does not yet implement reinforcement-learning updates, persistent training export of runtime trajectories, or a course in which successive goals inherit one shared world state.

**5. Lenses and the Observatory for inspecting model behavior**

The lens infrastructure exposes token predictions at selected positions and layers using the model's existing output head. Captures include candidate probabilities, uncertainty, available reasoning-span annotations, and model/configuration provenance. Training diagnostics can request lens captures, while the Observatory provides an interactive interface for inference inspection.

Additional inspection supports a selected hidden-state Jacobian row, measuring the sensitivity of one final hidden dimension to earlier hidden states. Identity controls compare readouts with the actual final output to help check the inspection mechanism.

They are diagnostic measurements, not proof that an internal representation has a particular semantic meaning. A fitted Jacobian lens and full training-example replay through the Observatory should not be claimed as completed features.

**6. A more systematic data and curriculum workflow**

ConceptBlock authoring, curriculum courses, configured reasoning spans, and environment assignments provide a common organization for training material. Dataset services support shared catalog access and creation workflows, while supervision policies distinguish supplied context from model-generated targets.

Typed annotations and dedicated atom-handling paths provide infrastructure for representing structured items such as entities and tool identifiers. Trained-specialist reliability on those items remains unverified.

Recurrent demonstration records exist as authoring artifacts, but their native trainer ingestion and persistent runtime-trajectory export remain unfinished.

**7. Modular capabilities and a foundation for specialist orchestration**

Model, provider, and tool registries establish explicit identities and lifecycle boundaries for available capabilities. The host includes model process management and route/sub-model/synthesis orchestration. Plugin-owned capabilities include areas such as filesystem access, calculation, scheduling, memory, voice, input, and UI operations.

A plugin or callable tool is not automatically an independently trained specialist model. Natural-language delegation among trained specialists still needs its own end-to-end demonstration and evaluation.

**8. Live discovery of the system's available capabilities**

The Live System World projects registered providers into a navigable environment with a central hub and provider locations. Availability changes as providers load, change, or unload, and the system exposes navigation information and tool inspection.

Live navigation is implemented; automatically verified live navigation goals and general autonomous operation across these capabilities remain separate integration and evaluation work.

**9. Perception and grounded observations**

Digital and physical perception capabilities have been consolidated into a provider-owned subsystem. Its scope includes screen/UI observations, camera acquisition, visual operators, calibration, tracking, and physical world-state representations.

**10. Resource-aware operation and a more inspectable training platform**

The host includes RAM/VRAM-aware admission decisions, reserve thresholds, deferred loading, and eviction of eligible idle models. Training uses configurable token-budget batching to control activation-memory demand. Lens capture has a separate bounded scratch-memory budget. Checkpoint, configuration, and telemetry work supports more traceable experiments.

The resource controls do not yet establish that GRIM coexists acceptably with normal user activity. That requires measurements of task performance, memory use, latency, and interference under concurrent workloads. Training-memory controls should also be evaluated separately from deployment-time coexistence.

**Planned research and implementation**

| Direction | Intended contribution | Current boundary |
| --- | --- | --- |plan
| Independently trained specialists and natural-language delegation | Test whether a shared controller can compose specialized capabilities into complex behavior | Registry/orchestration foundations exist; comparative learned-system results are still needed |
| Self-consistency | Compare multiple candidate answers or trajectories and use agreement or verified outcomes to improve selection | Planned direction for this report; no dedicated implementation was located in the reviewed material |
| Feedback-driven self-correction | Teach a model to repair a failed attempt using execution and goal-check feedback | Documented proposal; collection, scored groups, RL loss, and optimizer integration remain pending |
| Recurrent training data | Train on action–observation sequences matching runtime interaction | Authoring demonstrations exist; native ingestion and persistent runtime export remain incomplete |
| Stateful multi-goal courses | Study longer-horizon progress across related goals in a shared world | Current scheduled examples reset independently |
| Broader live-system tasks | Transfer controlled goal following to real capabilities and observations | Discovery and capability primitives exist; general verification and controller integration need work |
| Efficient personalization | Adapt behavior without retraining all base parameters | Model-side LoRA math and training checkpoints exist; continuous personalization/publication lifecycle is unfinished |
| Concurrent-workload evaluation | Quantify the trade-off between capability and desktop usability | Resource controls exist; measured coexistence results are still required |

Self-consistency and self-correction should be evaluated separately: the first compares candidates, while the second revises behavior in response to feedback. Both increase compute demand, so any gain should be reported alongside its resource and latency cost.

Principal references:

- [Reasoning and Goal contract](../GRIM_REASONING_DATA_CONTRACT.md), [World Lab](../WORLD_LAB.md), and [Live System World](../LIVE_WORLD.md).
- [Lens implementation and boundaries](../../resources/models/GRIM-text/Shared/Lenses/README.md) and [Observatory](../ui/OBSERVATORY.md).
- [Runtime dataset services](../RUNTIME_DATASET_STREAM.md), [plugin API](../PLUGIN_API.md), and [perception provider](../../plugins/perception_plugin/README.md).
- [Orchestration implementation map](../multimodelorchestrationintegration.codoc.md), read alongside current provider documentation where older ownership descriptions are superseded.
- [Resource admission source](../../MMO/Core/ResourceCoordinator.cpp), [model-loading source](../../MMO/Core/ModelLoader.cpp), and [token-budget batching](../model/BATCHING.md).
- [Self-correction proposal](../model/plans/SELF_CORRECTION_RL.md), [RL loop status](../model/plans/RL_TRAINING_LOOP.md), and [Define-stage proposal](../model/plans/DEFINE_STAGE.md).
