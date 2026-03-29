# ExecutionBlock Structured Execution Cutover Flow

> Maintained Mermaid flow companion for `EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md`.
>
> Update this file in the **same change** whenever workstream ordering, ownership boundaries, payload transport, validation gates, runtime execution flow, inference selector flow, or deleted legacy paths change.

## Mandatory maintenance rule

This file must reflect the **current enforced cutover flow**. It is not allowed to lag behind reality or preserve deleted legacy branches for nostalgia.

Every qualifying change must:

- update the affected Mermaid diagram(s)
- remove deleted nodes/edges immediately
- adjust labels to reflect current owner modules and gates
- record the diagram delta in `EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md`

## Status legend

- **Current path** = expected live cutover sequencing / enforced runtime path
- **Gate** = validation or completion checkpoint
- **Deleted path** = must be removed from the diagrams once actually deleted in code

## Workstream execution flow

```mermaid
flowchart TD
    WS0[WS0<br/>ExecutionBlock deflation + split boundary]
    WS1[WS1<br/>Canonical execution metadata]
    WS2[WS2<br/>Canonical builder replaces __SLOTS__]
    WS3[WS3<br/>GRMT v11 compiled payload]
    WS5[WS5<br/>Phase1 remap + fragmentation rules]
    WS4[WS4<br/>Shared execution payload validator]
    WS6[WS6<br/>Row-local runtime orchestration]
    WS7[WS7<br/>Delete silent execution skips]
    WS8[WS8<br/>Align training + validation]
    WS3A[WS3A<br/>Tensor-backed selector/checkpoint ownership]
    WS9[WS9<br/>Inference + selector bind-or-mask]
    WS10[WS10<br/>Tests + smoke validation]
    DONE[Cutover verification<br/>GRMT rebuild + smoke runs]

    WS0 --> WS1 --> WS2 --> WS3 --> WS5 --> WS4 --> WS6 --> WS7 --> WS8 --> WS3A --> WS9 --> WS10 --> DONE
```

## Same-change maintenance flow

```mermaid
flowchart LR
    Change[Architecture-affecting change] --> Assess{Touches ownership, flow,<br/>validation, transport, or deletion status?}
    Assess -- No --> Minor[Optional note only]
    Assess -- Yes --> Plan[Update cutover plan]
    Assess -- Yes --> Codoc[Update living codoc]
    Assess -- Yes --> Diagram[Update flow companion]
    Plan --> Validate[Run relevant validation<br/>build/test/error checks]
    Codoc --> Validate
    Diagram --> Validate
    Validate --> Gate{Workstream gate satisfied?}
    Gate -- No --> Stay[Keep workstream open<br/>record remaining gaps]
    Gate -- Yes --> Advance[Advance to next workstream<br/>and update status]
```

## Structured execution cutover flow

```mermaid
flowchart TD
    subgraph Authoring[Canonical authoring + compilation]
        Record[StructuredExecutionRecord]
        Builder[ConceptExecutionSequenceBuilder]
        Compiled[CompiledStructuredExecutionPayload<br/>execution_active + token_exec_slots<br/>+ compiled_bootstrap_bindings<br/>+ teacher_steps<br/>+ slot_selection_targets]
        Record --> Builder --> Compiled
    end

    subgraph Dataset[Dataset + sequence transport]
        Sequence[TrainingSequence]
        Phase1[Phase1 remap<br/>BOS/EOS + padding +<br/>fragmentation rejection]
        SampleView[TrainingSampleView]
        Batch[BatchPayload]
        Compiled --> Sequence --> Phase1 --> SampleView --> Batch
    end

    subgraph Validation[Shared gate]
        Validator[ExecutionPayloadValidation]
        Batch --> Validator
    end

    subgraph Training[Training / validation runtime]
        LossPath[computeLossBatch]
        TrainPath[autogradTrainingStep]
        ExecBlock[ExecutionBlock row-local runtime]
        Validator --> LossPath
        Validator --> TrainPath
        TrainPath --> ExecBlock
    end

    subgraph Inference[Inference / generation]
        PromptMap[Explicit prompt slot map]
        Policy[DecodeTimeNumPolicy<br/>candidate set + fixed slot_features +<br/>null/ambiguity/bind-or-mask]
        Selector[DecodeTimeSlotSelectorLayer<br/>learned scoring over {NULL} ∪ L]
        Sampler[Sampler<br/>mask or bind <NUM> before sampling]
        PromptMap --> Policy
        Policy --> Selector --> Policy --> Sampler
    end

    Phase1 -. compiled metadata .-> Validator
    ExecBlock -. row-local memory/state .-> Policy
```

## Current artifact status

- **Workstream flow status:** Workstream 0 in progress
- **Maintenance flow status:** Active immediately
- **Structured execution flow status:** Target cutover contract established; initial ExecutionBlock deflation/split work has started

## Workstream 0 implementation note

```mermaid
flowchart LR
    Public[`execution_block_GPU.hpp`\npublic surface]
    Coord[`execution_block_GPU.cu`\npublic coordinator]
    Internal[`execution_block_internal.hpp`\nprivate shared internals]
    Mem[`execution_block_memory_stream_GPU.cu`\nmemory/bootstrap helpers]
    Data[`execution_block_data_stream_GPU.cu`\ndata/entropy helpers]

    Public --> Coord
    Coord --> Internal
    Coord --> Mem
    Coord --> Data
    Mem --> Internal
    Data --> Internal
```

## Diagram update checklist

When a future change lands, update the affected diagram(s) to reflect:

1. workstream status or ordering changes
2. renamed or split owner modules
3. new or removed payload fields
4. validator entry-point changes
5. runtime execution-routing changes
6. selector ownership or bind-or-mask flow changes
7. deleted fallback or compatibility branches
