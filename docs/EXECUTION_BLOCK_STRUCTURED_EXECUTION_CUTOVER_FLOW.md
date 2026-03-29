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

## Current enforced runtime flow

```mermaid
flowchart TD
    subgraph Training[Training / validation runtime today]
        Batch[Batch/runtime metadata]
        Train[AutogradTraining row loop]
        Scratch[ScratchBlock row-local atom view]
        Exec[ExecutionBlock executeStep / crossAttentionRead]
        Batch --> Train --> Scratch --> Exec
    end

    subgraph Inference[Inference / generation today]
        Prompt[Prompt slot map + decode token metadata]
        DecodeScratch[Decode-time ScratchBlock row-local atom view]
        DecodeExec[ExecutionBlock decode runtime]
        Sampler[Sampler]
        MaskNUM[Mask <NUM> before sampling<br/>until explicit selector exists]
        Prompt --> DecodeScratch --> DecodeExec
        Sampler --> MaskNUM
    end
```

## Current artifact status

- **Workstream flow status:** Workstream 1 complete; Workstream 2 next
- **Maintenance flow status:** Active immediately
- **Structured execution flow status:** Current enforced runtime reflects the locked Workstream 0 boundary with Workstream 1 metadata types canonicalized; compiled execution payload flows through TrainingSequence → TrainingSampleView → BatchPayload; future builder/format/selector workstreams are still pending

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

## Current runtime-side Workstream 0 hardening

```mermaid
flowchart TD
    Scratch[ScratchBlock buffers\nbatch-global detect state]
    Extract[extractRowLocalAtomView(...)]
    Train[AutogradTraining row loop]
    Decode[Inference decode step]
    Exec[executeStep(...)]
    Generate[Generation sampler]
    MaskNUM[Mask <NUM>\nuntil explicit selector exists]

    Scratch --> Extract
    Train --> Extract --> Exec
    Decode --> Extract --> Exec
    Generate --> MaskNUM
```

## Workstream 1 — metadata ownership and batch transport

```mermaid
flowchart TD
    subgraph Canonical[ExecutionMetadata.hpp — single definition site]
        TS[TeacherStep]
        CBB[CompiledBootstrapBinding]
        SST[SlotSelectionTarget]
        BLB[BootstrapLiteralBinding]
        SER[StructuredExecutionRecord]
        CSEP[CompiledStructuredExecutionPayload]
    end

    subgraph DataPipeline[Data pipeline — compiled payload transport]
        TSeq[TrainingSequence\nexecution_active + bindings + steps + targets]
        TSV[TrainingSampleView\nnon-owning pointers]
        BP[BatchPayload\nper-row arrays sized batch_size]
        Builder[buildBatchPayload\nPHASE 2 extract + PHASE 4 populate]
    end

    SER -->|sequence builder\nWS2| TSeq
    TSeq -->|getSample| TSV
    TSV --> Builder
    Builder --> BP

    subgraph Consumers[Downstream consumers]
        Loss[ComputeLossBatch\nteacher_steps]
        ExecBlock[ExecutionBlock\nslot_map + bindings]
        Validator[Shared validator\nWS4]
    end

    BP --> Loss
    BP --> ExecBlock
    BP --> Validator
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
