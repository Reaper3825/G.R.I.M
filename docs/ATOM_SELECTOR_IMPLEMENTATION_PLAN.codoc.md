---
name: Atom Selector Implementation Companion
overview: Living checklist companion to ATOM_SELECTOR_IMPLEMENTATION_PLAN.md. Tracks only what is implemented now vs what is still pending.
todos: []
isProject: false
---

# Atom Selector Implementation Companion Checklist

> Scope: checklist only.  
> Rule: only mark completed work that is already implemented in-tree.

## Workstream status

- [x] **W0 — Boundary lock**
  - [x] Selector/execution ownership boundary documented
  - [x] Config/parameter/data/autograd ownership boundaries documented
- [x] **W1 — Rip-out of execution-entangled selector ownership**
  - [x] Decode-time slot selector stack removed
  - [x] Legacy selector config/registry/checkpoint paths removed
  - [x] Execution no longer owns selector candidate logic
- [x] **W2 — Numeric meaning encoding input path**
  - [x] `NumberEncoderParameterTensors` registered via startup parameter registry
  - [x] `number_encoder_*` config leaves carried through HyperParameters grouping
  - [x] `BatchPayload` digit-place channels materialized and uploaded via `BatchDeviceBindings::d_atom_digit_*`
  - [x] `autograd::number_encode()` + `NumberEncoderGradFn` fused into shared forward
  - [x] NumberEncoder checkpoint save/load wiring present
- [ ] **W3 — Numeric supervision heads**
  - [ ] Next-atom metadata prediction head
  - [ ] Value reconstruction head
  - [ ] Digit-place contribution supervision head/loss
  - [ ] Contrastive numeric geometry objective
  - [ ] Explicit `digit_loss_mask` loss path for variable-length digit slots
- [ ] **W4 — Decode bridge into execution plumbing**
  - [ ] Selector output consumed by execution when active
  - [x] Temporary metadata -> `AtomTable` entry-id lookup -> placeholder fill bridge
  - [x] Removal of the unconditional stopgap; numeric placeholders are dynamically mask-or-bind
- [ ] **W5 — Parity + semantics validation**
  - [ ] End-to-end parity/semantics validation for numeric meaning + surface identity split

## Current in-tree evidence snapshot (checked items only)

- [x] NumberEncoder autograd node and forward integration exist:
  - [x] `resources/models/GRIM-text/Shared/TensorContract/GradFns/NumberEncoderGradFn.hpp`
  - [x] `resources/models/GRIM-text/Shared/TensorContract/GradFns/NumberEncoderGradFn.cu`
  - [x] `resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.cu`
- [x] NumberEncoder payload channels exist:
  - [x] `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`
  - [x] `resources/models/GRIM-text/Shared/Batching/BatchPayload.cu`
- [x] NumberEncoder parameter ownership exists:
  - [x] `resources/models/GRIM-text/training/Phases/Startup/Model/ParameterRegistry.hpp`
- [x] NumberEncoder checkpoint serialization exists:
  - [x] `resources/models/GRIM-text/Common/grim_model_serialization.cu`
  - [x] `resources/models/GRIM-text/Layers/Serialization/Serialization_save.cu`
  - [x] `resources/models/GRIM-text/Layers/Serialization/Serialization_load.cu`
  - [x] `resources/models/GRIM-text/Layers/Serialization/Serialization_validate.cu`

## Current pending-state note

- [x] Inference binds a model-confirmed terminal result into a session AtomTable and uses the numeric-meaning selector after no result is pending.
- [ ] W4 remains incomplete only for routing selector decisions into execution argument plumbing itself.
- [ ] Loss decomposition currently exposes `text_loss`, `mtp_loss`, and `execution_loss`; numeric supervision head channels are not yet wired, so W3 remains pending.

## Update rule

When implementation changes, update this checklist in the same change and only check boxes for behavior that is already present in code.
