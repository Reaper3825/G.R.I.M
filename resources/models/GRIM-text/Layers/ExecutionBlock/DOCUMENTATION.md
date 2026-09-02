# ExecutionBlock retirement boundary

The legacy differentiable register-machine execution path has been removed.
Shared forward now stops at contextual argument bootstrap seeding:

1. NumberEncoder materializes candidate keys for the execution-independent
   argument selector.
2. Encoder hidden states reach the configured bootstrap layer.
3. SlotSeedEncoder materializes contextual argument-slot seeds.
4. Shared forward continues through the remaining language-model layers
   without an execution gate, register bootstrap, op/arg/write/stop steps, or
   execution-memory readback.

The `ExecutionMemory` storage view and registry-owned ExecutionBlock parameter
tensors are deleted. The `execution_block_enabled` config field remains only
as migration plumbing and is not consumed by shared forward, training loss,
or inference.

The replacement architecture must introduce explicit
Generator/Agent → Candidate → Verifier decisions with
accept/reject/retry/repair outcomes. It must not restore teacher-forced
transition targets or route execution supervision through the data loader.
