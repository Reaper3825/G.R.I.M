# ExecutionBlock

ExecutionBlock is a row-level register-machine side channel. Its memory is **not** timestep-aligned: the live `ExecutionMemory` after `executeStep(...)` is the row-final post-execution register state.

Durable trainable execution-block tensors are owned by
`StartupParameterRegistry::execution_block_parameters` and initialized by
`ParameterGroupRegistration::initializeExecutionBlockParameterTensors(...)`.
The old `ExecutionBlockLayer` runtime shell is deleted. Shared forward now
passes the registry-owned parameter bundle, explicit
`ExecutionBlockConstructionHP`, and runtime-owned diagnostics workspace
explicitly into execution-block free ops.

## Causal forward contract

For autoregressive LM training/inference, row-final execution memory may only be injected/read at the final valid token of that same row:

- row span is `payload.seq_lengths[b]`, not padded `payload.max_seq_len`;
- bootstrap reads only the valid row prefix `[b * max_seq_len, b * max_seq_len + seq_lengths[b])`;
- `executeStep(...)` builds context over that valid row span and injects only into `row_final = b * max_seq_len + seq_lengths[b] - 1` on the execution-layer output;
- `crossAttentionRead(...)` in shared causal forward is called on the **next layer input** (and later layer inputs), never on the same layer that wrote the memory, with `token_offset=row_final` and `row_tokens=1`;
- the returned delta is zero-padded back into the full tensor at `row_final` only.

Earlier token positions must never consume row-final `ExecutionMemory`. If token `t` needs execution memory, the architecture must first materialize a timestep-owned/prefix-owned memory snapshot for that `t`; reusing the row-final memory is a future-token leak.

## Why full-row readback is forbidden

A row can contain numeric atoms at positions greater than `t`. Bootstrapping `ExecutionMemory` from the full row and then applying readback to every hidden row position creates this illegal dependency:

$$
h_t \leftarrow h_t + f(M_{0:S}) \quad \text{where } M_{0:S} \text{ includes atoms from } u > t
$$

That violates autoregressive causality even if attention itself is causal, because the side-channel memory bypasses the attention mask. The safe row-level equations are:

$$
\hat{h}^{L}_{S-1} \leftarrow \hat{h}^{L}_{S-1} + \mathrm{result\_emb}
$$

$$
x^{L+1}_{S-1} \leftarrow x^{L+1}_{S-1} + f(M_{0:S})
$$

where $S = \text{seq\_lengths}[b]$, $S-1$ is the final valid token in the row, and $L$ is the execution layer. The execution layer may export the immediate step result, but persistent `ExecutionMemory` is first consumed on the next layer input.

## Fixed result slots

`execution_block_result_slot_mode == 1` is only legal when `execution_block_result_slot_index` equals the current row-final absolute token index. Any other fixed slot would inject row-final memory into an earlier/global token and must fail loudly.

## Future architecture

A fully per-token ExecutionBlock path requires explicit snapshots:

$$
M_t = \operatorname{Execute}(\text{atoms}_{u \le t}, h_{u \le t})
$$

Then prompt token `t` may read only `M_t`. Until such snapshots exist,
ExecutionBlock remains a row-final augmentation within a prompt, not an
all-prompt-position hidden-state source. Generated decode tokens are different:
the completed prompt-final memory is entirely in their causal past, so cached
decode windows may read that fixed session memory at downstream layers without
requiring per-prompt-token snapshots. They must not re-bootstrap it from the
decode window or rerun prompt execution.

Decode-time result emission is a separate, strict boundary. The write slot of a
step is exposed only when the learned stop controller classifies that completed
step as `STOP`. It is therefore an explicit terminal-step result, not a
"most-recent slot" heuristic. Max-step exhaustion exposes no result. The value
must be valid and finite before generation may unmask the matching numeric atom
placeholder, and the emitted value is registered in a session-owned AtomTable
so it remains a fully bound numeric token during later cached decode.
