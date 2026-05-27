# ExecutionBlock

ExecutionBlock is a row-level register-machine side channel. Its memory is **not** timestep-aligned: the live `ExecutionMemory` after `executeStep(...)` is the row-final post-execution register state.

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

Then token `t` may read only `M_t`. Until such snapshots exist, ExecutionBlock remains a row-final augmentation and auxiliary-supervision mechanism, not an all-position hidden-state source.
