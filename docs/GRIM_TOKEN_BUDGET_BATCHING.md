# Token-budget batching

Training uses `training.config.max_tokens_per_batch` in `ai_config.json` as
the padded-token overflow limit for each microbatch. The selected model's
compiled `max_seq_len` remains the context window; batching does not change it.

For the current router context of 1,024:

| Token budget | Maximum context rows per microbatch |
| --- | --- |
| 10,240 | 10 |
| 4,096 | 4 |
| 2,048 | 2 |
| 1,024 | 1 |

The initial local configuration uses 10,240, preserving the previous capacity.
Set a smaller budget to lower activation memory. The budget must accommodate at
least one complete context row. A non-multiple is accepted; spare slots smaller
than a complete row remain unused.

## Packing contract

- Count padded slots: `selected_row_count * compiled_context_window`.
- Preserve existing sequence/window boundaries, targets, padding, atom metadata,
  named spans, curriculum ordering, and attention behavior.
- Keep each selected example in its own context row. Do not concatenate examples,
  introduce attention masks between examples, or shorten the context window.
- Close a batch before the next complete context row would exceed the budget.
- Emit the final non-empty batch, even if it does not fill the budget. No valid
  tail rows are discarded.
- Finalized `batch_size` is a derived row capacity for runtime consumers, not an
  additional authored example-count limit. Migrate the old raw `batch_size`
  setting to `max_tokens_per_batch = old_batch_size * compiled_max_seq_len`.

With fixed context rows, short examples still incur padding. This change makes
the token limit authoritative; it does not add ragged attention or concatenate
short examples to eliminate padding.

## Accumulation and control

`gradient_accumulation_steps` still counts microbatches. Within an optimizer
window, primary loss means are weighted by their supervised-token counts
(valid gaps for the atom-identification model). The retrieval objective is
weighted separately by query count. Reported microbatch loss means stay unscaled.
The existing pre-clipping accumulation normalization is applied once.

An incomplete final optimizer window is flushed at the epoch boundary. Learning
rate step planning includes this final update using ceiling division per epoch.

`--batch-token-budget N` overrides the padded-token budget. The legacy explicit
`--batch-size N` and stability row-count override translate to `N * context` token
slots. Control messages append `max_tokens_per_batch`; older messages without it
retain their explicit row-count interpretation. The control server launches the
trainer with the token budget.

## Validation

Run `python scripts/test_token_budget_batching.py`. It compiles and executes only
focused CPU tests using an existing compiler. Coverage includes overflow, fixed
context width, padded-token accounting, row order, deterministic shuffling,
partial batches, payload targets/boundaries, config derivation, control-message
serialization, and accumulation weighting. Trainer builds and training execution
are separate checks and are not performed by this test runner.
