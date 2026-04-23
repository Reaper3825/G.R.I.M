# ScratchBlock Reasoning Layer

Structured reasoning layer with atom detection (numbers, URLs, emails, paths, dates, code literals). See [Tokenizer.md](Tokenizer.md) for atom tokens.

## Buffer desync after `autograd::add(emb, pos_emb)`
After ScratchBlock forward, copy `ts->cached_embeddings` back to `ctx.embedding_tensor.data`. Otherwise Layer 0 receives stale pre-ScratchBlock data.

## Backward via dropout grad tap
- Set `grad_output_tap` on `DropoutGradFn` **before** calling `loss_tensor.backward()`.
- Pass the captured gradient into ScratchBlock backward.
- Do **not** check `has_grad()` on dropout outputs — it is always false for non-leaf tensors.
